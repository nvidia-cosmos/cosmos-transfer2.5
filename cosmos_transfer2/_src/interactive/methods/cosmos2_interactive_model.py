# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections
import math
import random
from contextlib import contextmanager
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torch.distributed._tensor.api import DTensor
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_transfer2._src.imaginaire.model import ImaginaireModel
from cosmos_transfer2._src.imaginaire.modules.denoiser_scaling import RectifiedFlowScaling
from cosmos_transfer2._src.imaginaire.modules.edm_sde import EDMSDE
from cosmos_transfer2._src.imaginaire.utils import log, misc
from cosmos_transfer2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_transfer2._src.imaginaire.utils.context_parallel import (
    broadcast,
    broadcast_split_tensor,
    cat_outputs_cp,
    find_split,
)
from cosmos_transfer2._src.imaginaire.utils.count_params import count_params
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.utils.ema import FastEmaModelUpdater
from cosmos_transfer2._src.imaginaire.utils.fsdp_helper import hsdp_device_mesh
from cosmos_transfer2._src.imaginaire.utils.optim_instantiate import get_base_scheduler
from cosmos_transfer2._src.interactive.configs.method_configs.config_cosmos2_interactive_base import (
    IS_PREPROCESSED_KEY,
    Cosmos2InteractiveModelConfig,
)
from cosmos_transfer2._src.interactive.utils.basic_utils import PRECISION_MAP
from cosmos_transfer2._src.interactive.utils.torch_future import clip_grad_norm_ as clip_grad_norm_impl_
from cosmos_transfer2._src.predict2.conditioner import DataType, GeneralConditioner
from cosmos_transfer2._src.predict2.configs.video2world.defaults.conditioner import (
    Video2WorldCondition,
)
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_transfer2._src.predict2.modules.denoiser_scaling import (
    RectifiedFlow_sCMWrapper as TrigFlow2RectifiedFlowScaling,
)
from cosmos_transfer2._src.predict2.networks.model_weights_stats import WeightTrainingStat
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import TextEncoder
from cosmos_transfer2._src.predict2.tokenizers.base_vae import BaseVAE
from cosmos_transfer2._src.predict2.utils.dtensor_helper import (
    DTensorFastEmaModelUpdater,
    broadcast_dtensor_model_states,
)


class Cosmos2InteractiveModel(ImaginaireModel):
    """
    Base class for distillation and interactive deployment of Cosmos2.5 series of models.

    Works with the distillation-specific trainer:
        cosmos_transfer2/_src/interactive/trainer/trainer_distillation.py
    """

    # ------------------------ Initialization & configuration ------------------------
    def __init__(self, config: Cosmos2InteractiveModelConfig):
        super().__init__()

        # Statically type the config attribute so downstream accessors are well-typed.
        self.config: Cosmos2InteractiveModelConfig = config

        self.precision = PRECISION_MAP[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.info(f"Setting precision to {self.precision}")

        self.init_parallelism()
        self.build_model()
        self.setup_data_key()
        self.setup_sampler_and_scheduler()
        self.setup_loss_specs()

    def setup_sampler_and_scheduler(self) -> None:
        self.sigma_data = self.config.sigma_data
        self.sde: EDMSDE = lazy_instantiate(self.config.sde)  # type: ignore
        self.scaling = RectifiedFlowScaling(
            self.sigma_data, self.config.rectified_flow_t_scaling_factor, self.config.rectified_flow_loss_weight_uniform
        )
        self.trigflow_scaler = TrigFlow2RectifiedFlowScaling(self.sigma_data)

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key
        self.input_caption_key = self.config.input_caption_key

    def setup_loss_specs(self) -> None:
        self.loss_reduce = getattr(self.config, "loss_reduce", "mean")
        self.loss_scale = getattr(self.config, "loss_scale", 1.0)
        assert self.loss_reduce in ["mean", "sum"]
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        if self.config.multiply_noise_by_video_len:
            self.video_noise_multiplier = math.sqrt(self.config.state_t)
        else:
            self.video_noise_multiplier = 1.0

    def init_parallelism(self) -> None:
        """Set up FSDP device mesh and data-parallel world size."""
        # parallelism for model parameters
        if self.config.fsdp_shard_size > 1:
            self.fsdp_device_mesh = hsdp_device_mesh(
                sharding_group_size=self.config.fsdp_shard_size,
            )
        else:
            self.fsdp_device_mesh = None

        # parallelism for data/activations
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

    # ------------------------ Model construction & apply FSDP ------------------------
    @misc.timer("Cosmos2InteractiveModel: build_net")
    def build_net(self, net_config_dict):
        """Instantiate a single network from config and build FSDP for it if needed."""

        with torch.device("meta"):
            net = lazy_instantiate(net_config_dict)

        self._param_count = count_params(net, verbose=False)

        if self.fsdp_device_mesh:
            net.fully_shard(mesh=self.fsdp_device_mesh)  # type: ignore
            net = fully_shard(net, mesh=self.fsdp_device_mesh, reshard_after_forward=True)

        net.to_empty(device="cuda")  # type: ignore
        # IMPORTANT: model init should not depend on current tensor shape, or it can handle DTensor shape.
        net.init_weights()  # type: ignore

        if self.fsdp_device_mesh:
            # recall model weight init; be careful for buffers!
            broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
            for name, param in net.named_parameters():  # type: ignore
                assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    @misc.timer("Cosmos2InteractiveModel: build_model")
    def build_model(self):
        """
        Build networks and the parameter-less conditioner.
        """
        # Text encoder
        self.text_encoder = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self.text_encoder = TextEncoder(self.config.text_encoder_config)

        # Negative text prompt embedding (optional): if distilling a teacher, neg embed is used
        # for teacher cfg during distillation.
        self.neg_embed = (
            easy_io.load(self.config.neg_embed_path) if getattr(self.config, "neg_embed_path", "") else None
        )

        # Tokenizer
        self.tokenizer: BaseVAE = lazy_instantiate(self.config.tokenizer)  # type: ignore
        assert self.tokenizer.latent_ch == self.config.state_ch, (
            f"latent_ch {self.tokenizer.latent_ch} != state_shape {self.config.state_ch}"
        )

        # Diffusion Network (student net, aka generator)
        self.net = self.build_net(self.config.net)

        # EMA of the student net
        if self.config.ema.enabled:
            self.net_ema = self.build_net(self.config.net)
            self.net_ema.requires_grad_(False)

            if self.fsdp_device_mesh:
                self.net_ema_worker = DTensorFastEmaModelUpdater()
            else:
                self.net_ema_worker = FastEmaModelUpdater()
            s = self.config.ema.rate
            self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        # Conditioner (encapsulates text and video frame conditions in one object)
        self.conditioner: GeneralConditioner = lazy_instantiate(self.config.conditioner)  # type: ignore

        # Optional condition postprocessor, e.g.:
        # - control input encoding in Transfer2
        # - action condition processing in action-conditioned Predict2
        # - camera-data processing in camera-conditioned Predict2, etc.
        self.condition_postprocessor: Any = (
            lazy_instantiate(self.config.condition_postprocessor)
            if getattr(self.config, "condition_postprocessor", None)
            else None
        )
        log.info(f"\n\n==============config condition_postprocessor: {self.config.condition_postprocessor}")
        log.info(f"\n\n==============condition_postprocessor: {self.condition_postprocessor}")

    # ------------------------ Optimizer & scheduler utils ------------------------
    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        """Creates the optimizer and scheduler for the model."""
        # instantiate the net optimizer
        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": net_optimizer}

        net_scheduler = get_base_scheduler(net_optimizer, self, scheduler_config)
        self.scheduler_dict = {"net": net_scheduler}

        return net_optimizer, net_scheduler

    def is_student_phase(self, iteration: int) -> bool:
        """Return True when we are in the student update phase.
        Defaults to identity. To be overridden by child classes if alternate
        student/critic update phases are needed.
        """
        return True

    def get_effective_iteration(self, iteration: int):
        """Return effective iteration index used for EMA scheduling (base: identity)."""
        return iteration

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """Return the list of optimizers to step at this iteration (base: student only)."""
        return [self.optimizer_dict["net"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Return the list of LR schedulers to step at this iteration (base: student only)."""
        return [self.scheduler_dict["net"]]

    def optimizers_schedulers_step(self, grad_scaler: torch.cuda.amp.GradScaler, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            grad_scaler.step(optimizer)
            grad_scaler.update()

        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def optimizers_zero_grad(self, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad()

    # ------------------------ Distributed / context parallel ------------------------
    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def sync(self, tensor, condition):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            tensor = broadcast(tensor, cp_group)
        return tensor

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Video2WorldCondition,
        uncondition: Video2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        time_B_T: torch.Tensor,
    ):
        """
        Broadcast and split the input data and condition for model parallelism (context parallelism only).
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()

        # For video data, broadcast/split along the temporal dimension when using context parallelism.
        if cp_size > 1:
            self.net.enable_context_parallel(cp_group)
            if condition.is_video:
                # Perform spatial split only when it's required, i.e. temporal split is not enough.
                # Refer to "find_split" definition for more details.
                use_spatial_split = cp_size > x0_B_C_T_H_W.shape[2] or x0_B_C_T_H_W.shape[2] % cp_size != 0
                after_split_shape = find_split(x0_B_C_T_H_W.shape, cp_size) if use_spatial_split else None
                if use_spatial_split:
                    x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C T H W -> B C (T H W)")
                    if epsilon_B_C_T_H_W is not None:
                        epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "B C T H W -> B C (T H W)")
                x0_B_C_T_H_W = broadcast_split_tensor(
                    x0_B_C_T_H_W, seq_dim=2, process_group=cp_group
                )  # actual shape is [B, C, T * H * W] now
                epsilon_B_C_T_H_W = broadcast_split_tensor(
                    epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group
                )  # actual shape is [B, C, T * H * W] now
                if use_spatial_split:  # reshape back to [B, C, T', H', W'] format
                    x0_B_C_T_H_W = rearrange(
                        x0_B_C_T_H_W, "B C (T H W) -> B C T H W", T=after_split_shape[0], H=after_split_shape[1]
                    )
                    if epsilon_B_C_T_H_W is not None:
                        epsilon_B_C_T_H_W = rearrange(
                            epsilon_B_C_T_H_W,
                            "B C (T H W) -> B C T H W",
                            T=after_split_shape[0],
                            H=after_split_shape[1],
                        )
                if time_B_T is not None:
                    assert time_B_T.ndim == 2, "time_B_T should be 2D tensor"
                    if time_B_T.shape[-1] == 1:  # single sigma / time is shared across all frames
                        time_B_T = broadcast(time_B_T, cp_group)
                    else:  # different sigma for each frame
                        time_B_T = broadcast_split_tensor(time_B_T, seq_dim=1, process_group=cp_group)
                if condition is not None:
                    condition = condition.broadcast(cp_group)
                if uncondition is not None:
                    uncondition = uncondition.broadcast(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, uncondition, epsilon_B_C_T_H_W, time_B_T

    # ------------------------ Checkpointing helpers ------------------------
    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        return net_state_dict

    def model_dict(self) -> Dict[str, Any]:
        return {"net": self.net}

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Load weights for the student net (and its EMA), ignoring any extra heads."""
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)

            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )
                return _IncompatibleKeys(
                    missing_keys=reg_results.missing_keys + ema_results.missing_keys,
                    unexpected_keys=reg_results.unexpected_keys + ema_results.unexpected_keys,
                )

            return reg_results

        log.critical("load model in non-strict mode")
        log.critical(non_strict_load_model(self.net, _reg_state_dict), rank0_only=False)  # type: ignore
        if self.config.ema.enabled:
            log.critical("load ema model in non-strict mode")
            log.critical(non_strict_load_model(self.net_ema, _ema_state_dict), rank0_only=False)  # type: ignore

    # ------------------------ EMA management ------------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    @contextmanager
    def ema_scope(self, context: str | None = None, is_cpu: bool = False):
        """Temporarily swap the student net's weights to EMA weights (FSDP-aware) inside a context.

        This is typically used for validation or inference with EMA weights without affecting
        the ongoing training weights. After the context exits, the original training weights
        are restored.
        """
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

    # ------------------------ Helper methods and utils for callbacks ------------------------
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.sigma_data)

    def model_param_stats(self) -> Dict[str, int]:
        """Return simple statistics about the student net parameters."""
        return {"total_learnable_param_num": self._param_count}

    def get_video_height_width(self) -> Tuple[int, int]:
        return VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]

    def get_video_latent_height_width(self) -> Tuple[int, int]:
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]
        return height // self.tokenizer.spatial_compression_factor, width // self.tokenizer.spatial_compression_factor

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    def _configure_video_condition(
        self,
        condition: Video2WorldCondition,
        latent_state: torch.Tensor,
        num_conditional_frames: torch.Tensor,
    ) -> Video2WorldCondition:
        """Apply video2world conditioning (frame masks etc.) to a condition object."""
        return condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=0,  # will not take effect since num_conditional_frames is provided
            random_max_num_conditional_frames=0,  # will not take effect since num_conditional_frames is provided
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=None,
        )

    def _apply_vid2vid_conditioning(
        self,
        net_state_in_B_C_T_H_W: torch.Tensor,
        c_noise_B_1_T_1_1: torch.Tensor,
        condition: Video2WorldCondition,
        num_channels: int,
        noise_level_B_1_T_1_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Apply vid2vid-style conditioning to the network input and optionally adjust noise level for conditional frames.

        This performs two steps when the batch is a video:
        1) Replace the chosen frames with GT frames as conditioning frames. The conditioning frame IDs are sampled from data batch and stored in the condition object.
        2) If enabled, adjust the effective noise level for those conditional frames to be very low (clean).

        If the batch is not video, the inputs are returned unchanged and the mask is None.
        """
        if not condition.is_video:  # This flag indicates the *data* batch is a video, not the condition object.
            return net_state_in_B_C_T_H_W, c_noise_B_1_T_1_1, None

        condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
        condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, num_channels, 1, 1, 1).type_as(
            net_state_in_B_C_T_H_W
        )
        net_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + net_state_in_B_C_T_H_W * (
            1 - condition_video_mask
        )

        if self.config.use_clean_cond_timesteps:
            noise_level_cond_B_1_T_1_1 = torch.arctan(
                torch.ones_like(noise_level_B_1_T_1_1) * self.config.sigma_conditional
            )
            _, _, _, c_noise_cond_B_1_T_1_1 = self.trigflow_scaler(noise_level_cond_B_1_T_1_1)
            condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
            c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                1 - condition_video_mask_B_1_T_1_1
            )

        return net_state_in_B_C_T_H_W, c_noise_B_1_T_1_1, condition_video_mask

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        is_image = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image else self.input_data_key
        if isinstance(self.net, WeightTrainingStat):
            if is_image:
                self.net.accum_image_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size
            else:
                self.net.accum_video_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        if not self.config.grad_clip:
            max_norm = 1e10

        def _clean_and_clip(params):
            for param in params:
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            return clip_grad_norm_impl_(
                params,
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )

        if getattr(self, "net_fake_score", None):
            _clean_and_clip(self.net_fake_score.parameters())
        if getattr(self, "net_discriminator_head", None):
            _clean_and_clip(self.net_discriminator_head.parameters())
        return _clean_and_clip(self.net.parameters()).cpu()

    # ------------------------ Trainer hooks ------------------------
    def on_before_zero_grad(
        self,
        _optimizer: torch.optim.Optimizer,
        _scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        """
        Per-iteration hook run before optim.zero_grad().

        - In critic phase, sync low-precision fake_score / discriminator params from their
          FP32 master weights via `update_master_weights`, since the main low-precision
          callback only handles the student net.
        - In student phase, leave critic nets untouched and update the EMA weights
          (`net_ema`) from the current student weights (`net`) using `ema_beta`.
        """
        # The main net and its master weights are handled by the low precision callback.
        # Manually update the fake score and discriminator if needed.
        if not self.is_student_phase(iteration):
            from cosmos_transfer2._src.interactive.utils.misc import update_master_weights

            if getattr(self, "net_fake_score", None):
                update_master_weights(self.optimizer_dict["fake_score"])
            if getattr(self, "net_discriminator_head", None):
                update_master_weights(self.optimizer_dict["discriminator"])
            return

        # Student phase: update EMA for the student net
        if self.config.ema.enabled:
            ema_beta = self.ema_beta(self.get_effective_iteration(iteration))
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()

        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)
        if getattr(self, "net_teacher", None):
            self.net_teacher = self.net_teacher.to(memory_format=memory_format, **self.tensor_kwargs)
        if getattr(self, "net_fake_score", None):
            self.net_fake_score = self.net_fake_score.to(memory_format=memory_format, **self.tensor_kwargs)
        if getattr(self, "net_discriminator_head", None):
            self.net_discriminator_head = self.net_discriminator_head.to(
                memory_format=memory_format, **self.tensor_kwargs
            )

        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:  # compatible with old config
            if torch.__version__ < "2.3":
                log.warning(
                    "torch.compile in Pytorch version older than 2.3 doesn't work well with activation checkpointing.\n"
                    "It's very likely there will be no significant speedup from torch.compile.\n"
                    "Please use at least 24.04 Pytorch container, or imaginaire4:v7 container."
                )
            # Increasing cache size.
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # dynamic=False means that a separate kernel is created for each shape.
            self.net = torch.compile(self.net, dynamic=False, disable=not self.config.use_torch_compile)
            if getattr(self, "net_teacher", None):
                self.net_teacher = torch.compile(
                    self.net_teacher, dynamic=False, disable=not self.config.use_torch_compile
                )
            if getattr(self, "net_fake_score", None):
                self.net_fake_score = torch.compile(
                    self.net_fake_score, dynamic=False, disable=not self.config.use_torch_compile
                )
            if getattr(self, "net_discriminator_head", None):
                self.net_discriminator_head = torch.compile(
                    self.net_discriminator_head, dynamic=False, disable=not self.config.use_torch_compile
                )

    # ------------------------ Data loading ------------------------
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor], set_video_condition: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Video2WorldCondition, Video2WorldCondition]:
        """
        Prepare raw/latent states and (un)condition objects from an input batch.

        - Supports both image and video batches, inferred from `input_image_key` / `input_data_key`.
        - Encodes the raw frames into latent space using the tokenizer.
        - Builds condition and uncondition. Uncondition is using empty or negative prompt (if provided)
        - When 'set_video_condition' is True, applies video2world temporal conditioning by sampling a
          shared number of conditional frames per example and setting the corresponding masks.
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        data_type = DataType.IMAGE if is_image_batch else DataType.VIDEO

        # Video tokens
        raw_state = data_batch[input_key]  # raw video
        latent_state = self.encode(raw_state).contiguous().float()  # latent state (i.e. video tokens)

        # Text embeddings: reuse precomputed ones if provided (inference scripts often pass them),
        # otherwise compute online when enabled.
        if "t5_text_embeddings" in data_batch:
            if "t5_text_mask" not in data_batch:
                data_batch["t5_text_mask"] = torch.ones(data_batch["t5_text_embeddings"].shape[:2], device="cuda")
        elif self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            caption_key = getattr(self.config, "input_caption_key", self.input_caption_key)
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, caption_key)
            # For legacy reason it's called t5_text_embeddings. Supports CR1 embeddings too.
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        # Condition object: including the text embeddings and neg text embeddings (if provided)
        if self.neg_embed is not None:
            data_batch["neg_t5_text_embeddings"] = repeat(
                self.neg_embed.to(**self.tensor_kwargs),
                "l d -> b l d",
                b=data_batch["t5_text_embeddings"].shape[0],
            )
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(data_type)
        uncondition = uncondition.edit_data_type(data_type)

        # Update Condition object: set video condition masks etc. to support video2world mode.
        if set_video_condition:
            num_conditional_frames = self._sample_num_conditional_frames(latent_state, data_batch)
            condition = self._configure_video_condition(
                condition=condition,
                latent_state=latent_state,
                num_conditional_frames=num_conditional_frames,
            )
            uncondition = self._configure_video_condition(
                condition=uncondition,
                latent_state=latent_state,
                num_conditional_frames=num_conditional_frames,
            )

        # Apply optional condition postprocessor (e.g., add the control inputs from
        # data batch to the condition object in Transfer2)
        if self.condition_postprocessor is not None:
            condition, uncondition = self.condition_postprocessor(
                model=self,
                condition=condition,
                uncondition=uncondition,
                latent_state=latent_state,
                data_batch=data_batch,
            )

        return raw_state, latent_state, condition, uncondition

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_data_key should be present in the data_batch."
        )
        return is_image

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str | None = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str | None = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all((data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

            expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
            original_length = data_batch[input_key].shape[2]
            assert original_length == expected_length, (
                f"Input video length doesn't match expected length specified by state_t: {original_length} != {expected_length}"
            )

    def _sample_num_conditional_frames(
        self, latent_state: torch.Tensor, data_batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Sample a random number of conditional frames for the video batch given the provided probabilities.

        Args:
            latent_state (torch.Tensor): latent state (i.e. video tokens)
            data_batch (dict[str, torch.Tensor]): data batch

        Returns:
            torch.Tensor: number of conditional frames
        """
        B, _, T, _, _ = latent_state.shape
        if T == 1:
            return torch.zeros(B, dtype=torch.int32)

        num_cf_from_batch = data_batch.get("num_conditional_frames", None)
        if num_cf_from_batch is not None:
            if isinstance(num_cf_from_batch, torch.Tensor):
                if num_cf_from_batch.ndim == 0:
                    return torch.ones(B, dtype=torch.int32) * int(num_cf_from_batch.item())
                return num_cf_from_batch.to(dtype=torch.int32, device="cpu")
            return torch.ones(B, dtype=torch.int32) * int(num_cf_from_batch)

        if getattr(self.config, "conditional_frames_probs", None):
            frames_options = list(self.config.conditional_frames_probs.keys())
            weights = list(self.config.conditional_frames_probs.values())
            return torch.tensor(
                random.choices(frames_options, weights=weights, k=B),
                dtype=torch.int32,
            )

        return torch.randint(
            self.config.min_num_conditional_frames,
            self.config.max_num_conditional_frames + 1,
            size=(B,),
            dtype=torch.int32,
        )

    # ------------------------ Core denoise function ------------------------
    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        noise_level: torch.Tensor,
        condition: Video2WorldCondition,
        **kwargs,
    ) -> DenoisePrediction:
        """
        Network forward to denoise the input noised data given noise level, and condition.

        The math formulation follows the generalized EDM formulation using the c_in, c_out, c_skip coefficients.
        By setting these coefficients differently (see self.scaling), this denoiser function supports different
        parameterizations of the noise level: EDM, Rectified Flow, TrigFlow, etc.

        Args:
            xt_B_C_T_H_W (torch.Tensor): The input noised-input data.
            noise_level (torch.Tensor): The noise level under the chosen parameterization. Generalized to support different parameterizations:
                - EDM: this is the sigma_B_T. Range from 0 to 1.
                - Rectified Flow: this is the normalized 'time' parameter. Range from 0 to 1.
                - TrigFlow: this is the angle (also called 'time') parameter in the trigonometry parameterization. Range from 0 to pi/2.
            condition (Video2WorldCondition): contains the GT frames, text embeddings, the conditiong frames & frame masks, etc.

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred) or velocity prediction (v_pred) if necessary.
        """
        B, C, T, H, W = xt_B_C_T_H_W.shape

        # Prep noise level param for network input
        if noise_level.ndim == 1:
            noise_level_B_T = repeat(noise_level, "b -> b t", t=T)
        elif noise_level.ndim == 2:
            noise_level_B_T = noise_level
        else:
            raise ValueError(f"noise_level shape {noise_level.shape} is not supported")
        noise_level_B_1_T_1_1 = rearrange(noise_level_B_T, "b t -> b 1 t 1 1")

        # Convert noise level time to EDM-formulation coefficients
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.trigflow_scaler(
            noise_level_B_1_T_1_1
        )

        # Prepare noised signal for network input (aka preconditioning)
        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        # Apply vid2vid conditioning: replace chosen frames with GT frames as conditioning frames
        net_state_in_B_C_T_H_W, c_noise_B_1_T_1_1, condition_video_mask = self._apply_vid2vid_conditioning(
            net_state_in_B_C_T_H_W=net_state_in_B_C_T_H_W,
            c_noise_B_1_T_1_1=c_noise_B_1_T_1_1,
            condition=condition,
            num_channels=C,
            noise_level_B_1_T_1_1=noise_level_B_1_T_1_1,
        )

        # Denoiser net forward pass
        net_out = self.net(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Match model precision to avoid dtype mismatch with FSDP
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **self.tensor_kwargs
            ),  # Keep FP32 for numerical stability in timestep embeddings
            **condition.to_dict(),
        )
        net_output_B_C_T_H_W = net_out.to(dtype=xt_B_C_T_H_W.dtype)

        # Reconstruction of x0 following generalized EDM formulation
        # Note: compatible with Rectified Flow if the c_* coefficients are set to rectified flow scaling
        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W

        # Replace conditioning frames in the output with GT frames to zero out loss and prevent gradients on them.
        if condition.is_video and self.config.replace_cond_output_with_gt:
            gt_frames = condition.gt_frames.type_as(x0_pred_B_C_T_H_W)
            condition_video_mask = condition_video_mask.type_as(x0_pred_B_C_T_H_W)
            x0_pred_B_C_T_H_W = gt_frames * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

        return DenoisePrediction(x0=x0_pred_B_C_T_H_W)

    def forward(self, xt, t, condition: Video2WorldCondition) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            t (torch.Tensor): The time parameter in trigflow parameterization representing noise level.
            condition (Video2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        return self.denoise(xt, t, condition, net_type="student")

    # ------------------------ Training step implementation ------------------------
    def draw_training_time_and_epsilon_student(
        self, x0_size: torch.Size, condition: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw the noise level and convert to its corresponding trigflow time parameter.
        Also return a unit gaussian noise to be added to the data during training.
        """
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device="cuda")
        sigma_B = self.sde.sample_t(batch_size).to(device="cuda")
        sigma_B_T = repeat(sigma_B, "b -> b t", t=x0_size[2])  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_T = sigma_B_T * multiplier
        time_B_T = torch.arctan(sigma_B_T / self.sigma_data)
        return time_B_T.double(), epsilon

    def draw_training_time_critic(self, x0_size: torch.Size, condition: Any) -> torch.Tensor:
        """
        Draw the noise level and convert to its corresponding trigflow time parameter.
        The noise is drawn from the teacher model's noise schedule (typically with timestep shift applied).
        """
        batch_size = x0_size[0]
        if self.config.timestep_shift > 0:
            sigma_B = torch.rand(batch_size).to(device="cuda").double()
            sigma_B = self.config.timestep_shift * sigma_B / (1 + (self.config.timestep_shift - 1) * sigma_B)
            sigma_B_T = repeat(sigma_B, "b -> b t", t=x0_size[2])
            time_B_T = torch.arctan(sigma_B_T / (1 - sigma_B_T))
            return time_B_T
        sigma_B = self.sde_D.sample_t(batch_size).to(device="cuda")
        sigma_B_T = repeat(sigma_B, "b -> b t", t=x0_size[2])  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_T = sigma_B_T * multiplier
        time_B_T = torch.arctan(sigma_B_T / self.config.sigma_data)
        return time_B_T.double()

    def single_train_step(self, data_batch: dict[str, torch.Tensor], iteration: int) -> None:
        """
        Performs a single training step for the diffusion model.

        This method is model-agnostic and delegates the core losses to
        `training_step_generator` and `training_step_critic`, which should
        be implemented in individual method classes.
        """
        pass

    # ------------------------ Inference & sampling ------------------------
    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        net_type: str = "student",
    ) -> Callable:
        """
        Assemble the denoise function and the condition from the data batch into a callable function.
        The callable function will be used iteratively in the sampling process during inference.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()

        num_conditional_frames = data_batch.get("num_conditional_frames", 0)
        _, x0, condition, _ = self.get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=x0,  # x0: clean, tokenized latent video
            random_min_num_conditional_frames=0,  # inference time will use the fixed num_conditional_frames
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        if condition.is_video and cp_size > 1:
            condition = condition.broadcast(cp_group)

        # For inference, ensure context parallel is disabled when parallel_state is not initialized.
        if not parallel_state.is_initialized():
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        @torch.no_grad()
        def x0_fn(noise_x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            raw_x0 = self.denoise(noise_x, time, condition, net_type=net_type).x0
            return raw_x0

        return x0_fn

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        num_steps: int = 4,
        init_noise: torch.Tensor = None,
        net_type: str = "student",
        # mid_t: List[float] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            num_steps (int): number of steps for the diffusion process
        """
        del kwargs
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]  # type: ignore

        x0_fn = self.get_x0_fn_from_batch(data_batch, net_type=net_type)

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )

        if self.net.is_context_parallel_enabled:  # type: ignore
            init_noise = broadcast_split_tensor(init_noise, seq_dim=2, process_group=self.get_context_parallel_group())

        # Sampling steps
        x = init_noise.to(torch.float64)
        ones = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        if num_steps > len(self.config.selected_sampling_time):
            log.warning(
                f"num_steps {num_steps} greater than selected_sampling_time {len(self.config.selected_sampling_time)}. Computing the sampling steps based on provided number of steps and timestep shift {self.config.timestep_shift}."
            )
            t_steps = self.get_trigflow_sampling_timesteps(num_steps, self.config.timestep_shift)
        else:
            t_steps = self.config.selected_sampling_time[:num_steps] + [0]

        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            # log.info(f"=============t_cur: {t_cur}, t_next: {t_next}")
            x = x0_fn(x.float(), t_cur * ones).to(torch.float64)
            if t_next > 1e-5:
                x = math.cos(t_next) * x / self.config.sigma_data + math.sin(t_next) * init_noise
        samples = x.float()
        if self.net.is_context_parallel_enabled:  # type: ignore
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())
        return torch.nan_to_num(samples)

    @staticmethod
    def get_trigflow_sampling_timesteps(num_steps: int, timestep_shift: float = 5.0) -> list[float]:
        """
        Given number of inference steps and a timestep shift (which shifts the timesteps towards higher noise levels),
        create a list of trigflow timesteps.
        """

        # Create evenly-spaced grid for timesteps in Rectified Flow
        step_size = 1 / num_steps
        timesteps_rf = torch.linspace(step_size, 1.0, num_steps)
        # Apply shift: t_shifted = shift * t_rf / (1 + (shift - 1) * t_rf)
        timesteps_rf_shifted = timestep_shift * timesteps_rf / (1 + (timestep_shift - 1) * timesteps_rf)
        # In Rectified Flow, sigma = t_rf / (1 - t_rf)
        sigma = timesteps_rf_shifted / (1 - timesteps_rf_shifted)
        # sigma is consistent between rf and trigflow. In TrigFlow, sigma = tan(t_trigflow)
        timesteps_trigflow = torch.arctan(sigma)
        # Append 0 in the end for the last denoising hop
        return timesteps_trigflow.numpy().tolist()[::-1] + [0]
