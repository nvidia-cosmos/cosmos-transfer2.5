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

from dataclasses import field
from typing import Callable, Dict, Optional, Tuple, cast

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed import get_process_group_ranks

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.context_parallel import broadcast_split_tensor
from cosmos_transfer2._src.predict2.conditioner import DataType
from cosmos_transfer2._src.predict2.models.text2world_model_rectified_flow import IS_PREPROCESSED_KEY
from cosmos_transfer2._src.predict2.models.video2world_model_rectified_flow import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import (
    ConditionLocationList,
)
from cosmos_transfer2._src.predict2_multiview.models.multiview_vid2vid_model_rectified_flow import (
    compute_empty_and_negative_text_embeddings,
    inplace_compute_text_embeddings_online_multiview,
    training_step_multiview,
)
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import (
    ControlVideo2WorldModelRectifiedFlow,
    ControlVideo2WorldRectifiedFlowConfig,
)
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.conditioner import (
    MultiViewControlVideo2WorldCondition,
)

CONTROL_WEIGHT_KEY = "control_weight"


@attrs.define(slots=False)
class MultiviewControlVideo2WorldRectifiedFlowConfig(ControlVideo2WorldRectifiedFlowConfig):
    train_base_model: bool = False
    min_num_conditional_frames_per_view: int = 1
    max_num_conditional_frames_per_view: int = 2
    train_sample_views_range: Tuple[int, int] | None = None
    condition_locations: ConditionLocationList = field(default_factory=lambda: ConditionLocationList([]))
    state_t: int = 0
    view_condition_dropout_max: int = 0
    online_text_embeddings_as_dict: bool = True  # For backward compatibility with old experiments
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames


class MultiviewControlVideo2WorldModelRectifiedFlow(ControlVideo2WorldModelRectifiedFlow):
    def __init__(self, config: MultiviewControlVideo2WorldRectifiedFlowConfig, *args, **kwargs):
        self.is_new_training = True
        self.copy_weight_strategy = config.copy_weight_strategy
        self.hint_keys = []
        for key in config.hint_keys.split("_"):
            if key == "hdmap":
                self.hint_keys.append("hdmap_bbox")
            elif key != "bbox":
                self.hint_keys.append(key)
        self.hint_keys = [f"control_input_{key}" for key in self.hint_keys]
        super(ControlVideo2WorldModelRectifiedFlow, self).__init__(config, *args, **kwargs)
        if config.train_base_model:
            log.warning("MultiviewControlVideo2WorldModelRectifiedFlow: Training base model")
        else:
            log.warning("MultiviewControlVideo2WorldModelRectifiedFlow: Freezing base model")
            self.freeze_base_model()
        log.info(self.net, rank0_only=True)
        self.state_t = config.state_t
        self.empty_string_text_embeddings = None
        self.neg_text_embeddings = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            compute_empty_and_negative_text_embeddings(self)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.state_t)
        if n_views > 1:
            return self.encode_cp(state)
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = super().encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n_views = latent.shape[2] // self.state_t
        if n_views > 1:
            return self.decode_cp(latent)
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = super().decode(latent)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state

    @torch.no_grad()
    def encode_cp(self, state: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.state_t)
        if n_views > cp_size:
            raise ValueError(
                f"n_views must be less than or equal to cp_size, got n_views={n_views} and cp_size={cp_size}"
            )
        state_V_B_C_T_H_W = rearrange(state, "B C (V T) H W -> V B C T H W", V=n_views)
        state_input = torch.zeros((cp_size, *state_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        state_input[0:n_views] = state_V_B_C_T_H_W
        local_state_V_B_C_T_H_W = broadcast_split_tensor(state_input, seq_dim=0, process_group=cp_group)
        local_state = rearrange(local_state_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        encoded_state = super().encode(local_state)
        encoded_state_list = [torch.empty_like(encoded_state) for _ in range(cp_size)]
        dist.all_gather(encoded_state_list, encoded_state, group=cp_group)
        encoded_state = torch.cat(encoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return encoded_state

    @torch.no_grad()
    def decode_cp(self, latent: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = latent.shape[2] // self.state_t
        latent_V_B_C_T_H_W = rearrange(latent, "B C (V T) H W -> V B C T H W", V=n_views)
        latent_input = torch.zeros((cp_size, *latent_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        latent_input[0:n_views] = latent_V_B_C_T_H_W
        local_latent_V_B_C_T_H_W = broadcast_split_tensor(latent_input, seq_dim=0, process_group=cp_group)
        local_latent = rearrange(local_latent_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        decoded_state = super().decode(local_latent)
        decoded_state_list = [torch.empty_like(decoded_state) for _ in range(cp_size)]
        dist.all_gather(decoded_state_list, decoded_state, group=cp_group)
        decoded_state = torch.cat(decoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return decoded_state

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return training_step_multiview(self, data_batch, iteration)

    def inplace_compute_text_embeddings_online(self, data_batch: dict[str, torch.Tensor]) -> None:
        inplace_compute_text_embeddings_online_multiview(self, data_batch)

    def compute_text_embeddings_online(self, data_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Temporary function from transfer2-mv dependency transition from diffusion/v2 to predict2
        return self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: MultiViewControlVideo2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        n_views = x0_B_C_T_H_W.shape[2] // self.state_t
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        reshape_sigma_B_T = False
        if sigma_B_T is not None:
            assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
            if sigma_B_T.shape[-1] != 1:
                assert sigma_B_T.shape[-1] % n_views == 0, (
                    f"sigma_B_T temporal dimension T must either be 1 or a multiple of sample_n_views. Got T={sigma_B_T.shape[-1]} and sample_n_views={n_views}"
                )
                sigma_B_T = rearrange(sigma_B_T, "B (V T) -> (B V) T", V=n_views).contiguous()
                reshape_sigma_B_T = True
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = super().broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if reshape_sigma_B_T:
            sigma_B_T = rearrange(sigma_B_T, "(B V) T -> B (V T)", V=n_views)
        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "latent_view_indices_B_T" in data_batch:
            log.debug("latent_view_indices_B_T already in data_batch")
            return data_batch
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu()[0])
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)

        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.config.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        # data_batch_with_latent_view_indices = data_batch.copy()
        data_batch["latent_view_indices_B_T"] = latent_view_indices_B_T

        return data_batch

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        input_key = self.input_data_key if input_key is None else input_key
        is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True

        num_video_frames_per_view = (
            self.tokenizer.get_pixel_num_frames(self.state_t)
            if is_preprocessed
            else data_batch["num_video_frames_per_view"]
        )
        if isinstance(num_video_frames_per_view, torch.Tensor):
            num_video_frames_per_view = int(num_video_frames_per_view.cpu()[0])
        n_views = data_batch[input_key].shape[2] // num_video_frames_per_view

        keys_to_normalize = []
        if input_key in data_batch:
            keys_to_normalize.append(input_key)
        for key in data_batch.keys():
            if key.startswith("control_input_") and data_batch[key] is not None:
                keys_to_normalize.append(key)

        for key in keys_to_normalize:
            data_batch[key] = rearrange(data_batch[key], "B C (V T) H W -> (B V) C T H W", V=n_views)
        super()._normalize_video_databatch_inplace(data_batch, input_key)
        for key in keys_to_normalize:
            data_batch[key] = rearrange(data_batch[key], "(B V) C T H W -> B C (V T) H W", V=n_views)

    @torch.no_grad()
    def _encode_raw_and_control_inputs(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        # Batch encode disabled to avoid OOM at generation time. GPU allocation very close to 100% early on - revisit after new sac
        encoded_tensors = [self.encode(t) for t in tensors]
        # """Encode several (B, C, V*T, H, W) tensors with a single self.encode call.
        # The tensors are concatenated along the batch dimension, encoded once,
        # and then split back to their original batch sizes.
        # """
        # if len(tensors) == 1:
        #     return [self.encode(tensors[0])]
        # batch_sizes = [t.shape[0] for t in tensors]
        # stacked = torch.cat(tensors, dim=0) # (B_total, C, V*T, H, W)
        # encoded = self.encode(stacked) # (B_total, C, V*T, D)
        # encoded_tensors = torch.split(encoded, batch_sizes, dim=0)
        # encoded_tensors = [t.contiguous().float().to(**self.tensor_kwargs) for t in encoded_tensors]
        return encoded_tensors

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, MultiViewControlVideo2WorldCondition]:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        self._normalize_video_databatch_inplace(data_batch_with_latent_view_indices)
        self._augment_image_dim_inplace(data_batch_with_latent_view_indices)
        is_image_batch = self.is_image_batch(data_batch_with_latent_view_indices)

        input_key = self.input_image_key if is_image_batch else self.input_data_key
        raw_state = data_batch_with_latent_view_indices[input_key]

        condition = self.conditioner(data_batch_with_latent_view_indices)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        assert self.config.net.vace_has_mask is False, "VACE has mask is not yetsupported for multiview control"

        # Combined encoding of raw state and control inputs
        num_modalities = 0
        raw_and_control_inputs = [raw_state]
        latent_control_input = []
        for hint_key in self.hint_keys:
            control_input = getattr(condition, hint_key, None)

            if control_input is not None:
                raw_and_control_inputs.append(control_input)
                num_modalities += 1
        assert num_modalities > 0, "No control input found"
        encoded_tensors = self._encode_raw_and_control_inputs(raw_and_control_inputs)
        latent_state = encoded_tensors[0]
        zero_latent_state = torch.zeros_like(latent_state).to(**self.tensor_kwargs)

        key_id = 1
        for hint_key in self.hint_keys:
            log.debug(f"hint_key: {hint_key}")
            if getattr(condition, hint_key, None) is not None:
                log.debug(f"Adding control input for {hint_key}")
                latent_control_input.append(encoded_tensors[key_id])
                key_id += 1
            else:
                latent_control_input.append(zero_latent_state)

        latent_control_input = torch.cat(latent_control_input, dim=1)

        condition = cast(MultiViewControlVideo2WorldCondition, condition)
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=None,
            view_condition_dropout_max=self.config.view_condition_dropout_max,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input,
            control_weight=data_batch.get(CONTROL_WEIGHT_KEY, 1.0),
        )
        return raw_state, latent_state, condition

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `velocity_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `velocity_fn` which applies a denoising operation on an input using both the conditioned and unconditioned states for rectified flow.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `velocity_fn(noise, noise_x, timestep)` that takes three arguments and returns velocity prediction

        The returned function is suitable for use in rectified flow scenarios where a velocity field is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch_with_latent_view_indices:
            num_conditional_frames: int | list[int] = data_batch_with_latent_view_indices[NUM_CONDITIONAL_FRAMES_KEY]
            log.debug(f"Using {num_conditional_frames=} from data batch")
        else:
            num_conditional_frames: int = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(
                data_batch_with_latent_view_indices
            )
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch_with_latent_view_indices)

        is_image_batch = self.is_image_batch(data_batch_with_latent_view_indices)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, data_batch_condition = self.get_data_and_condition(data_batch_with_latent_view_indices)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )

        assert data_batch_condition.latent_control_input is not None, "No control input found"

        condition = condition.set_control_condition(
            latent_control_input=data_batch_condition.latent_control_input.to(self.tensor_kwargs["device"]),
            control_weight=data_batch.get(CONTROL_WEIGHT_KEY, 1.0),
        )
        uncondition = uncondition.set_control_condition(
            latent_control_input=data_batch_condition.latent_control_input.to(self.tensor_kwargs["device"]),
            control_weight=data_batch.get(CONTROL_WEIGHT_KEY, 1.0),
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)
        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)

            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict[str, torch.Tensor],
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        process_group = parallel_state.get_context_parallel_group()
        cp_size = len(get_process_group_ranks(process_group))
        samples_B_C_T_H_W = super().generate_samples_from_batch(
            data_batch_with_latent_view_indices,
            guidance,
            seed,
            state_shape,
            n_sample,
            is_negative_prompt,
            num_steps,
            shift,
            **kwargs,
        )
        if cp_size > 1:
            num_views = samples_B_C_T_H_W.shape[2] // self.state_t
            H = samples_B_C_T_H_W.shape[3]
            W = samples_B_C_T_H_W.shape[4]
            assert (self.state_t * H * W) % cp_size == 0, (
                "Flattened, one view sequence length must be divisible by CP size."
            )
            """
            We merge the video into contiguous views, as it's scattered per-frame and per-height hunks when using CP.

            1. Flatten all dimensions (T, H, W) into a single sequence.
               Input: B C T H W -> B C (T H W)

            2. The flattened sequence is a concatenation of chunks from each CP rank.
               Because we broadcast-split PER VIEW the sequence structure is:
               [Rank0_View0_Chunk] [Rank0_View1_Chunk] ... [Rank0_ViewN_Chunk]
               [Rank1_View0_Chunk] [Rank1_View1_Chunk] ... [Rank1_ViewN_Chunk]
               ...

               We need to regroup these chunks by VIEW first:
               Input: B C (cp_size num_views chunk)
               Output: B C num_views (cp_size chunk)

               This creates a contiguous block for each view:
               View0: [Rank0_Chunk] [Rank1_Chunk] ... [RankN_Chunk] (This reconstructs the full View0 volume)
               View1: [Rank0_Chunk] [Rank1_Chunk] ... [RankN_Chunk] (This reconstructs the full View1 volume)

            3. Reshape the contiguous view blocks back into (T, H, W).
               The total size of (cp_size * chunk) equals exactly (T * H * W) for one view.
               Input: B C num_views (T H W)
               Output: B C (num_views T) H W
            """
            samples_B_C_T_H_W = rearrange(samples_B_C_T_H_W, "B C T H W -> B C (T H W)")
            samples_B_C_T_H_W = rearrange(
                samples_B_C_T_H_W,
                "B C (cp_size num_views chunk) -> B C num_views (cp_size chunk)",
                cp_size=cp_size,
                num_views=num_views,
            )
            samples_B_C_T_H_W = rearrange(
                samples_B_C_T_H_W,
                "B C num_views (T H W) -> B C (num_views T) H W",
                T=self.state_t,
                H=H,
                W=W,
            )

        return samples_B_C_T_H_W

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition,
    ):
        # Handle control conditioning
        if hasattr(condition, "latent_control_input"):
            # The control conditioning is already set in the condition object
            pass

        # Call parent's denoise method
        return super().denoise(noise, xt_B_C_T_H_W, timesteps_B_T, condition)
