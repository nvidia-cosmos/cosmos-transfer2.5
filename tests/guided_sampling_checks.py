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

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpus(1)


@pytest.fixture(params=["cpu", "cuda"])
def sampler(request, monkeypatch):
    # Import after the test harness selects the GPU required by model dependencies.
    from cosmos_transfer2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import (
        ControlVideo2WorldModelRectifiedFlow,
    )

    # Bypass checkpoint construction, retaining the actual sampling loop and UniPC solver.
    model = ControlVideo2WorldModelRectifiedFlow.__new__(ControlVideo2WorldModelRectifiedFlow)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(state_t=2, use_kerras_sigma_at_inference=False)
    model.tensor_kwargs = {"device": request.param, "dtype": torch.float32}
    model.input_data_key, model.input_image_key = "video", "images"
    model.tokenizer = SimpleNamespace(get_pixel_num_frames=lambda t: 4 * (t - 1) + 1)
    model.net = SimpleNamespace(is_context_parallel_enabled=False)
    model.sample_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    inputs = []
    samples = []

    def velocity_fn(noise, sample, timestep):
        inputs.append(sample.clone())
        index = model.sample_scheduler.step_index
        sigma = model.sample_scheduler.sigmas[0 if index is None else index]
        # Analytic denoiser: its clean-sample prediction is exactly 0.75.
        return (sample - 0.75) / sigma

    monkeypatch.setattr(model, "get_velocity_fn_from_batch", lambda *args, **kwargs: velocity_fn)
    original_step = model.sample_scheduler.step

    def step(model_output, timestep, sample, **kwargs):
        samples.append(sample.clone())
        return original_step(model_output, timestep, sample, **kwargs)

    monkeypatch.setattr(model.sample_scheduler, "step", step)
    return model, inputs, samples


def generate(model, mask=None, threshold=100):
    batch = {"video": torch.zeros((1, 3, 5, 2, 2), dtype=torch.uint8)}
    if mask is not None:
        batch["x0_spatial_condition"] = {
            "x0": torch.full((1, 1, 2, 2, 2), -0.5, device=model.tensor_kwargs["device"]),
            "x_sigma_mask": torch.tensor(mask, device=model.tensor_kwargs["device"]),
            "step_threshold": threshold,
        }
    return model.generate_samples_from_batch(batch, state_shape=(1, 2, 2, 2), num_steps=5, seed=42)


@pytest.mark.parametrize("mask", [1.0, 0.5, [[1.0, 0.0], [0.0, 1.0]]])
def test_guided_sampler_uses_same_sample_for_denoising_and_solver(sampler, mask):
    model, inputs, samples = sampler
    result = generate(model, mask)
    assert len(inputs) == len(samples) == 5
    for denoiser_input, solver_sample in zip(inputs, samples):
        torch.testing.assert_close(solver_sample, denoiser_input, rtol=0, atol=0)
    torch.testing.assert_close(result, torch.full_like(result, 0.75), rtol=0, atol=1e-5)


@pytest.mark.parametrize("mask,threshold", [(0.0, 100), (1.0, -1)])
def test_disabled_guidance_matches_unconditioned_sampling(sampler, mask, threshold):
    model, _, _ = sampler
    expected = generate(model)
    actual = generate(model, mask, threshold)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_guidance_threshold_keeps_solver_inputs_consistent(sampler):
    model, inputs, samples = sampler
    generate(model, mask=1.0, threshold=2)
    for denoiser_input, solver_sample in zip(inputs, samples):
        torch.testing.assert_close(solver_sample, denoiser_input, rtol=0, atol=0)
