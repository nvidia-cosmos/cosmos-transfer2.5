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

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_transfer2._src.imaginaire.utils.config_helper import override
from cosmos_transfer2._src.predict2.configs.common.defaults.ema import PowerEMAConfig
from cosmos_transfer2._src.predict2.configs.common.defaults.tokenizer import DummyJointImageVideoConfig
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import (
    VideoPredictionControlConditioner,
    VideoPredictionControlConditionerImageContext,
)
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.net import TRANSFER2_CONTROL2WORLD_NET_2B
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import (
    ControlVideo2WorldModelRectifiedFlow,
    ControlVideo2WorldRectifiedFlowConfig,
)

"""
pytest -s cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow_test.py --all
"""


@pytest.fixture
def control_video2world_rectified_flow_config():
    control_video2world_model_config = ControlVideo2WorldRectifiedFlowConfig(
        tokenizer=DummyJointImageVideoConfig,
        conditioner=VideoPredictionControlConditioner,
        net=TRANSFER2_CONTROL2WORLD_NET_2B,
        ema=PowerEMAConfig,
        state_t=3,
    )
    control_video2world_model_config = override(control_video2world_model_config)
    return control_video2world_model_config


@pytest.fixture
def control_video2world_rectified_flow_with_image_context_config():
    control_video2world_model_config = ControlVideo2WorldRectifiedFlowConfig(
        tokenizer=DummyJointImageVideoConfig,
        conditioner=VideoPredictionControlConditionerImageContext,  # Use image context conditioner
        net=TRANSFER2_CONTROL2WORLD_NET_2B,
        ema=PowerEMAConfig,
        state_t=3,
    )
    control_video2world_model_config = override(control_video2world_model_config)

    # Add image context dimension to the config
    img_context_dim = 1152

    # Allow modifying the struct to add new fields
    OmegaConf.set_struct(control_video2world_model_config.net, False)
    control_video2world_model_config.net.extra_image_context_dim = img_context_dim
    OmegaConf.set_struct(control_video2world_model_config.net, True)

    return control_video2world_model_config


@pytest.fixture
def mock_checkpoint_files():
    """Create temporary mock checkpoint files for testing."""
    temp_dir = tempfile.mkdtemp()

    # Create mock checkpoint data with control_blocks and control_embedder keys
    mock_checkpoint_data = {
        "control_blocks.0.weight": torch.randn(10, 10),
        "control_blocks.0.bias": torch.randn(10),
        "control_blocks.1.weight": torch.randn(10, 10),
        "control_blocks.1.bias": torch.randn(10),
        "control_embedder.weight": torch.randn(5, 5),
        "control_embedder.bias": torch.randn(5),
        "other_param": torch.randn(3, 3),
    }

    checkpoint_paths = []
    for i in range(2):  # Create 2 mock checkpoints
        checkpoint_path = os.path.join(temp_dir, f"checkpoint_{i}.pt")
        torch.save(mock_checkpoint_data, checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

    yield checkpoint_paths, mock_checkpoint_data

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


@pytest.mark.flaky(max_runs=3)
def test_load_multi_branch_checkpoints_empty_list(control_video2world_rectified_flow_config):
    """Test load_multi_branch_checkpoints with empty checkpoint list."""
    with (
        patch(
            "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ControlVideo2WorldModelRectifiedFlow.set_up_model"
        ),
        patch(
            "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.Video2WorldModelRectifiedFlow.__init__",
            return_value=None,
        ),
    ):
        model = ControlVideo2WorldModelRectifiedFlow.__new__(ControlVideo2WorldModelRectifiedFlow)

        # Manually set required attributes
        model.config = control_video2world_rectified_flow_config
        model.is_new_training = True
        model.copy_weight_strategy = "first_n"
        model.hint_keys = ["control_input_edge", "control_input_vis", "control_input_depth", "control_input_seg"]

        # Create a mock net attribute
        mock_net = MagicMock()
        mock_net.num_control_branches = 2
        model.net = mock_net

        # Test with empty list - should return early with warning
        with patch("cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.log") as mock_log:
            model.load_multi_branch_checkpoints([])
            mock_log.warning.assert_called_once_with("No checkpoint paths provided for control branches")


@pytest.mark.flaky(max_runs=3)
def test_load_multi_branch_checkpoints_none_paths(control_video2world_rectified_flow_config):
    """Test load_multi_branch_checkpoints with None checkpoint paths."""
    with (
        patch(
            "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ControlVideo2WorldModelRectifiedFlow.set_up_model"
        ),
        patch(
            "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.Video2WorldModelRectifiedFlow.__init__",
            return_value=None,
        ),
    ):
        model = ControlVideo2WorldModelRectifiedFlow.__new__(ControlVideo2WorldModelRectifiedFlow)

        # Manually set required attributes
        model.config = control_video2world_rectified_flow_config
        model.is_new_training = True
        model.copy_weight_strategy = "first_n"
        model.hint_keys = ["control_input_edge", "control_input_vis", "control_input_depth", "control_input_seg"]

        # Create a mock net attribute
        mock_net = MagicMock()
        mock_net.num_control_branches = 2
        model.net = mock_net

        # Mock necessary components
        with (
            patch(
                "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ModelWrapper"
            ) as mock_wrapper,
            patch("cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.log") as mock_log,
        ):
            # Mock the model wrapper and state dict
            mock_wrapper_instance = MagicMock()
            mock_wrapper.return_value = mock_wrapper_instance
            mock_wrapper_instance.state_dict.return_value = {"net.some_param": torch.randn(2, 2)}

            # Test with None paths - should log warnings and continue
            model.load_multi_branch_checkpoints([None, None])

            # Should have 2 warning calls for None paths
            assert mock_log.warning.call_count == 2
            mock_log.warning.assert_any_call("No checkpoint path provided for control branch 0")
            mock_log.warning.assert_any_call("No checkpoint path provided for control branch 1")


@pytest.mark.flaky(max_runs=3)
def test_load_multi_branch_checkpoints_pytorch_format(control_video2world_rectified_flow_config, mock_checkpoint_files):
    """Test load_multi_branch_checkpoints with PyTorch checkpoint format."""
    checkpoint_paths, mock_data = mock_checkpoint_files
    with (
        patch(
            "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ControlVideo2WorldModelRectifiedFlow.set_up_model"
        ),
        patch(
            "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.Video2WorldModelRectifiedFlow.__init__",
            return_value=None,
        ),
    ):
        model = ControlVideo2WorldModelRectifiedFlow.__new__(ControlVideo2WorldModelRectifiedFlow)

        # Manually set required attributes
        model.config = control_video2world_rectified_flow_config
        model.is_new_training = True
        model.copy_weight_strategy = "first_n"
        model.hint_keys = ["control_input_edge", "control_input_vis", "control_input_depth", "control_input_seg"]

        # Create a mock net attribute
        mock_net = MagicMock()
        mock_net.num_control_branches = 2
        model.net = mock_net

        # Mock necessary components
        with (
            patch(
                "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ModelWrapper"
            ) as mock_wrapper,
            patch(
                "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch("cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.log") as mock_log,
        ):
            # Mock the model wrapper and state dict
            mock_wrapper_instance = MagicMock()
            mock_wrapper.return_value = mock_wrapper_instance
            mock_wrapper_instance.state_dict.return_value = {"net.some_param": torch.randn(2, 2)}
            mock_wrapper_instance.load_state_dict = MagicMock()

            # Test loading PyTorch checkpoints
            model.load_multi_branch_checkpoints(checkpoint_paths)

            # Verify load_state_dict was called for each checkpoint
            assert mock_wrapper_instance.load_state_dict.call_count == 2

            # Verify the key mapping was done correctly
            for call_idx, call in enumerate(mock_wrapper_instance.load_state_dict.call_args_list):
                loaded_keys = call[0][0]  # First argument to load_state_dict

                # Check that control_blocks keys were mapped to control_blocks_{nc}
                assert f"control_blocks_{call_idx}.0.weight" in loaded_keys
                assert f"control_blocks_{call_idx}.0.bias" in loaded_keys
                assert f"control_blocks_{call_idx}.1.weight" in loaded_keys
                assert f"control_blocks_{call_idx}.1.bias" in loaded_keys

                # Check that control_embedder keys were mapped to control_embedder.{nc}
                assert f"control_embedder.{call_idx}.weight" in loaded_keys
                assert f"control_embedder.{call_idx}.bias" in loaded_keys

                # Check that other keys are preserved
                assert "other_param" in loaded_keys


@pytest.mark.flaky(max_runs=3)
def test_load_multi_branch_checkpoints_with_credentials(control_video2world_rectified_flow_config):
    """Test load_multi_branch_checkpoints with S3 credentials configuration."""
    with (
        patch(
            "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ControlVideo2WorldModelRectifiedFlow.set_up_model"
        ),
        patch(
            "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.Video2WorldModelRectifiedFlow.__init__",
            return_value=None,
        ),
    ):
        model = ControlVideo2WorldModelRectifiedFlow.__new__(ControlVideo2WorldModelRectifiedFlow)

        # Manually set required attributes
        model.config = control_video2world_rectified_flow_config
        model.is_new_training = True
        model.copy_weight_strategy = "first_n"
        model.hint_keys = ["control_input_edge", "control_input_vis", "control_input_depth", "control_input_seg"]

        # Create a mock net attribute
        mock_net = MagicMock()
        mock_net.num_control_branches = 1
        model.net = mock_net

        # Set up config with base_load_from credentials
        from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict

        model.config.base_load_from = LazyDict({"credentials": "test/credentials/path.secret"})

        # Mock necessary components
        with (
            patch(
                "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ModelWrapper"
            ) as mock_wrapper,
            patch(
                "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.S3StorageReader"
            ) as mock_s3_reader,
            patch(
                "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.torch.distributed.is_initialized",
                return_value=False,
            ),
            patch("cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.dcp") as mock_dcp,
        ):
            # Mock the model wrapper and state dict
            mock_wrapper_instance = MagicMock()
            mock_wrapper.return_value = mock_wrapper_instance
            mock_wrapper_instance.state_dict.return_value = {"net.some_param": torch.randn(2, 2)}
            mock_wrapper_instance.load_state_dict = MagicMock()

            # Mock S3 storage reader
            mock_s3_reader_instance = MagicMock()
            mock_s3_reader.return_value = mock_s3_reader_instance

            # Test with S3 path (should use credentials from config)
            s3_checkpoint_paths = ["s3://bucket/path/to/checkpoint"]
            model.load_multi_branch_checkpoints(s3_checkpoint_paths)

            # Verify S3StorageReader was called with correct credentials
            mock_s3_reader.assert_called_once_with(
                credential_path="test/credentials/path.secret", path="s3://bucket/path/to/checkpoint/model"
            )


@pytest.mark.flaky(max_runs=3)
def test_load_multi_branch_checkpoints_key_mapping(control_video2world_rectified_flow_config):
    """Test that checkpoint keys are correctly mapped to model keys."""
    # Create a temporary checkpoint with specific keys
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")

    # Mock checkpoint data with expected keys
    checkpoint_data = {
        "control_blocks.0.layer.weight": torch.randn(5, 5),
        "control_blocks.1.layer.bias": torch.randn(5),
        "control_embedder.proj.weight": torch.randn(3, 3),
        "other_module.weight": torch.randn(2, 2),
    }
    torch.save(checkpoint_data, checkpoint_path)

    try:
        with (
            patch(
                "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ControlVideo2WorldModelRectifiedFlow.set_up_model"
            ),
            patch(
                "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.Video2WorldModelRectifiedFlow.__init__",
                return_value=None,
            ),
        ):
            model = ControlVideo2WorldModelRectifiedFlow.__new__(ControlVideo2WorldModelRectifiedFlow)

            # Manually set required attributes
            model.config = control_video2world_rectified_flow_config
            model.is_new_training = True
            model.copy_weight_strategy = "first_n"
            model.hint_keys = ["control_input_edge", "control_input_vis", "control_input_depth", "control_input_seg"]

            # Create a mock net attribute
            mock_net = MagicMock()
            mock_net.num_control_branches = 2
            model.net = mock_net

            # Mock necessary components
            with (
                patch(
                    "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.ModelWrapper"
                ) as mock_wrapper,
                patch(
                    "cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow.torch.distributed.is_initialized",
                    return_value=False,
                ),
            ):
                # Mock the model wrapper and state dict
                mock_wrapper_instance = MagicMock()
                mock_wrapper.return_value = mock_wrapper_instance
                mock_wrapper_instance.state_dict.return_value = {"net.some_param": torch.randn(2, 2)}

                # Capture the loaded state dict
                loaded_state_dict = None

                def capture_load_state_dict(state_dict):
                    nonlocal loaded_state_dict
                    loaded_state_dict = state_dict

                mock_wrapper_instance.load_state_dict.side_effect = capture_load_state_dict

                # Test loading the checkpoint
                model.load_multi_branch_checkpoints([checkpoint_path])

                # Verify the key mapping
                assert loaded_state_dict is not None

                # Check that control_blocks keys were mapped correctly (branch 0)
                assert "control_blocks_0.0.layer.weight" in loaded_state_dict
                assert "control_blocks_0.1.layer.bias" in loaded_state_dict

                # Check that control_embedder keys were mapped correctly
                assert "control_embedder.0.proj.weight" in loaded_state_dict

                # Check that other keys are preserved
                assert "other_module.weight" in loaded_state_dict

                # Verify original keys are not present
                assert "control_blocks.0.layer.weight" not in loaded_state_dict
                assert "control_embedder.proj.weight" not in loaded_state_dict

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)


"""
Usage:
    pytest -s cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow_test.py -k test_load_multi_branch_checkpoints
"""
