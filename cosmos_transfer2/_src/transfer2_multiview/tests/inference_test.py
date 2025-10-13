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

"""
Inference test for ControlVideo2WorldInference using subprocess to run the actual inference script.

pytest cosmos_transfer2/_src/transfer2_multiview/tests/inference_test.py --all -v
"""

import os
import pickle
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.utils.helper_test import RunIf

# Experiment configuration
EXPERIMENT_NAME = "buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k"
CHECKPOINT_PATH = "s3://bucket/cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k-0/checkpoints/iter_000005000"
CONTEXT_PARALLEL_SIZE = 8
MIN_GPUS = 8
GUIDANCE = 7.0
CONTROL_WEIGHT = 1.0
GOOD_PSNR = 30.0

# Fixed seed for reproducibility
FIXED_SEED = 0


def get_project_root() -> Path:
    """Find the project root directory (where imaginaire4 is located)."""
    # Start from the current test file location
    current_file = Path(__file__).resolve()

    # Navigate up from cosmos_transfer2/_src/transfer2_multiview/tests/ to project root
    # We know we're 4 levels deep from the project root
    project_root = current_file.parent.parent.parent.parent

    # Verify we found the right directory by checking for expected files/dirs
    if not (project_root / "cosmos_transfer2._src.imaginaire").exists():
        # If the expected structure isn't found, try using the current working directory
        project_root = Path.cwd()
        if not (project_root / "cosmos_transfer2._src.imaginaire").exists():
            raise RuntimeError(
                f"Could not find project root. Expected 'cosmos_transfer2._src.imaginaire' directory in {project_root}"
            )

    return project_root


def compare_psnr(array1: np.ndarray, array2: np.ndarray, max_pixel_value: float = 1.0) -> float:
    """Compare PSNR between two arrays."""
    array1 = np.clip(array1, 0.0, max_pixel_value)
    array2 = np.clip(array2, 0.0, max_pixel_value)
    overall_mse = ((array1 - array2) ** 2).mean()
    return 10 * np.log10((max_pixel_value**2) / overall_mse) if overall_mse > 0 else float("inf")


def run_inference_subprocess(
    experiment_name: str,
    checkpoint_path: str,
    context_parallel_size: int,
    save_root: str,
    max_samples: int,
    guidance: float,
    seed: int,
    control_weight: float,
    master_port: int = 12341,
) -> subprocess.CompletedProcess:
    """
    Run the inference script using torchrun with specified parameters.

    Args:
        experiment_name: Name of the experiment configuration
        checkpoint_path: Path to model checkpoint
        context_parallel_size: Number of GPUs for context parallelism
        save_root: Directory to save outputs
        max_samples: Maximum number of samples to generate
        guidance: Guidance scale for generation
        seed: Random seed for reproducibility
        control_weight: Control weight for generation
        master_port: Port for distributed training

    Returns:
        CompletedProcess object with the result of the subprocess
    """
    # Get project root directory
    project_root = get_project_root()

    # Build the command
    cmd = [
        "torchrun",
        f"--nproc_per_node={context_parallel_size}",
        f"--master_port={master_port}",
        "-m",
        "cosmos_transfer2._src.transfer2_multiview.inference.inference",
        "--experiment",
        experiment_name,
        "--ckpt_path",
        checkpoint_path,
        "--context_parallel_size",
        str(context_parallel_size),
        "--guidance",
        str(guidance),
        "--seed",
        str(seed),
        "--max_samples",
        str(max_samples),
        "--save_root",
        save_root,
        "--control_weight",
        str(control_weight),
        "--deterministic",
        "--save_data_batch",
        "--save_npy",
    ]

    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    logger.info(f"Running command from {project_root}: {' '.join(cmd)}")

    # Run the subprocess
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(project_root),
        text=True,
    )

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(f"Inference script failed with return code {result.returncode}")

    return result


@pytest.fixture(scope="module")
def output_dir() -> Path:
    """
    Fixture to create an output directory for test results.
    """
    output_path = Path(f"/tmp/{uuid.uuid4()}")
    output_path.mkdir(exist_ok=True, parents=True)

    yield output_path

    # Cleanup after tests (optional)
    shutil.rmtree(output_path, ignore_errors=True)


class TestInferenceSubprocess:
    """Test suite for running inference.py via subprocess with torchrun."""

    @pytest.mark.L1
    @RunIf(min_gpus=MIN_GPUS, requires_file=["credentials/s3_checkpoint.secret", "credentials/pbss_dir.secret"])
    def test_inference_with_expected_outputs(self, output_dir: Path):
        """Test inference and compare with expected outputs."""
        # Create a unique subdirectory for this test

        test_output_dir = output_dir / "test_expected"
        test_output_dir.mkdir(exist_ok=True, parents=True)

        # Run inference
        logger.info("Running inference, this takes several minutes...")
        max_samples = 1
        result = run_inference_subprocess(
            experiment_name=EXPERIMENT_NAME,
            checkpoint_path=CHECKPOINT_PATH,
            context_parallel_size=CONTEXT_PARALLEL_SIZE,
            save_root=str(test_output_dir),
            max_samples=max_samples,
            guidance=GUIDANCE,
            seed=FIXED_SEED,
            control_weight=CONTROL_WEIGHT,
        )

        # Check that the process completed successfully
        assert result.returncode == 0, "Inference script should complete successfully"
        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": "credentials/pbss_dir.secret",
            }
        )

        # Check and compare outputs
        for i in range(max_samples):
            # Test inputs are the same
            batch = pickle.load(open(test_output_dir / f"data_batch_{i}.pkl", "rb"))
            expected_batch = easy_io.load(
                f"s3://testing/projects/cosmos/transfer2_multiview/tests/expected_output/data_batch_{i}.pkl"
            )
            np.testing.assert_allclose(batch["video"].cpu().numpy(), expected_batch["video"].cpu().numpy())

            # Test outputs are the same
            output_path = test_output_dir / f"infer_from_train_{i}.npy"
            output_npy = np.load(output_path)
            assert output_path.exists(), f"Output file {output_path} should exist"

            expected_npy_path = (
                f"s3://testing/projects/cosmos/transfer2_multiview/tests/expected_output/infer_from_train_{i}.npy"
            )
            expected_npy = easy_io.load(expected_npy_path)
            psnr = compare_psnr(output_npy, expected_npy)
            logger.info(f"PSNR: {psnr:.2f} dB")
            assert psnr > GOOD_PSNR, f"PSNR should be greater than {GOOD_PSNR}, got {psnr}"

    @pytest.mark.L1
    @RunIf(min_gpus=MIN_GPUS, requires_file=["credentials/s3_checkpoint.secret", "credentials/pbss_dir.secret"])
    def test_inference_is_deterministic(self, output_dir: Path):
        """Test inference is deterministic."""

        # Create a unique subdirectory for this test
        test_output_dir1 = output_dir / "test_expected1"
        test_output_dir1.mkdir(exist_ok=True, parents=True)

        # Create a unique subdirectory for this test
        test_output_dir2 = output_dir / "test_expected2"
        test_output_dir2.mkdir(exist_ok=True, parents=True)

        # Run inference
        logger.info("Running inference, this takes several minutes...")
        max_samples = 1
        for save_root in [test_output_dir1, test_output_dir2]:
            result = run_inference_subprocess(
                experiment_name=EXPERIMENT_NAME,
                checkpoint_path=CHECKPOINT_PATH,
                context_parallel_size=CONTEXT_PARALLEL_SIZE,
                save_root=str(save_root),
                max_samples=max_samples,
                guidance=GUIDANCE,
                seed=FIXED_SEED,
                control_weight=CONTROL_WEIGHT,
            )
            # Check that the process completed successfully
            assert result.returncode == 0, "Inference script should complete successfully"

        # Test inputs are the same
        batch1 = pickle.load(open(test_output_dir1 / f"data_batch_0.pkl", "rb"))
        batch2 = pickle.load(open(test_output_dir2 / f"data_batch_0.pkl", "rb"))
        np.testing.assert_allclose(batch1["video"].cpu().numpy(), batch2["video"].cpu().numpy())

        # Test outputs are the same
        run1 = np.load(test_output_dir1 / f"infer_from_train_0.npy")
        run2 = np.load(test_output_dir2 / f"infer_from_train_0.npy")
        psnr = compare_psnr(run1, run2)
        logger.info(f"PSNR: {psnr:.2f} dB")
        assert psnr > GOOD_PSNR, f"PSNR should be greater than {GOOD_PSNR}, got {psnr}"
