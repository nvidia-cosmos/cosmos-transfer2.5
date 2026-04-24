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

import gc

import torch

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig

_TEXT_ENCODER: TextEncoder | None = None
_TEXT_ENCODER_CKPT_PATH: str | None = None


def get_reason1_embeddings(
    text: str, reason1_ckpt_path: str, device: str = "cuda", text_encoder: TextEncoder | None = None
):
    """
    Get reason1 embeddings for a given text.
    Output (1, 512, 100352) torch.bfloat16 embeddings

    Args:
        text: Input text to encode
        reason1_ckpt_path: Path to the Reason1 checkpoint
        device: Device to run the encoder on (default: "cuda")
    """
    if text_encoder is None:
        global _TEXT_ENCODER, _TEXT_ENCODER_CKPT_PATH

        # Reinitialize if checkpoint path has changed
        if _TEXT_ENCODER is None or _TEXT_ENCODER_CKPT_PATH != reason1_ckpt_path:
            config = TextEncoderConfig(embedding_concat_strategy="full_concat", ckpt_path=reason1_ckpt_path)
            text_encoder = TextEncoder(config, device=device)
            _TEXT_ENCODER = text_encoder
            _TEXT_ENCODER_CKPT_PATH = reason1_ckpt_path
        else:
            text_encoder = _TEXT_ENCODER

    text_embeddings = text_encoder.compute_text_embeddings_online(
        {
            "text": [text],
        },
        "text",
    )
    return text_embeddings


def unload_text_encoder() -> None:
    """
    Unload the text encoder and release GPU/CPU memory.

    This function:
    - Moves the text encoder model to CPU
    - Deletes the global text encoder instance
    - Clears the cached checkpoint path
    - Empties CUDA cache
    - Forces garbage collection

    Use this when you're done with text encoding to free up memory.
    """
    global _TEXT_ENCODER, _TEXT_ENCODER_CKPT_PATH

    if _TEXT_ENCODER is not None:
        try:
            # Move model to CPU first to release GPU memory
            if hasattr(_TEXT_ENCODER, "model") and hasattr(_TEXT_ENCODER.model, "to"):
                _TEXT_ENCODER.model.to("cpu")  # type: ignore[attr-defined]
            elif hasattr(_TEXT_ENCODER, "to"):
                _TEXT_ENCODER.to("cpu")  # type: ignore[attr-defined]

            # Delete the encoder instance
            del _TEXT_ENCODER
        except Exception as e:
            print(f"Warning: Error during text encoder cleanup: {e}")

    # Reset global variables
    _TEXT_ENCODER = None
    _TEXT_ENCODER_CKPT_PATH = None

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    log.info("Text encoder unloaded and memory released")


# PYTHONPATH=. python  cosmos_transfer2/_src/av/causal/fast_infer/v2/text_encoder.py
if __name__ == "__main__":
    import torch

    # Default checkpoint path for testing
    CKPT_PATH = "/lustre/fs1/portfolios/nvr/projects/nvr_torontoai_videogen/users/ruilongl/imaginaire4/checkpoints/text_encoder/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model"

    prompt = "A dashcam-style urban street scene at a modern intersection during daylight. The traffic light is red, and several pedestrians are crossing the street: one person walking in front of the car, two people crossing from the left sidewalk, and a cyclist riding through the intersection. Smooth camera motion should follow a natural idle state as the car waits at the red light. Add subtle environmental animation: pedestrians walking forward, cyclist pedaling past, distant cars moving slowly in traffic, and soft movement in the clouds. Keep lighting realistic with warm afternoon tones, and maintain high clarity and natural motion consistent with real dashcam footage."
    text_embeddings = get_reason1_embeddings(prompt, reason1_ckpt_path=CKPT_PATH, device="cuda:0")
    torch.save(
        text_embeddings,
        "/lustre/fs1/portfolios/nvr/projects/nvr_torontoai_videogen/users/ruilongl/imaginaire4/data_local/text_embeddings_A_dashcam-style_urban_street_scene.pt",
    )
    print(text_embeddings.shape, text_embeddings.dtype, text_embeddings.device)
