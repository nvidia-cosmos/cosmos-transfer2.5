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
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config

from cosmos_transfer2._src.imaginaire.utils import checkpoint_db
from cosmos_transfer2._src.reason1.tokenizer.processor import AutoProcessor, Processor, build_tokenizer


@pytest.fixture
def loaders(monkeypatch):
    resolver = Mock(return_value="resolved-tokenizer")
    loader = Mock(return_value=SimpleNamespace(eos_token_id=1, pad_token_id=0))
    monkeypatch.setattr(checkpoint_db, "get_checkpoint_path", resolver)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", loader)
    return resolver, loader


def test_explicit_local_directory_bypasses_checkpoint_lookup(loaders, tmp_path):
    resolver, loader = loaders
    directory = str(tmp_path / "tokenizer")
    processor = Processor(cache_dir=directory)
    resolver.assert_not_called()
    loader.assert_called_once_with(directory)
    assert (processor.eos_id, processor.pad_id) == (1, 0)


def test_default_directory_uses_checkpoint_lookup(loaders):
    resolver, loader = loaders
    Processor()
    resolver.assert_called_once_with(
        "s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-3B-Instruct/"
    )
    loader.assert_called_once_with("resolved-tokenizer")


def test_builder_forwards_local_directory(loaders):
    resolver, loader = loaders
    processor = build_tokenizer("Qwen/Qwen2.5-0.5B", cache_dir="local-tokenizer")
    resolver.assert_not_called()
    loader.assert_called_once_with("local-tokenizer")
    assert not processor.is_vision_tokenizer


def test_unsupported_name_fails_before_loading(loaders):
    resolver, loader = loaders
    with pytest.raises(ValueError, match="please check if the tokenizer is available"):
        Processor(name="unsupported", cache_dir="local-tokenizer")
    resolver.assert_not_called()
    loader.assert_not_called()


def test_real_auto_processor_loads_offline_local_tokenizer(monkeypatch, tmp_path):
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, "hello": 3}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.save_pretrained(tmp_path)
    Qwen2Config().save_pretrained(tmp_path)
    resolver = Mock(side_effect=AssertionError("Local tokenizer must not resolve or download checkpoints"))
    monkeypatch.setattr(checkpoint_db, "get_checkpoint_path", resolver)
    processor = Processor(name="Qwen/Qwen2.5-0.5B", cache_dir=str(tmp_path))
    resolver.assert_not_called()
    assert processor.processor("hello")["input_ids"] == [3]
    assert (processor.eos_id, processor.pad_id) == (1, 0)
