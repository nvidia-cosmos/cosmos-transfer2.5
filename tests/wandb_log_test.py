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

import pytest
import torch

from cosmos_transfer2._src.predict2.callbacks.wandb_log import _LossRecord


@pytest.mark.gpus(1)
@pytest.mark.parametrize(
    "local,peer,expected",
    [
        ((0, 0, 0), (6, 12, 2), (3, 6)),
        ((6, 12, 2), (0, 0, 0), (3, 6)),
        ((0, 0, 0), (0, 0, 0), (0, 0)),
        ((2, 4, 1), (6, 12, 1), (4, 8)),
        ((2, 4, 1), (12, 24, 3), (3.5, 7)),
    ],
)
def test_loss_record_global_average(monkeypatch, local, peer, expected):
    loss, edm_loss, count = local
    record = _LossRecord(
        loss=torch.tensor(loss, device="cuda", dtype=torch.float32) if count else 0,
        edm_loss=torch.tensor(edm_loss, device="cuda", dtype=torch.float32) if count else 0,
        iter_count=count,
    )
    original_loss = record.loss
    calls = []

    def reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.SUM
        torch.testing.assert_close(tensor, torch.tensor(local, device="cuda", dtype=torch.float32))
        tensor.add_(torch.tensor(peer, device="cuda", dtype=torch.float32))
        calls.append(tensor)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    assert record.get_stat() == pytest.approx(expected)
    assert len(calls) == 1
    assert record == _LossRecord()
    if count:
        assert original_loss.item() == loss
