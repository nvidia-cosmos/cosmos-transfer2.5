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

from dataclasses import dataclass, field

import torch.distributed as dist
from torch.distributed import ProcessGroup


@dataclass
class HierarchicalCPGroups:
    """Container for hierarchical context parallel groups."""

    rank: int

    HW_ranks: tuple[int, ...] = field(default_factory=tuple)
    HW_group: ProcessGroup | None = None
    T_ranks: tuple[int, ...] = field(default_factory=tuple)
    T_group: ProcessGroup | None = None
    THW_ranks: tuple[int, ...] = field(default_factory=tuple)
    THW_group: ProcessGroup | None = None
    V_ranks: tuple[int, ...] = field(default_factory=tuple)
    V_group: ProcessGroup | None = None
    VHW_ranks: tuple[int, ...] = field(default_factory=tuple)
    VHW_group: ProcessGroup | None = None


def create_hierarchical_cp_groups(world_size: int, rank: int, V: int, T: int) -> HierarchicalCPGroups:
    """
    Create hierarchical context parallel groups.

    The CP strategy is splitting along V then T then HW.

        for example, for V=1 T=4 on 8 GPUs.
            - we prioritize V, so we cp_size_V = V = 1.
            - we then split T, so cp_size_T = T = 4.
            - we then split HW, so cp_size_HW = world_size // (cp_size_V * cp_size_T) = 8 // (1 * 4) = 2.
            - so for split T then HW, the cp_size_THW = cp_size_T * cp_size_HW = 4 * 2 = 8.
            - so for split V then HW, the cp_size_VHW = cp_size_V * cp_size_HW = 1 * 2 = 2.
        - HW cp group:
            - rank 0,1: [0, 1] after gather --> T0 HW
            - rank 2,3: [2, 3] after gather --> T1 HW
            - rank 4,5: [4, 5] after gather --> T2 HW
            - rank 6,7: [6, 7] after gather --> T3 HW
        - T cp group:
            - rank 0,2,4,6: [0, 2, 4, 6] after gather --> T HW0
            - rank 1,3,5,7: [1, 3, 5, 7] after gather --> T HW1
        - THW cp group:
            - rank 0,1,2,3,4,5,6,7: [0, 1, 2, 3, 4, 5, 6, 7] after gather --> T HW
        - V cp group (cp_size_V is 1):
            - None
        - VHW cp group (same as HW cp group since cp_size_V is 1):
            - rank 0,1: [0, 1] after gather --> T0 HW
            - rank 2,3: [2, 3] after gather --> T1 HW
            - rank 4,5: [4, 5] after gather --> T2 HW
            - rank 6,7: [6, 7] after gather --> T3 HW

        another example, for V=4 T=2 on 16 GPUs.
            - we prioritize V, so we cp_size_V = V = 4.
            - we then split T, so cp_size_T = T = 2.
            - we then split HW, so cp_size_HW = world_size // (cp_size_V * cp_size_T) = 16 // (4 * 2) = 2.
            - so for split T then HW, the cp_size_THW = cp_size_T * cp_size_HW = 2 * 2 = 4.
            - so for split V then HW, the cp_size_VHW = cp_size_V * cp_size_HW = 4 * 2 = 8.
        - HW cp group:
            - rank 0,1: [0, 1] after gather --> V0 T0 HW
            - rank 2,3: [2, 3] after gather --> V0 T1 HW
            - rank 4,5: [4, 5] after gather --> V1 T0 HW
            - rank 6,7: [6, 7] after gather --> V1 T1 HW
            - rank 8,9: [8, 9] after gather --> V2 T0 HW
            - rank 10,11: [10, 11] after gather --> V2 T1 HW
            - rank 12,13: [12, 13] after gather --> V3 T0 HW
            - rank 14,15: [14, 15] after gather --> V3 T1 HW
        - T cp group:
            - rank 0,2: [0, 2] after gather --> T V0 HW0
            - rank 1,3: [1, 3] after gather --> T V0 HW1
            - rank 4,6: [4, 6] after gather --> T V1 HW0
            - rank 5,7: [5, 7] after gather --> T V1 HW1
            - rank 8,10: [8, 10] after gather --> T V2 HW0
            - rank 9,11: [9, 11] after gather --> T V2 HW1
            - rank 12,14: [12, 14] after gather --> T V3 HW0
            - rank 13,15: [13, 15] after gather --> T V3 HW1
        - THW cp group:
            - rank 0,1,2,3: [0, 1, 2, 3] after gather --> T V0 HW
            - rank 4,5,6,7: [4, 5, 6, 7] after gather --> T V1 HW
            - rank 8,9,10,11: [8, 9, 10, 11] after gather --> T V2 HW
            - rank 12,13,14,15: [12, 13, 14, 15] after gather --> T V3 HW
        - V cp group:
            - rank 0,4,8,12: [0, 4, 8, 12] after gather --> T0 V HW0
            - rank 1,5,9,13: [1, 5, 9, 13] after gather --> T0 V HW1
            - rank 2,6,10,14: [2, 6, 10, 14] after gather --> T1 V HW0
            - rank 3,7,11,15: [3, 7, 11, 15] after gather --> T1 V HW1
        - VHW cp group:
            - rank 0,1,4,5,8,9,12,13: [0, 1, 4, 5, 8, 9, 12, 13] after gather --> T0 V HW
            - rank 2,3,6,7,10,11,14,15: [2, 3, 6, 7, 10, 11, 14, 15] after gather --> T1 V HW

    Args:
        world_size: Total number of GPUs.
        V: Number of views/videos to split across.
        T: Number of temporal chunks to split across.

    Returns:
        HierarchicalCPGroups containing all process groups and their sizes/ranks.
    """

    def is_power_of_2(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0

    dist_initialized = True if dist.is_initialized() else False
    groups = HierarchicalCPGroups(rank=rank)

    # Only split if the size is a power of 2.
    cp_size_V = min(V, world_size) if is_power_of_2(V) else 1
    cp_size_T = min(T, world_size // cp_size_V) if is_power_of_2(T) else 1
    cp_size_HW = world_size // (cp_size_V * cp_size_T)
    cp_size_THW = cp_size_T * cp_size_HW
    cp_size_VHW = cp_size_V * cp_size_HW

    # Rank layout: rank = V_idx * cp_size_THW + T_idx * cp_size_HW + HW_idx
    # Decode current rank's indices
    v_idx = rank // cp_size_THW
    t_idx = (rank % cp_size_THW) // cp_size_HW
    hw_idx = rank % cp_size_HW

    # Create HW groups: ranks with same V_idx and T_idx
    for v in range(cp_size_V):
        for t in range(cp_size_T):
            ranks = tuple([v * cp_size_THW + t * cp_size_HW + hw for hw in range(cp_size_HW)])
            group = dist.new_group(ranks) if dist_initialized else None
            if v_idx == v and t_idx == t:
                groups.HW_ranks = ranks
                groups.HW_group = group

    # Create T groups: ranks with same V_idx and HW_idx
    for v in range(cp_size_V):
        for hw in range(cp_size_HW):
            ranks = tuple([v * cp_size_THW + t * cp_size_HW + hw for t in range(cp_size_T)])
            group = dist.new_group(ranks) if dist_initialized else None
            if v_idx == v and hw_idx == hw:
                groups.T_ranks = ranks
                groups.T_group = group

    # Create THW groups: ranks with same V_idx
    for v in range(cp_size_V):
        ranks = tuple([v * cp_size_THW + t * cp_size_HW + hw for t in range(cp_size_T) for hw in range(cp_size_HW)])
        group = dist.new_group(ranks) if dist_initialized else None
        if v_idx == v:
            groups.THW_ranks = ranks
            groups.THW_group = group

    # Create V groups: ranks with same T_idx and HW_idx
    for t in range(cp_size_T):
        for hw in range(cp_size_HW):
            ranks = tuple([v * cp_size_THW + t * cp_size_HW + hw for v in range(cp_size_V)])
            group = dist.new_group(ranks) if dist_initialized else None
            if t_idx == t and hw_idx == hw:
                groups.V_ranks = ranks
                groups.V_group = group

    # Create VHW groups: ranks with same T_idx
    for t in range(cp_size_T):
        ranks = tuple([v * cp_size_THW + t * cp_size_HW + hw for v in range(cp_size_V) for hw in range(cp_size_HW)])
        group = dist.new_group(ranks) if dist_initialized else None
        if t_idx == t:
            groups.VHW_ranks = ranks
            groups.VHW_group = group

    return groups


def test_hierarchical_cp_groups(world_size: int, V: int, T: int):
    if world_size == 1:
        # no CP needed -- all ranks belong to the group 0.
        expected = {
            "HW_groups": [
                (0,),
            ],
            "T_groups": [
                (0,),
            ],
            "THW_groups": [
                (0,),
            ],
            "V_groups": [
                (0,),
            ],
            "VHW_groups": [
                (0,),
            ],
        }
    elif (
        (world_size == 2 and V == 1 and T == 1)
        or (world_size == 2 and V == 1 and T == 3)
        or (world_size == 2 and V == 3 and T == 1)
        or (world_size == 2 and V == 5 and T == 5)
    ):
        # cannot split V or T, so split HW.
        expected = {
            "HW_groups": [
                (0, 1),
            ],
            "T_groups": [
                (0,),
                (1,),
            ],
            "THW_groups": [
                (0, 1),
            ],
            "V_groups": [
                (0,),
                (1,),
            ],
            "VHW_groups": [
                (0, 1),
            ],
        }
    elif (world_size == 2 and V == 1 and T == 2) or (world_size == 2 and V == 1 and T == 4):
        # can not split V but can split T, so split T.
        expected = {
            "HW_groups": [
                (0,),
                (1,),
            ],
            "T_groups": [
                (0, 1),
            ],
            "THW_groups": [
                (0, 1),
            ],
            "V_groups": [
                (0,),
                (1,),
            ],
            "VHW_groups": [
                (0,),
                (1,),
            ],
        }
    elif world_size == 4 and V == 1 and T == 2:
        # can not split V but can split T, and also has to split HW
        expected = {
            "HW_groups": [
                (0, 1),
                (2, 3),
            ],
            "T_groups": [
                (0, 2),
                (1, 3),
            ],
            "THW_groups": [
                (0, 1, 2, 3),
            ],
            "V_groups": [
                (0,),
                (1,),
                (2,),
                (3,),
            ],
            "VHW_groups": [
                (0, 1),
                (2, 3),
            ],
        }
    elif world_size == 2 and V == 2:
        # can split V
        expected = {
            "HW_groups": [
                (0,),
                (1,),
            ],
            "T_groups": [
                (0,),
                (1,),
            ],
            "THW_groups": [
                (0,),
                (1,),
            ],
            "V_groups": [
                (0, 1),
            ],
            "VHW_groups": [
                (0, 1),
            ],
        }
    elif world_size == 8 and V == 2 and T == 2:
        # can split V, T, but also has to split HW
        expected = {
            "HW_groups": [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
            ],
            "T_groups": [
                (0, 2),
                (1, 3),
                (4, 6),
                (5, 7),
            ],
            "THW_groups": [
                (0, 1, 2, 3),
                (4, 5, 6, 7),
            ],
            "V_groups": [
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ],
            "VHW_groups": [
                (0, 1, 4, 5),
                (2, 3, 6, 7),
            ],
        }
    elif world_size == 4 and V == 2 and T == 2:
        # can split V and T
        expected = {
            "HW_groups": [
                (0,),
                (1,),
                (2,),
                (3,),
            ],
            "T_groups": [
                (0, 1),
                (2, 3),
            ],
            "THW_groups": [
                (
                    0,
                    1,
                ),
                (
                    2,
                    3,
                ),
            ],
            "V_groups": [
                (0, 2),
                (1, 3),
            ],
            "VHW_groups": [
                (0, 2),
                (1, 3),
            ],
        }
    elif world_size == 8 and V == 1 and T == 4:
        # can not split V but can split T, and also has to split HW
        expected = {
            "HW_groups": [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
            ],
            "T_groups": [
                (0, 2, 4, 6),
                (1, 3, 5, 7),
            ],
            "THW_groups": [
                (0, 1, 2, 3, 4, 5, 6, 7),
            ],
            "V_groups": [
                (0,),
                (1,),
                (2,),
                (3,),
                (4,),
                (5,),
                (6,),
                (7,),
            ],
            "VHW_groups": [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
            ],
        }
    else:
        raise ValueError(f"Unsupported world_size: {world_size}, V: {V}, T: {T}")

    results = {
        rank: create_hierarchical_cp_groups(world_size=world_size, rank=rank, V=V, T=T) for rank in range(world_size)
    }

    HW_groups = []
    T_groups = []
    THW_groups = []
    V_groups = []
    VHW_groups = []
    for rank, result in results.items():
        assert rank in result.HW_ranks, "Itself should be in its own group."
        assert rank in result.T_ranks, "Itself should be in its own group."
        assert rank in result.THW_ranks, "Itself should be in its own group."
        assert rank in result.V_ranks, "Itself should be in its own group."
        assert rank in result.VHW_ranks, "Itself should be in its own group."
        HW_groups.append(result.HW_ranks)
        T_groups.append(result.T_ranks)
        THW_groups.append(result.THW_ranks)
        V_groups.append(result.V_ranks)
        VHW_groups.append(result.VHW_ranks)
    assert set(HW_groups) == set(expected["HW_groups"]), f"HW_groups: {HW_groups}; expected: {expected['HW_groups']}"
    assert set(T_groups) == set(expected["T_groups"]), f"T_groups: {T_groups}; expected: {expected['T_groups']}"
    assert set(THW_groups) == set(expected["THW_groups"]), (
        f"THW_groups: {THW_groups}; expected: {expected['THW_groups']}"
    )
    assert set(V_groups) == set(expected["V_groups"]), f"V_groups: {V_groups}; expected: {expected['V_groups']}"
    assert set(VHW_groups) == set(expected["VHW_groups"]), (
        f"VHW_groups: {VHW_groups}; expected: {expected['VHW_groups']}"
    )

    print(f"[world_size={world_size}, V={V}, T={T}] passed.")


if __name__ == "__main__":
    test_hierarchical_cp_groups(world_size=1, V=1, T=1)
    test_hierarchical_cp_groups(world_size=1, V=1, T=3)
    test_hierarchical_cp_groups(world_size=1, V=4, T=3)

    test_hierarchical_cp_groups(world_size=2, V=1, T=1)
    test_hierarchical_cp_groups(world_size=2, V=1, T=3)
    test_hierarchical_cp_groups(world_size=2, V=3, T=1)
    test_hierarchical_cp_groups(world_size=2, V=5, T=5)

    test_hierarchical_cp_groups(world_size=2, V=1, T=2)
    test_hierarchical_cp_groups(world_size=2, V=1, T=4)

    test_hierarchical_cp_groups(world_size=4, V=1, T=2)

    test_hierarchical_cp_groups(world_size=2, V=2, T=1)
    test_hierarchical_cp_groups(world_size=2, V=2, T=3)

    test_hierarchical_cp_groups(world_size=8, V=2, T=2)

    test_hierarchical_cp_groups(world_size=4, V=2, T=2)

    test_hierarchical_cp_groups(world_size=8, V=1, T=4)
