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
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

import torch


@contextmanager
def nullcontext_decorator(*args, **kwargs):
    yield


ENABLE_NVTX_DECORATOR = os.environ.get("ENABLE_NVTX_DECORATOR", "0") == "1"
NVTXRangeDecorator = torch.cuda.nvtx.range if ENABLE_NVTX_DECORATOR else nullcontext_decorator


class GPUUtilization:
    def __init__(self):
        self.running_avg = 0.0
        self.count = 0
        self.should_stop = False
        # The presence of GIL makes this thread-safe implicitly
        # self.lock = threading.Lock()

    def start(self):
        self.thread = threading.Thread(target=self._measure_gpu_utilization)
        self.thread.start()

    def stop(self):
        self.should_stop = True
        self.thread.join()

    def _measure_gpu_utilization(self):
        while not self.should_stop:
            utilization = torch.cuda.utilization()
            self.running_avg = (self.running_avg * self.count + utilization) / (self.count + 1)
            self.count += 1
            time.sleep(0.1)


def timeit(repeats: int, f: Callable, *args, **kwargs) -> tuple[float, float, Any]:
    # warmup, to trigger jit compilation etc.
    warmups = kwargs.pop("warmups", 2)
    for _ in range(warmups):
        f(*args, **kwargs)

    # Launch a background thread to measure GPU utilization
    gpu_utilization = GPUUtilization()
    gpu_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats + 1)]

    torch.cuda.synchronize()

    # Measure time and GPU utilization
    gpu_utilization.start()
    for i in range(repeats):
        gpu_events[i].record()
        with NVTXRangeDecorator(f"timeit_repeat_{i}"):
            results: Any = f(*args, **kwargs)
    gpu_events[-1].record()

    # Collect measurements
    torch.cuda.synchronize()
    gpu_utilization.stop()

    times = [gpu_events[i].elapsed_time(gpu_events[i + 1]) for i in range(repeats)]
    # print(f"Timeit measurement times: {times}")

    avg_time = sum(times) / repeats / 1000.0  # ms to s
    avg_utilization = gpu_utilization.running_avg

    return avg_time, avg_utilization, results
