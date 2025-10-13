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

import cosmos_transfer2._src.predict2.datasets.webdataset as webdataset
from cosmos_transfer2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo
from cosmos_transfer2._src.imaginaire.utils import log


class MadsWebdataset(webdataset.Dataset):
    """
    Small adapter for the mads webdataset for incorrectly formatted root in wdinfo.
    """

    def __init__(self, *args, is_train: bool = True, **kwargs):
        self.is_train = is_train
        super().__init__(*args, **kwargs)

    def parse_dataset_info(self, dataset_info: list[DatasetInfo], use_multithread: bool = True):
        if use_multithread:
            log.warning("MadsWebdataset: use_multithread is not supported, setting to False")
            use_multithread = False
        super().parse_dataset_info(dataset_info, use_multithread)
        val_parts = ["part_000000", "part_000001"]  # 2x1000 samples for validation
        assert self.wdinfo.total_key_count == len(self.wdinfo.tar_files) * self.wdinfo.chunk_size, (
            f"Total key count does not match the number of tar files * chunk size: {self.wdinfo.total_key_count} != {len(self.wdinfo.tar_files)} * {self.wdinfo.chunk_size}"
        )
        log.info(f"MadsWebdataset: train+val tar files: {len(self.wdinfo.tar_files)}")
        if self.is_train:
            # Filter out the val parts
            self.wdinfo.tar_files = [
                tar_sample
                for tar_sample in self.wdinfo.tar_files
                if not any(val_part in tar_sample.path for val_part in val_parts)
            ]
            log.info(f"MadsWebdataset: {len(self.wdinfo.tar_files)} train tar files")
        else:
            # Filter out the train parts
            self.wdinfo.tar_files = [
                tar_sample
                for tar_sample in self.wdinfo.tar_files
                if any(val_part in tar_sample.path for val_part in val_parts)
            ]
            log.info(f"MadsWebdataset: {len(self.wdinfo.tar_files)} val tar files")

        # Update total key count
        if (
            self.wdinfo.tar_files[0].root
            == "mads/cosmos-mads-dataset-av-multiview-0710/v0/driving/resolution_720/aspect_ratio_16_9/duration_5_10/"
        ) or (
            self.wdinfo.tar_files[0].root
            == "mads/cosmos-mads-dataset-transfer2-multiview-0823/v0/driving/resolution_720/aspect_ratio_16_9/duration_5_10/"
        ):
            self.wdinfo.total_key_count = 143_003  # Hardcoded for now due to incorrect wdinfo in S3.
        else:
            self.wdinfo.total_key_count = len(self.wdinfo.tar_files) * self.wdinfo.chunk_size
