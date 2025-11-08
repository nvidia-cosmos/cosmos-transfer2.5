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

import copy
from typing import Optional

from cosmos_transfer2._src.predict2.datasets.augmentors.text_transforms_for_video import TextTransformForVideo


class TextTransformForVideoCustomizedKey(TextTransformForVideo):
    def __init__(
        self,
        input_keys: dict,
        output_keys: Optional[list] = None,
        args: Optional[dict] = None,
        return_embedding_key: str = "t5_text_embeddings",
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.return_embedding_key = return_embedding_key
        self.original_embeddings_key = args["original_embeddings_key"]
        self.driving_dataloader_config = args["driving_dataloader_config"]
        self.video_id = args["video_id"]
        self.view_id = args["view_id"]

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """
        meta = copy.deepcopy(
            data_dict[self.captions_key]  # self.captions_key is metas
        )  # Metadata for all views. We only pass the metadata for one view to the text transform augmentor
        caption_id = int(self.video_id)
        data_dict[self.captions_key] = {self.caption_windows_key: meta[caption_id][self.caption_windows_key]}
        data_dict = super().__call__(data_dict)
        data_dict[self.captions_key] = meta  # restore meta
        # Remove new lines from captions
        data_dict[f"ai_caption_{self.view_id}"] = data_dict["ai_caption"].replace("\n\n", " ").replace("\n", " ")

        del data_dict["ai_caption"]

        if self._load_embeddings:
            data_dict[self.return_embedding_key] = data_dict["t5_text_embeddings"]
            del data_dict["t5_text_embeddings"]
        else:
            assert "t5_text_embeddings" not in data_dict, (
                "t5_text_embeddings should not be in data_dict if embeddings are not loaded"
            )
            if self.original_embeddings_key in data_dict:
                del data_dict[self.original_embeddings_key]
        return data_dict
