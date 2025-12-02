# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor

@tensorclass
class EgoExoBatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    ego_unique_objects_identifier: torch.LongTensor
    exo_unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor

@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)

@tensorclass
class EgoExoBatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    ego_img_batch: torch.FloatTensor
    exo_img_batch: torch.FloatTensor
    ego_obj_to_frame_idx: torch.IntTensor
    exo_obj_to_frame_idx: torch.IntTensor
    ego_masks: torch.BoolTensor
    exo_masks: torch.BoolTensor
    metadata: EgoExoBatchedVideoMetaData
    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.ego_obj_to_frame_idx.unbind(dim=-1)
        # 可以输出来看看这两个张量是什么意思？
        # 之所以只计算ego视角的，是因为我们的ego中存在的物体，我们得想办法从exo中分割出来
        # unbind(dim=-1)：将张量按最后一个维度拆分成两个张量。
        flat_idx = video_idx * self.num_frames + frame_idx
        # 返回一个整型张量 flat_idx，每个对象的值是其对应帧在整个扁平化批次中的索引。
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.ego_img_batch.transpose(0, 1).flatten(0, 1), self.exo_img_batch.transpose(0, 1).flatten(0, 1)

@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]

@dataclass
class EgoExoVideoDatapoint:
    """Refers to an image/video and all its annotations"""

    ego_frames: List[Frame]
    exo_frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    # img_batch: 一个list [8, 3, 256, 256]
    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    # [8, 2, 3, 256, 256]
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id #表示在数据集中的位置id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int) #[0,0],[0,0],[0,1][0,1]...这样来记录有哪些帧哪些物体是在每个batch里面的
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0) #[8, 6, 512, 512]
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    ) # [8, 6, 3]
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )

def egoexo_collate_fn(
    batch: List[EgoExoVideoDatapoint],
    dict_key,
) -> EgoExoBatchedVideoDatapoint:
    """
    Args:
        batch: A list of EgoExoVideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    ego_img_batch = []
    exo_img_batch = []
    for video in batch:
        ego_img_batch += [torch.stack([frame.data for frame in video.ego_frames], dim=0)]
        exo_img_batch += [torch.stack([frame.data for frame in video.exo_frames], dim=0)]

    # img_batch: 一个list [8, 3, 256, 256]
    ego_img_batch = torch.stack(ego_img_batch, dim=0).permute((1, 0, 2, 3, 4))
    exo_img_batch = torch.stack(exo_img_batch, dim=0).permute((1, 0, 2, 3, 4))
    # [8, 2, 3, 256, 256]
    T = ego_img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    ego_step_t_objects_identifier = [[] for _ in range(T)]
    ego_step_t_frame_orig_size = [[] for _ in range(T)]

    ego_step_t_masks = [[] for _ in range(T)]
    ego_step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    exo_step_t_objects_identifier = [[] for _ in range(T)]
    exo_step_t_frame_orig_size = [[] for _ in range(T)]

    exo_step_t_masks = [[] for _ in range(T)]
    exo_step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id #表示在数据集中的位置id
        orig_frame_size = video.size
        for t, frame in enumerate(video.ego_frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                ego_step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int) #[0,0],[0,0],[0,1][0,1]...这样来记录有哪些帧哪些物体是在每个batch里面的
                )
                ego_step_t_masks[t].append(obj.segment.to(torch.bool))
                ego_step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                ego_step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

        for t, frame in enumerate(video.exo_frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                exo_step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int) #[0,0],[0,0],[0,1][0,1]...这样来记录有哪些帧哪些物体是在每个batch里面的
                )
                exo_step_t_masks[t].append(obj.segment.to(torch.bool))
                exo_step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                exo_step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    ego_obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in ego_step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    ego_masks = torch.stack([torch.stack(masks, dim=0) for masks in ego_step_t_masks], dim=0) #[8, 6, 512, 512]
    ego_objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in ego_step_t_objects_identifier], dim=0
    ) # [8, 6, 3]
    ego_frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in ego_step_t_frame_orig_size], dim=0
    )

    exo_obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in ego_step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    exo_masks = torch.stack([torch.stack(masks, dim=0) for masks in exo_step_t_masks], dim=0) #[8, 6, 512, 512]
    exo_objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in exo_step_t_objects_identifier], dim=0
    ) # [8, 6, 3]
    exo_frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in exo_step_t_frame_orig_size], dim=0
    )

    return EgoExoBatchedVideoDatapoint(
        ego_img_batch=ego_img_batch,
        exo_img_batch=exo_img_batch,
        ego_obj_to_frame_idx=ego_obj_to_frame_idx,
        exo_obj_to_frame_idx=exo_obj_to_frame_idx,
        ego_masks=ego_masks,
        exo_masks=exo_masks,
        metadata=EgoExoBatchedVideoMetaData(
            ego_unique_objects_identifier=ego_objects_identifier,
            exo_unique_objects_identifier=exo_objects_identifier,
            frame_orig_size=ego_frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )
