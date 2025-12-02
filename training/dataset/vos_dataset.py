# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from copy import deepcopy

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.dataset.vos_segment_loader import JSONSegmentLoader

from training.utils.data_utils import Frame, Object, VideoDatapoint, EgoExoVideoDatapoint

MAX_RETRIES = 100


class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        # print("len(self.video_dataset):", len(self.video_dataset))
        self.repeat_factors *= multiplier  # 2
        print(f"Raw dataset length = {len(self.video_dataset)}") # 60

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _get_datapoint(self, idx):

        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # sample a video
                video, segment_loader = self.video_dataset.get_video(idx)
                # sample frames and object indices to be used in a datapoint
                sampled_frms_and_objs = self.sampler.sample(
                    video, segment_loader, epoch=self.curr_epoch
                )
                break  # Succesfully loaded video
            except Exception as e:
                if self.training:
                    logging.warning(
                        f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    )
                    idx = random.randrange(0, len(self.video_dataset))
                else:
                    # Shouldn't fail to load a val video
                    raise e

        datapoint = self.construct(video, sampled_frms_and_objs, segment_loader)
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct(self, video, sampled_frms_and_objs, segment_loader):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        # sampled_frms_and_objs指的是什么？
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids = sampled_frms_and_objs.object_ids

        images = []
        rgb_images = load_images(sampled_frames)
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            images.append(
                Frame(
                    data=rgb_images[frame_idx],
                    objects=[],
                )
            )
            # We load the gt segments associated with the current frame
            if isinstance(segment_loader, JSONSegmentLoader):
                segments = segment_loader.load(
                    frame.frame_idx, obj_ids=sampled_object_ids
                )
            else:
                segments = segment_loader.load(frame.frame_idx)
            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[obj_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )
        return VideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)

class EgoExoVOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        # print("len(self.video_dataset):", len(self.video_dataset))
        self.repeat_factors *= multiplier  # 2
        print(f"Raw dataset length = {len(self.video_dataset)}") # 60

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _get_datapoint(self, idx):

        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # sample a video
                ego_video, exo_video, ego_segment_loader, exo_segment_loader = self.video_dataset.get_video(idx)
                # sample frames and object indices to be used in a datapoint
                sampled_frms_and_objs = self.sampler.sample(
                    ego_video, exo_video, ego_segment_loader, exo_segment_loader, epoch=self.curr_epoch
                )
                break  # Succesfully loaded video
            except Exception as e:
                if self.training:
                    logging.warning(
                        f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    )
                    idx = random.randrange(0, len(self.video_dataset))
                else:
                    # Shouldn't fail to load a val video
                    raise e

        datapoint = self.construct(ego_video, exo_video, ego_segment_loader, exo_segment_loader, sampled_frms_and_objs)
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct(self, ego_video, exo_video, ego_segment_loader, exo_segment_loader, sampled_frms_and_objs):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        # sampled_frms_and_objs指的是什么？
        ego_sampled_frames = sampled_frms_and_objs.ego_frames
        exo_sampled_frames = sampled_frms_and_objs.exo_frames
        sampled_object_ids = sampled_frms_and_objs.object_ids

        ego_images = []
        exo_images = []
        ego_rgb_images = load_images(ego_sampled_frames)
        exo_rgb_images = load_images(exo_sampled_frames)
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, frame in enumerate(ego_sampled_frames):
            w, h = ego_rgb_images[frame_idx].size
            ego_images.append(
                Frame(
                    data=ego_rgb_images[frame_idx],
                    objects=[],
                )
            )

            # We load the gt segments associated with the current frame
            if isinstance(ego_segment_loader, JSONSegmentLoader):
                segments = ego_segment_loader.load(
                    frame.frame_idx, obj_ids=sampled_object_ids
                )
            else:
                segments = ego_segment_loader.load(frame.frame_idx)
            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[obj_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                ego_images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )
        for frame_idx, frame in enumerate(exo_sampled_frames):
            w, h = exo_rgb_images[frame_idx].size
            exo_images.append(
                Frame(
                    data=exo_rgb_images[frame_idx],
                    objects=[],
                )
            )

            # We load the gt segments associated with the current frame
            if isinstance(exo_segment_loader, JSONSegmentLoader):
                segments = exo_segment_loader.load(
                    frame.frame_idx, obj_ids=sampled_object_ids
                )
            else:
                segments = exo_segment_loader.load(frame.frame_idx)
            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                            segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[obj_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                exo_images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )

        return EgoExoVideoDatapoint(
            ego_frames=ego_images,
            exo_frames=exo_images,
            video_id=ego_video.video_id,  # 这里两者的video_id是一样的吗？
            size=(h, w),
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)

def load_images(frames):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)
