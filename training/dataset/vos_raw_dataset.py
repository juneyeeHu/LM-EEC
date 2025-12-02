# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from natsort import natsorted
import numpy as np

from os import path, replace
import json
from pycocotools import mask as mask_utils
from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    EgoExoSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
)
from torch.utils.data.dataset import Dataset

@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode  # False
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

class EgoExoRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        num_frames = 8,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode  # False
        self.truncate_video = truncate_video
        self.mask_file = "annotation.json"
        self.num_frames = num_frames
        subset = os.listdir(self.img_folder)
        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_ids = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )
        self.video_names = []
        self.frames = {}
        for take_id in self.video_ids:
            annotation_path = os.path.join(self.img_folder, take_id, "annotation.json")
            if not os.path.exists(annotation_path):
                continue
            with open(annotation_path, "r") as fp:
                annotation = json.load(fp)
            masks = annotation["masks"]

            for object_name, cams in masks.items():
                ego_cams = [x for x in masks[object_name].keys() if "aria" in x]
                if len(ego_cams) < 1:
                    continue
                ego_cam_name = ego_cams[0]
                ego_frames = list(cams[ego_cam_name].keys())
                for cam_name, cam_data in cams.items():
                    if not os.path.isdir(
                        os.path.join(self.img_folder, take_id, cam_name)
                    ):
                        continue
                    exo_frames = list(cam_data.keys())
                    if cam_name == ego_cam_name:
                        continue

                    frames = np.intersect1d(ego_frames, exo_frames)
                    frames = natsorted(frames)
                    if len(frames) < self.num_frames:
                        continue

                    vid = path.join(
                        take_id, ego_cam_name, cam_name, object_name.replace("/", "-")
                    )
                    self.frames[vid] = [None] * len(frames)
                    for i, f in enumerate(frames):
                        self.frames[vid][i] = f
                    self.video_names.append(vid)


        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        take_id, ego_cam_name, cam_name, object_name = video_name.split("/")[-4:]

        frames = self.frames[video_name]
        annotation_path = os.path.join(self.img_folder, take_id, self.mask_file)
        with open(annotation_path, "r") as fp:
            annotation = json.load(fp)
        masks_data = annotation["masks"]

        ego_gt_data = masks_data[object_name][ego_cam_name]
        exo_gt_data = masks_data[object_name][cam_name]
        ego_segment_loader = EgoExoSegmentLoader(ego_gt_data, frames)
        exo_segment_loader = EgoExoSegmentLoader(exo_gt_data, frames)
        ego_frames = []
        exo_frames = []
        for _, fpath in enumerate(frames[:: self.sample_rate]):
            fid = int(fpath) # 表示帧号
            ego_frames.append(VOSFrame(fid, image_path=os.path.join(self.img_folder, take_id, ego_cam_name, fpath+".jpg")))
            exo_frames.append(VOSFrame(fid, image_path=os.path.join(self.img_folder, take_id, cam_name, fpath+".jpg")))
        ego_video = VOSVideo(video_name, idx, ego_frames)
        exo_video = VOSVideo(video_name, idx, exo_frames)
        return ego_video, exo_video, ego_segment_loader, exo_segment_loader

    def __len__(self):
        return len(self.video_names)

class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(
        self,
        egoexo_root,
        ego_cam_name,
        max_jump,
        is_bl,
        subset=None,
        num_frames=3,
        max_num_obj=1,
        finetune=False,
        augmentation=False,
        swap=False,
    ):
        self.egoexo_root = egoexo_root
        self.frame_folder = "rgb"
        self.mask_file = "annotation.json"
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames # 4
        self.max_num_obj = max_num_obj # 1
        self.augmentation = augmentation
        self.swap = swap

        self.videos = []
        self.frames = {}

        self.takes = sorted(os.listdir(self.egoexo_root))

        for take_id in self.takes:
            annotation_path = os.path.join(self.egoexo_root, take_id, "annotation.json")
            if not os.path.exists(annotation_path):
                continue
            with open(annotation_path, "r") as fp:
                annotation = json.load(fp)
            masks = annotation["masks"]

            for object_name, cams in masks.items():
                ego_cams = [x for x in masks[object_name].keys() if "aria" in x]
                if len(ego_cams) < 1:
                    continue
                ego_cam_name = ego_cams[0]
                ego_frames = list(cams[ego_cam_name].keys())
                for cam_name, cam_data in cams.items():
                    if not os.path.isdir(
                        os.path.join(self.egoexo_root, take_id, cam_name)
                    ):
                        continue
                    exo_frames = list(cam_data.keys())
                    if cam_name == ego_cam_name:
                        continue

                    frames = np.intersect1d(ego_frames, exo_frames) # 找到 ego_frames 和 exo_frames 两个数组中的共同元素（即交集）
                    if len(frames) < num_frames:
                        continue

                    vid = path.join(
                        take_id, ego_cam_name, cam_name, object_name.replace("/", "-")
                    )
                    self.frames[vid] = [None] * len(frames)
                    for i, f in enumerate(frames):
                        self.frames[vid][i] = path.join(cam_name, object_name, f)  # eg. cam01/bike wheel_0/1080
                    self.videos.append(vid)

        print(
            "%d out of %d videos accepted in %s."
            % (len(self.videos), len(self.takes), egoexo_root)
        )

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.01, 0.01, 0.01, 0),
            ]
        )

        self.pair_im_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=0 if finetune or self.is_bl else 15,
                    shear=0 if finetune or self.is_bl else 10,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=im_mean,
                ),
            ]
        )

        self.pair_gt_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=0 if finetune or self.is_bl else 15,
                    shear=0 if finetune or self.is_bl else 10,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ),
            ]
        )

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.03, 0.03, 0),
                transforms.RandomGrayscale(0.05),
            ]
        )

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (480, 480),
                        scale=(0.25, 1.00),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                ]
            )

            self.all_gt_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (480, 480),
                        scale=(0.25, 1.00),
                        interpolation=InterpolationMode.NEAREST,
                    ),
                ]
            )
        else:
            self.all_im_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (480, 480),
                        scale=(0.36, 1.00),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                ]
            )

            self.all_gt_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (480, 480),
                        scale=(0.36, 1.00),
                        interpolation=InterpolationMode.NEAREST,
                    ),
                ]
            )

        # Final transform without randomness
        self.final_gt_transform = transforms.Compose(
            [
                transforms.Resize((480, 480), interpolation=InterpolationMode.NEAREST),
            ]
        )

        self.final_im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((480, 480)),
                im_normalization,
            ]
        )

    def get_images(self, frames_idx, take_root, frames, cam_name):
        info_frames = []
        images = []
        masks = []
        sequence_seed = np.random.randint(2147483647)
        for f_idx in frames_idx:
            components = frames[f_idx].split("/")
            object_name = "/".join(components[1:-1])
            f_name = components[-1]
            rgb_name = f"{f_name}.jpg"
            # rgb_path = os.path.join(self.egoexo_root, take_id, cam_name, rgb_name)
            rgb_path = os.path.join(take_root, cam_name, rgb_name)

            annotation_path = os.path.join(take_root, self.mask_file)
            with open(annotation_path, "r") as fp:
                annotation = json.load(fp)
            masks_data = annotation["masks"]

            gt_data = masks_data[object_name][cam_name][f_name]

            info_frames.append(rgb_path)

            this_im = Image.open(rgb_path).convert("RGB")
            if self.augmentation:
                reseed(sequence_seed)
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
            this_gt = mask_utils.decode(gt_data) * 255
            this_gt = Image.fromarray(this_gt)
            if self.augmentation:
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)
            else:
                this_gt = self.final_gt_transform(this_gt)
            this_im = self.final_im_transform(this_im)
            this_gt = np.array(this_gt)
            this_gt[this_gt != 255] = 0

            images.append(this_im)
            masks.append(this_gt)
        return images, masks, info_frames

    def one_hot_gt(self, masks, target_objects):
        cls_gt = np.zeros((self.num_frames, 480, 480), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 480, 480), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = masks == l
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = this_mask[0]
        cls_gt = np.expand_dims(cls_gt, 1)
        return cls_gt, first_frame_gt

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info["name"] = video

        take_id, ego_cam_name, cam_name, object_name = video.split("/")[-4:]

        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info["frames"] = []  # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(
                range(
                    max(0, frames_idx[-1] - this_max_jump),
                    min(length, frames_idx[-1] + this_max_jump + 1),
                )
            ).difference(set(frames_idx))
            while len(frames_idx) < num_frames:
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(
                    range(
                        max(0, frames_idx[-1] - this_max_jump),
                        min(length, frames_idx[-1] + this_max_jump + 1),
                    )
                )
                acceptable_set = acceptable_set.union(new_set).difference(
                    set(frames_idx)
                )

            frames_idx = natsorted(frames_idx)
            # frames_idx 的内容可能是类似 ['1', '10', '2', '20'] 这样的字符串或数字，默认的排序方法会把 "10" 排在 "1" 后面，
            # 而 "2" 会排在 "10" 后面。使用 natsorted 后，它会将这些元素自然地按数值大小顺序排序，结果应该是 ['1', '2', '10', '20']。
            images, masks, info_frames = self.get_images(
                frames_idx, os.path.join(self.egoexo_root, take_id), frames, cam_name
            )
            ego_images, ego_masks, _ = self.get_images(
                frames_idx,
                os.path.join(self.egoexo_root, take_id),
                frames,
                ego_cam_name,
            )
            info["frames"] = info_frames

            images = torch.stack(images, 0)
            ego_images = torch.stack(ego_images, 0)

            labels = np.unique(ego_masks[0])
            # Remove background
            labels = labels[labels != 0]

            target_objects = []
            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (ego_masks[0] == l).sum()
                    if pixel_sum > 10 * 10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30 * 30:
                            good_lables.append(l)
                        elif (
                            max((ego_masks[1] == l).sum(), (ego_masks[2] == l).sum())
                            < 20 * 20
                        ):
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)

            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(
                target_objects, size=self.max_num_obj, replace=False
            )

        info["num_objects"] = max(1, len(target_objects))

        masks = np.stack(masks, 0)
        ego_masks = np.stack(ego_masks, 0)
        # Generate one-hot ground-truth
        cls_gt, first_frame_gt = self.one_hot_gt(masks, target_objects)
        ego_cls_gt, ego_first_frame_gt = self.one_hot_gt(ego_masks, target_objects)
        # ego_cls_gt: [4, 1, 480, 480], ego_first_frame_gt: [1, 1, 480, 480]
        # 1 if object exist, 0 otherwise
        selector = [
            1 if i < info["num_objects"] else 0 for i in range(self.max_num_obj)
        ]
        selector = torch.FloatTensor(selector)

        if not self.swap:
            data = {
                "rgb": images,
                "first_frame_gt": first_frame_gt,
                "cls_gt": cls_gt,
                "ego_rgb": ego_images,
                "ego_first_frame_gt": ego_first_frame_gt,
                "ego_cls_gt": ego_cls_gt,
                "selector": selector,
                "info": info,
            }
        else:
            data = {
                "ego_rgb": images,
                "ego_first_frame_gt": first_frame_gt,
                "ego_cls_gt": cls_gt,
                "rgb": ego_images,
                "first_frame_gt": ego_first_frame_gt,
                "cls_gt": ego_cls_gt,
                "selector": selector,
                "info": info,
            }

        return data

    def __len__(self):
        return len(self.videos)


class SA1BRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
