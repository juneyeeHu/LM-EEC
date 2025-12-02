# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time

new_project_path = "/data/seg/LM-EEC"
if new_project_path not in sys.path:
    sys.path.insert(0, new_project_path)
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections import defaultdict
import json
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_ego
from natsort import natsorted
from pycocotools import mask as mask_utils
from os import path
# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette

def get_mask_by_key(self, ref_key):
    gt_data = self.masks_data[self.object_name][self.ref_cam_name][ref_key]
    this_gt = mask_utils.decode(gt_data) * 255
    mask = Image.fromarray(this_gt).convert("P")
    mask = np.array(mask, dtype=np.uint8)
    mask[mask != 255] = 0
    return mask

def load_masks_from_dir_by_json(
    input_mask_dir, video_name, object_name, cam_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    input_mask_json_path = os.path.join(input_mask_dir, video_name, "annotation.json")
    with open(input_mask_json_path, "r") as f:
        input_mask_json = json.load(f)

    gt_data = input_mask_json['masks'][object_name][cam_name][frame_name]
    this_gt = mask_utils.decode(gt_data) * 255 # [2160, 3840]
    #         this_gt = mask_utils.decode(gt_data) * 255
    #         self.palette = Image.fromarray(this_gt).getpalette()
    palette = Image.fromarray(this_gt).getpalette()
    # mask[mask != 255] = 0
    # palette = mask.getpalette()
    mask = Image.fromarray(this_gt).convert("P")
    mask = np.array(mask, dtype=np.uint8)
    mask[mask != 255] = 0
    per_obj_input_mask = get_per_obj_mask(mask)

    return per_obj_input_mask, palette


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)

def save_masks_code_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, frame_name+".json"
        )
        out_mask = output_mask.astype(np.uint8)
        out_img_coco = mask_utils.encode(
            np.asfortranarray((out_mask // 255).astype(np.uint8))
        ) # 为什么要除以255
        out_img_coco["counts"] = out_img_coco["counts"].decode("utf-8")
        with open(
                output_mask_path, "w+"
        ) as fp:
            json.dump(out_img_coco, fp)

        # save_ann_png(output_mask_path, output_mask, output_palette)



@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    video_frames,
    cam_name,
    ego_cam_name,
    object_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    """Run VOS inference on a single video with the given predictor."""
    # load the video frames and initialize the inference state on this video
    exo_video_dir = os.path.join(base_video_dir, video_name, cam_name)
    ego_video_dir = os.path.join(base_video_dir, video_name, ego_cam_name)
    print(f"Running inference on {exo_video_dir}")
    inference_state = predictor.init_state(
        ego_video_path=ego_video_dir, exo_video_path=exo_video_dir, video_frames=video_frames, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # fetch mask inputs from input_mask_dir (either only mask for the first frame, or all available masks)
    if not use_all_masks:
        # use only the first video's ground-truth mask as the input mask
        input_frame_inds = [0]
        # check and make sure we got at least one input frame
        if len(input_frame_inds) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}. "
                "Please make sure the input masks are available in the correct format."
            )
        input_frame_inds = sorted(set(input_frame_inds))
        # print("input_frame_inds", input_frame_inds)
    # add those input masks to SAM 2 inference state before propagation
    object_ids_set = None
    ego_masks = {}
    for input_frame_idx in range(inference_state["num_frames"]):
        try:
            per_obj_input_mask, input_palette =  load_masks_from_dir_by_json(
                input_mask_dir=base_video_dir,
                video_name=video_name,
                object_name=object_name,
                cam_name = ego_cam_name,
                frame_name=video_frames[input_frame_idx],
                per_obj_png_file=per_obj_png_file,
            )
            # print("per_obj_input_mask", per_obj_input_mask)
            # 取出字典中所有的矩阵值并添加到列表中
            for key, matrix in per_obj_input_mask.items():
                ego_masks[video_frames[input_frame_idx]] = matrix
        except FileNotFoundError as e:
            raise RuntimeError(
                f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e

    os.makedirs(os.path.join(output_mask_dir, video_name, ego_cam_name+"__"+cam_name, object_name), exist_ok=True)
    output_directory = os.path.join(output_mask_dir, video_name, ego_cam_name+"__"+cam_name, object_name)
    # output_palette = input_palette or DAVIS_PALETTE
    output_palette = input_palette
    video_segments = {}  # video_segments contains the per-frame segmentation results

    torch.cuda.synchronize()
    start_time = time.time()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, ego_masks, video_frames
    ):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask
    end_time = time.time()
    total_time = end_time - start_time
    num_frames = inference_state["num_frames"]
    print("num_frames", num_frames)
    fps = num_frames / total_time
    print(f"FPS: {fps:.2f}")
    # write the output masks as palette PNG files to output_mask_dir
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_code_to_dir(
            output_mask_dir=output_directory,
            video_name=video_name,
            frame_name=video_frames[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=predictor.image_size,
            width=predictor.image_size,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    """
    Run VOS inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        if per_obj_png_file or os.path.exists(
            os.path.join(input_mask_dir, video_name, f"{name}.png")
        ):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[idx],
                per_obj_png_file=per_obj_png_file,
                allow_missing=True,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask

    # run inference separately for each object in the video
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    for object_id in object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=inputs_per_object[object_id][input_frame_idx],
            )

        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):
            obj_scores = out_mask_logits.cpu().numpy()
            output_scores_per_object[object_id][out_frame_idx] = obj_scores

    # post-processing: consolidate the per-object scores into per-frame masks
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx in range(len(frame_names)):
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        if not per_obj_png_file:
            scores = predictor._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask

    # write the output masks as palette PNG files to output_mask_dir
    for frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )


def main():
    wo_first_mask=0
    num_empty_annotated_frames = 0
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="/data/seg/LM-EEC/checkpoints_EgoExo/checkpoint.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        default="/data/seg/EgoExo4D/val",
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        # required=True,
        default="/data/seg/DAVIS/2017/trainval/Annotations/480p",
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        # required=True,
        default="/data/seg/LM-EEC/egoexo_val",
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether use separate per-object PNG files for input and output masks "
        "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
        "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--track_object_appearing_later_in_video",
        action="store_true",
        help="whether to track objects that appear later in the video (i.e. not on the first frame; "
        "some VOS datasets like LVOS or YouTube-VOS don't have all objects appearing in the first frame)",
    )
    parser.add_argument(
        "--use_vos_optimized_video_predictor",
        action="store_true",
        help="whether to use vos optimized video predictor with all modules compiled",
    )
    parser.add_argument(
        "--swap",
        type=bool,
        default=False,
        help="default False for predicting exo view videos",
    )

    args = parser.parse_args()

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    predictor = build_sam2_video_predictor_ego(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=args.use_vos_optimized_video_predictor,
    )

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]
    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")
    req_frame_list = []
    vid_list = []
    for n_video, video_name in enumerate(video_names):
        video_path = os.path.join(args.output_mask_dir, video_name)

        # if os.path.exists(video_path):
        #     print(f"Skip video: {video_name}")
        #     continue

        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        annotation_path = os.path.join(args.base_video_dir, video_name, "annotation.json")
        with open(annotation_path, "r") as fp:
            annotation = json.load(fp)
        masks = annotation["masks"]
        subsample_idx = annotation["subsample_idx"]

        cams = os.listdir(os.path.join(args.base_video_dir, video_name))
        cams = [c for c in cams if os.path.isdir(os.path.join(args.base_video_dir, video_name, c))]
        ego_cam_name = None
        for cam_name in cams:
            if "aria" in cam_name:
                ego_cam_name = cam_name
        if ego_cam_name is None:
            continue

        for object_name, annotated_cams in masks.items():
            exo_cam_names = [c for c in cams if c != ego_cam_name]
            for cam_name in exo_cam_names:
                if cam_name in list(masks[object_name].keys()):
                    ego_frames = natsorted(
                        os.listdir(path.join(args.base_video_dir, video_name, ego_cam_name))
                    )
                    ego_frames = [int(f.split(".")[0]) for f in ego_frames]
                    exo_frames = natsorted(
                        os.listdir(path.join(args.base_video_dir, video_name, cam_name))
                    )
                    exo_frames = [int(f.split(".")[0]) for f in exo_frames]

                    if args.swap:
                        if annotated_cams.get(cam_name) is None:
                            continue
                        exo_frames = list(annotated_cams[cam_name].keys())
                        frames = np.intersect1d(ego_frames, exo_frames)
                        frames = natsorted(frames)
                    else:
                        if annotated_cams.get(ego_cam_name) is None:
                            continue
                        ego_frames = list(annotated_cams[ego_cam_name].keys())

                        frames = np.intersect1d(ego_frames, exo_frames) # 选择两者交集
                        frames = natsorted(frames)
                else:
                    continue

                if args.swap:
                    vos_inference(
                        predictor=predictor,
                        base_video_dir=args.base_video_dir,
                        input_mask_dir=args.input_mask_dir,
                        output_mask_dir=args.output_mask_dir,
                        video_name=video_name,
                        video_frames=frames,
                        cam_name=cam_name,
                        ego_cam_name=ego_cam_name,
                        object_name=object_name,
                        score_thresh=args.score_thresh,
                        use_all_masks=args.use_all_masks,
                        per_obj_png_file=args.per_obj_png_file,
                    )
                else:
                    vos_inference(
                        predictor=predictor,
                        base_video_dir=args.base_video_dir,
                        input_mask_dir=args.input_mask_dir,
                        output_mask_dir=args.output_mask_dir,
                        video_name=video_name,
                        video_frames=frames,
                        cam_name=ego_cam_name,
                        ego_cam_name=cam_name,
                        object_name=object_name,
                        score_thresh=args.score_thresh,
                        use_all_masks=args.use_all_masks,
                        per_obj_png_file=args.per_obj_png_file,
                    )

    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )
    print("num_empty_annotated_frames:", num_empty_annotated_frames)


if __name__ == "__main__":
    main()