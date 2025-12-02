import os
import argparse
import json
import time
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils
from PIL import Image
import numpy as np

def main(args):
    start_time = time.time()
    takes = os.listdir(args.input)

    annotation_path = args.gt
    with open(annotation_path, "r") as fp:
        annotation = json.load(fp)

    for take_id in tqdm(takes):
        if not os.path.isdir(os.path.join(args.pred, take_id)):
            continue
        result = process_take(take_id, annotation, args.pred)

        with open(os.path.join(args.pred, take_id, "annotations.json"), "w+") as fp:
            json.dump(result, fp)

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")


def get_folders(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def process_take(take_id, annotation, pred):
    pred_masks = {}
    cam_names = get_folders(os.path.join(pred, take_id))
    object_names = list(annotation["annotations"][take_id]["object_masks"].keys())
    take_anno = annotation["annotations"][take_id]["object_masks"]
    count_empty = 0
    ego_cam_name = cam_names[0].split("__")[0]
    for object_name in object_names:
        if ego_cam_name not in list(annotation["annotations"][take_id]["object_masks"][object_name].keys()):
            continue
        pred_masks[object_name] = {}
        for cams_str in cam_names:
            if cams_str.split("__")[1] not in list(annotation["annotations"][take_id]["object_masks"][object_name].keys()):
                continue

            pred_masks[object_name][cams_str] = {}

            f_ids = list(
                annotation["annotations"][take_id]["object_masks"][object_name][
                    cams_str.split("__")[0]
                ]["annotation"].keys()
            )

            for f_id in f_ids:
                f_str = f"{f_id}.json"
                pred_mask_path = os.path.join(
                    pred, take_id, cams_str, object_name, f_str
                )
                if not os.path.isfile(pred_mask_path):
                    continue

                with open(pred_mask_path, "r") as fp:
                    pred_mask_data = json.load(fp)

                pred_masks[object_name][cams_str][f_str.split(".")[0]] = pred_mask_data
    return {"masks": pred_masks, "subsample_idx": f_ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="EgoExo4D take data root", default="/data/seg/LM-EEC/egoexo_val")
    parser.add_argument("--gt", help="EgoExo4D gt annotations file path", default="/data/seg/EgoExo4D/relations_val.json")
    parser.add_argument("--pred", help="EgoExo4D predicted results", default="/data/seg/LM-EEC/egoexo_val")
    args = parser.parse_args()
    main(args)
