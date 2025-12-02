import torch
import os
import json
from collections import Counter
from fontTools.subset import subset
from pycocotools import mask as mask_utils
import torchvision.transforms.functional as F
from sympy.physics.units import percent
import matplotlib.pyplot as plt
import numpy as np
# 加载检查点
# checkpoint = torch.load('/data/seg/sam2-main/training/checkpoints_EgoExo20.pth')
# print(checkpoint.keys())  # 输出字典的键
#
# checkpoint1 = torch.load('/data/seg/sam2-main/checkpoints_EgoExo/checkpoint.pt')
# print(checkpoint1['loss'])

num = 0
num_all = 0
# 初始化一个 Counter 来统计 mask1_sum 的数量
mask1_sum_counter = Counter()
img_folder = "/data/seg/Ego-Exo4D/processed_xmem/train"
# subset = os.listdir("/data/seg/Ego-Exo4D/processed_xmem/train")
# subset = ["168d649c-bb35-4401-9373-225f0731b508"]
# 设置根目录路径
root_dir = "/data/seg/Ego-Exo4D/processed_xmem/train"

# 获取该目录下的所有子项
subset = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for take_id in subset:
    annotation_path = os.path.join(img_folder, take_id, "annotation.json")
    with open(annotation_path, "r") as fp:
        annotation = json.load(fp)
    masks = annotation["masks"]
    for object_name, cams in masks.items():
        ego_cams = [x for x in masks[object_name].keys() if "aria" in x]
        if len(ego_cams) < 1:
            continue
        ego_cam_name = ego_cams[0]
        # ego_frames = list(cams[ego_cam_name].keys())
        for cam_name, cam_data in cams.items():
            if cam_name==ego_cam_name:
                continue
            # print("cam_name:", cam_name)
            exo_gt_data = masks[object_name][cam_name]
            for frame_id, frame_gt in exo_gt_data.items():
                num_all += 1
                mask = mask_utils.decode(frame_gt)
                mask = torch.from_numpy(mask)
                # print("mask:", mask.shape)
                # torch.Size([2160, 3840])
                mask_sum = mask.sum()
                # print("mask_sum ", mask_sum)
                mask1 = F.resize(mask.unsqueeze(0).unsqueeze(0), [480, 480]).to(torch.bool)
                mask1 = mask1.to(torch.uint8)
                # mask1_sum = mask1.sum()
                # 统计 mask1_sum
                mask1_sum = mask1.sum().item()  # 获取 mask1_sum 的数值
                mask1_sum_counter[mask1_sum] += 1  # 更新统计结果
                # print("mask1_sum ", mask1_sum)
                # mask2 = torch.nn.functional.interpolate(
                #     mask1,
                #     size=(int(mask1.size(2)/2), int(mask1.size(3)/2)),
                #     # align_corners=False,
                #     mode="nearest",
                #     # antialias=True,  # use antialias for downsampling
                # )
                # mask2_sum = mask2.sum()
                """
                # 转换为 NumPy 数组以便可视化
                mask2 = mask2.squeeze(0).squeeze(0)  # 移除 batch 和 channel 维度
                mask2 = mask2.cpu().numpy()  # 确保将数据转移到 CPU

                # 使用 matplotlib 显示 mask2
                plt.imshow(mask2, cmap='gray')  # 使用灰度色图
                plt.colorbar()  # 可选，显示颜色条
                plt.title("Visualized Mask2")
                plt.show()
                # print("mask2_sum ", mask2_sum)
                """
                # if mask2_sum == 0:
                #     print("object_name:", object_name)
                #     print("cam_name:", cam_name)
                #     print("mask:", mask.shape)
                #     print("mask_sum ", mask_sum)
                #     print("mask1_sum ", mask1_sum)
                #     print("mask2_sum ", mask2_sum)
                #     num += 1
# 输出所有统计结果
print(f"Total number of masks: {num_all}")
print("Mask1 sum distribution:")
for mask_sum, count in mask1_sum_counter.items():
    print(f"mask1_sum = {mask_sum}, count = {count}")

# 保存 mask1_sum_counter 到文件
output_file = "/data/seg/sam2-main/training/mask1_sum_counter.json"
with open(output_file, "w") as f:
    json.dump(mask1_sum_counter, f)
print("num_all:", num_all)


# data = np.arange(16*2).reshape((4,2,2,2))
# tensor = torch.tensor(data)
# print(tensor)
# data1 = tensor.permute(1, 0, 2, 3).reshape(-1, 2, 2)
# print(data1)


# import torch
# import torch.nn.functional as F
#
# # 创建一个随机的相似性张量 [B, h*w, N-n]，假设 B=2，h*w=3，N-n=4
# similarity = torch.randn(2, 3, 4)
#
# # 打印原始相似性张量
# print("Original similarity tensor:")
# print(similarity)
#
# # 应用 softmax 操作，在 dim=1 维度上
# weights = F.softmax(similarity, dim=1)
#
# # 打印 softmax 后的结果
# print("\nSoftmax applied on similarity tensor:")
# print(weights)
#
# # 验证 softmax 后的每个位置的概率和是否为 1
# print("\nSum of probabilities along dim=1 (should be 1 for each row):")
# print(weights.sum(dim=1))  # 计算沿 dim=1 的和，应该输出一个全是 1 的张量

"""
segswap_path = "/data/seg/ego-exo4d-relation-main/correspondence/SegSwap/egoexo_result/ego-exo_val_results.json"
# 打开并加载 JSON 文件
with open(segswap_path, 'r') as file:
    data = json.load(file)

keys1 = data["ego-exo"]["results"].keys()
print(keys1)
"""

