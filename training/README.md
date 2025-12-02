# Training Code for EgoExo-SAM2

## Getting Started

1. 进入项目文件夹，配置环境：
    ```bash
   conda env create -f environment.yml
     ```
   这里我打包了我本地的conda环境，CUDA版本是11.8,若配置失败，可参考SAM 2的配置流程https://github.com/facebookresearch/sam2
2. 进入环境
    ```bash
    source activate sam2
     ```
3. 下载预训练的Checkpoints文件
    ```bash
    cd checkpoints && \
    ./download_ckpts.sh && \
    cd ..
   ```
4. 修改项目中的配置
- training/train.py文件中第274行可修改gpu使用数目
- 修改 /sam2/configs/sam2.1_hiera_b+_EgoExo_finetune.yaml文件中第16行为下载的数据集的train文件夹的地址：
   ```yaml
  img_folder: /data/seg/Ego-Exo4D/processed_xmem/train
   ```
- /sam2/configs/sam2.1_hiera_b+_EgoExo_finetune.yaml文件中第5行可修改batch_size数：
   ```yaml
  train_batch_size: 8 
   ```
- /sam2/configs/sam2.1_hiera_b+_EgoExo_finetune.yaml文件中第311行修改模型参数的存储路径：
   ```yaml
  checkpoint:
    save_dir: /data/seg/sam2-main/training/checkpoints_EgoExo
   ```
- /sam2/configs/sam2.1_hiera_b+_EgoExo_finetune.yaml文件中第322行修改模型预训练参数的存储路径，也就是刚才下载好的模型参数路径：
   ```yaml
      state_dict:
        _target_: training.utils.checkpoint_utils.load_checkpoint_and_apply_kernels
        checkpoint_path: /data/seg/sam2-main/checkpoints/sam2.1_hiera_base_plus.pt # PATH to SAM 2.1 checkpoint
   ```
  
5. 运行代码
   ```bash
    python training/train.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_EgoExo_finetune.yaml \
        --use-cluster 0 \
        --num-gpus 8
     ```