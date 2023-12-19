import torch_npu
import torch
import albumentations
import torch.optim
import torch.utils.data
import sys
import os
import torch.optim as optim
import segmentation_models_pytorch as smp
import time
from utils import (AverageMeter, print_epoch_stats, save_model, to_device,
                   create_data_loader, generate_paths, split_data,
                   download_smp_model, initialize_training, load_model)
from data_processing import Sentinel1_Dataset
from loss_metrics import XEDiceLoss, tp_fp_fn
from train import train_epochs, adjust_threshold_and_evaluate, soft_iou
from model import FloodwaterSegmentationModel
from config import *
# flake8: noqa

# 定义文件路径替换
label_replacements = [("LabelWater.tif", "LabelWater.tif")]
image_replacements = [("LabelWater.tif", "VV.tif"),
                      ("LabelWater.tif", "VH.tif")]

# 生成文件路径
label_paths = generate_paths(BASE_PATH, "chips/*/s1/*/LabelWater.tif",
                             label_replacements)
image_paths = generate_paths(BASE_PATH, "chips/*/s1/*/LabelWater.tif",
                             image_replacements)

# 数据拆分
label_splits = split_data(label_paths, TRAIN_PERCENTAGE, VAL_PERCENTAGE)
image_splits = split_data(image_paths, TRAIN_PERCENTAGE, VAL_PERCENTAGE)

# Sentinel1_Dataset初始化
train_dataset = Sentinel1_Dataset(image_splits['train'],
                                  label_splits['train'],
                                  transforms=train_transforms,
                                  min_normalize=MIN_NORMALIZE,
                                  max_normalize=MAX_NORMALIZE)
val_dataset = Sentinel1_Dataset(image_splits['val'],
                                label_splits['val'],
                                transforms=None,
                                min_normalize=MIN_NORMALIZE,
                                max_normalize=MAX_NORMALIZE)
test_dataset = Sentinel1_Dataset(image_splits['test'],
                                 label_splits['test'],
                                 transforms=None,
                                 min_normalize=MIN_NORMALIZE,
                                 max_normalize=MAX_NORMALIZE)

# 其他设置（NPU、文件夹创建、模型下载等）
os.makedirs(LOG_PATH, exist_ok=True)
download_smp_model(MODEL_PARAMS)

#NPU设置
torch.npu.set_device(DEVICE)
if torch.npu.is_available() and DEVICE.startswith('npu'):
    print(f'Using NPU: {DEVICE} ...')
else:
    print('Using CPU ...')

# 打印基本信息
print(f"""
TRAIN_CROP_SIZE = {TRAIN_CROP_SIZE}
TARGET_SIZE = {TARGET_SIZE}
NUM_WORKERS = {NUM_WORKERS}
PIN_MEMORY = {PIN_MEMORY}
BATCH_SIZE = {BATCH_SIZE}
EPS = {EPS}
EXPERIMENT_NAME = '{EXPERIMENT_NAME}'
ENCODER_NAME = '{MODEL_PARAMS['encoder_name']}'
MODEL_NAME = '{MODEL_PARAMS['model_name']}'
ENCODER_WEIGHTS = '{MODEL_PARAMS['encoder_weights']}'
IN_CHANNELS = {MODEL_PARAMS['in_channels']}
CLASSES = {MODEL_PARAMS['classes']}
PATIENCE = {PATIENCE}
N_EPOCHS = {N_EPOCHS}
LEARNING_RATE = {LEARNING_RATE}
EARLY_STOP_THRESHOLD = {EARLY_STOP_THRESHOLD}
EARLY_STOP_PATIENCE = {EARLY_STOP_PATIENCE}
""")

# 加载数据
train_loader = create_data_loader(train_dataset,
                                  BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=PIN_MEMORY)
val_loader = create_data_loader(val_dataset,
                                BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY)
test_loader = create_data_loader(test_dataset,
                                 BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=PIN_MEMORY)

# 模型初始化
model = FloodwaterSegmentationModel(**MODEL_PARAMS).to(DEVICE)
loss_func = XEDiceLoss().to(DEVICE)

# 优化器初始化
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Scheduler 参数
scheduler_params = {
    "mode": "min",
    "factor": 0.5,
    "patience": PATIENCE,
    "verbose": True
}

# 使用 initialize_training 函数初始化训练环境
model, optimizer, scheduler, best_metric, best_metric_epoch = initialize_training(
    CHECKPOINT_PATH, model, optimizer, scheduler_params, TRAIN_MODE)

if TRAIN_MODE == 'continue':
    print(
        f"Continuing training: Learning rate: {optimizer.param_groups[0]['lr']}, Best IOU: {best_metric} at epoch: {best_metric_epoch}"
    )

# 早停相关变量初始化
early_stop_counter = 0
start_epoch = best_metric_epoch if TRAIN_MODE == 'continue' else 1

train_result = train_epochs(model, train_loader, val_loader, optimizer,
                            loss_func, EXPERIMENT_NAME, LOG_PATH, scheduler,
                            N_EPOCHS, EARLY_STOP_THRESHOLD,
                            EARLY_STOP_PATIENCE, DEVICE, EPS, start_epoch,
                            best_metric, best_metric_epoch)

# 打印或处理返回的信息
print('--' * 20)
print("Training Completed:")
print(f"Total Training Time: {train_result['training_time'] / 60} minutes")
# print(f"Total Training Time: {train_result['training_time']} seconds")
