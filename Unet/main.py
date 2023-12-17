import torch_npu  # noqa:F401
import torch
import albumentations
import torch.optim
import torch.utils.data
import sys
import os
import segmentation_models_pytorch as smp
import time
# flake8: noqa: E402

from utils import (AverageMeter, print_epoch_stats, save_model, to_device,
                   create_data_loader, generate_paths, split_data,
                   download_smp_model)
from data_processing import Sentinel1_Dataset
from loss_metrics import XEDiceLoss, tp_fp_fn
from train import train_epochs
from model import FloodwaterSegmentationModel
#
BASE_PATH = "/home/HW/Pein/Floodwater/dataset"
TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.1
TEST_PERCENTAGE = round(1 - TRAIN_PERCENTAGE - VAL_PERCENTAGE, 2)
MIN_NORMALIZE = -77
MAX_NORMALIZE = 26
TRAIN_CROP_SIZE = 256

label_replacements = [("LabelWater.tif", "LabelWater.tif")]
image_replacements = [("LabelWater.tif", "VV.tif"),
                      ("LabelWater.tif", "VH.tif")]

label_paths = generate_paths(BASE_PATH, "chips/*/s1/*/LabelWater.tif",
                             label_replacements)
image_paths = generate_paths(BASE_PATH, "chips/*/s1/*/LabelWater.tif",
                             image_replacements)

label_splits = split_data(label_paths, TRAIN_PERCENTAGE, VAL_PERCENTAGE)
image_splits = split_data(image_paths, TRAIN_PERCENTAGE, VAL_PERCENTAGE)

# 数据增强
train_transforms = albumentations.Compose([
    albumentations.RandomCrop(TRAIN_CROP_SIZE, TRAIN_CROP_SIZE),
    albumentations.RandomRotate90(),
    albumentations.HorizontalFlip(),
    albumentations.VerticalFlip()
])

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

# 常量定义
NUM_WORKERS = 8
PIN_MEMORY = False
BATCH_SIZE = 64
EPS = 1e-7
PATIENCE = 5
N_EPOCHS = 200
LEARNING_RATE = 1e-2
EARLY_STOP_THRESHOLD = 5
EARLY_STOP_PATIENCE = PATIENCE * 5

EXPERIMENT_NAME = "Exp2_Unet_res34"
DEVICE = 'npu:0' if torch.npu.is_available() else 'cpu'
LOG_PATH = f"/home/HW/Pein/Floodwater/trained_models/{EXPERIMENT_NAME}"
os.makedirs(LOG_PATH, exist_ok=True)

# 模型参数
MODEL_PARAMS = {
    "model_name": "Unet",  # 模型类型
    "encoder_name": "resnet34",  # 编码器名称
    "encoder_weights": "imagenet",  # 编码器预训练权重
    "in_channels": 2,  # 输入通道数 (VV 和 VH)
    "classes": 2  # 输出类别数 (二分类)
}

#确认预训练模型已下载
download_smp_model(MODEL_PARAMS)

#训练参数
EARLY_STOP_THRESHOLD = 5
EARLY_STOP_PATIENCE = PATIENCE * 5

#NPU设置
torch.npu.set_device(DEVICE)
if torch.npu.is_available() and DEVICE.startswith('npu'):
    print(f'Using NPU: {DEVICE} ...')
else:
    print('Using CPU ...')

# 打印基本信息
print(f"""
NUM_WORKERS = {NUM_WORKERS}
PIN_MEMORY = {PIN_MEMORY}
BATCH_SIZE = {BATCH_SIZE}
EPS = {EPS}
EXPERIMENT_NAME = '{EXPERIMENT_NAME}'
BACKBONE = '{BACKBONE}'
PATIENCE = {PATIENCE}
N_EPOCHS = {N_EPOCHS}
LEARNING_RATE = {LEARNING_RATE}
EARLY_STOP_THRESHOLD = {EARLY_STOP_THRESHOLD}
EARLY_STOP_PATIENCE = {EARLY_STOP_PATIENCE}
""")

# 加载训练集、验证集和测试集
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

# 确认预训练模型已下载
download_smp_model(model_name="Unet",
                   encoder_name=encoder_name,
                   encoder_weights=encoder_weights)

# 模型初始化
model = FloodwaterSegmentationModel(**MODEL_PARAMS).to(DEVICE)

# 损失函数和优化器
loss_func = XEDiceLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode="max",
                                                 factor=0.5,
                                                 patience=PATIENCE,
                                                 verbose=True)

# 训练指标初始化
best_metric, best_metric_epoch = 0, 0
early_stop_counter, best_val_early_stopping_metric = 0, 0

# 开始训练
train_result = train_epochs(model, train_loader, val_loader, optimizer,
                            loss_func, EXPERIMENT_NAME, LOG_PATH, scheduler,
                            N_EPOCHS, EARLY_STOP_THRESHOLD,
                            EARLY_STOP_PATIENCE, DEVICE, EPS)

# 打印或处理返回的信息
print("Training Completed:")
print(
    f"Best IOU: {train_result['best_metric']} at Epoch: {train_result['best_metric_epoch']}"
)
print(f"Total Training Time: {train_result['training_time']} seconds")
