import torch
import torch_npu
import albumentations
# flake8: noqa

# 数据集和路径相关
BASE_PATH = "/home/HW/Pein/Floodwater/dataset"
TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.1
TEST_PERCENTAGE = round(1 - TRAIN_PERCENTAGE - VAL_PERCENTAGE, 2)

# 数据预处理和增强
MIN_NORMALIZE = -77
MAX_NORMALIZE = 26
TRAIN_CROP_SIZE = 256
TARGET_SIZE = 256

# 数据增强配置
train_transforms = albumentations.Compose([
    albumentations.RandomCrop(TRAIN_CROP_SIZE, TRAIN_CROP_SIZE),
    # RandomResizedCrop(TARGET_SIZE, TARGET_SIZE, scale=(0.75, 1.0), p=0.5),
    albumentations.RandomBrightnessContrast(brightness_limit=0.2,
                                            contrast_limit=0.2),
    albumentations.GaussNoise(var_limit=(10, 50)),
    albumentations.RandomRotate90(),
    albumentations.HorizontalFlip(),
    albumentations.VerticalFlip(),
    albumentations.Resize(TARGET_SIZE, TARGET_SIZE),
])

# 模型和训练相关参数
NUM_WORKERS = 4
PIN_MEMORY = False
BATCH_SIZE = 24
EPS = 1e-7

PATIENCE = 6
N_EPOCHS = 200
LEARNING_RATE = 1e-4
EARLY_STOP_THRESHOLD = 10  # scheduler调整的阈值
EARLY_STOP_PATIENCE = PATIENCE * 5  # 早停的耐心周期

# 模型参数
MODEL_PARAMS = {
    "model_name": "PAN",
    "encoder_name": "resnet101",
    "encoder_weights": "imagenet",
    "in_channels": 2,
    "classes": 2
}

# 设备和日志相关
DEVICE = 'npu:1' if torch.npu.is_available() else 'cpu'
EXPERIMENT_NAME = f"{MODEL_PARAMS['model_name']}-{MODEL_PARAMS['encoder_name']}-b{BATCH_SIZE}"
LOG_PATH = f"/home/HW/Pein/Floodwater/SMP/trained_models/{EXPERIMENT_NAME}"

# 训练模式：'new' 从头开始，'continue' 从checkpoint继续
TRAIN_MODE = 'new'  # 或 'continue'

# 如果是 'tuning' 模式，指定checkpoint路径
CHECKPOINT_PATH = '/home/HW/Pein/Floodwater/SMP/trained_models/Exp2-Unet-resnet101-b64/best_model.pt'
