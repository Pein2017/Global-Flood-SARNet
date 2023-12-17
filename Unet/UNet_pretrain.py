# flake8: noqa: E501 , F401
import glob
import tifffile
import matplotlib.pyplot as plt
import torch_npu
import torch
import torch.nn as nn
import albumentations
import torch.optim
import torch.utils.data
import sys
import os
import segmentation_models_pytorch as smp
import time
import numpy as np


def visualize_s2_img(s2_channel_paths):
    """
    Calculates and returns a 'SWIR' S2 false color composite ready for visualization.

    Args:
        s2_channel_paths (list of str): Paths to the ['B12', 'B7', 'B4'] bands of a S2 image.

    Returns:
        np.array: Image (H, W, 3) ready for visualization with e.g. plt.imshow(..)
    """
    img = []
    for path in s2_channel_paths:
        img.append(tifffile.imread(path))
    img = np.stack(img, axis=-1)
    return scale_S1_S2_img(img, sentinel=2)


def visualize_s1_img(path_vv, path_vh):
    """
    Calculates and returns a S1 false color composite ready for visualization.

    Args:
        path_vv (str): Path to the VV band.
        path_vh (str): Path to the VH band.

    Returns:
        np.array: Image (H, W, 3) ready for visualization with e.g. plt.imshow(..)
    """
    s1_img = np.stack((tifffile.imread(path_vv), tifffile.imread(path_vh)),
                      axis=-1)
    img = np.zeros((s1_img.shape[0], s1_img.shape[1], 3), dtype=np.float32)
    img[:, :, :2] = s1_img.copy()
    img[:, :, 2] = s1_img[:, :, 0] / s1_img[:, :, 1]
    s1_img = np.nan_to_num(s1_img)
    return scale_S1_S2_img(img, sentinel=1)


def scale_S1_S2_img(matrix, sentinel=2):
    """
    Returns a scaled (H,W,D) image which is more easily visually inspectable. Image is linearly scaled between
    min and max_value of by channel
    
    Args:
        matrix (np.array): (H,W,D) image to be scaled
        sentinel (int, optional): Sentinel 1 or Sentinel 2 image? Determines the min and max values for scalin, defaults to 2.

    Returns:
        np.array: Image (H, W, 3) ready for visualization with e.g. plt.imshow(..)
    """
    w, h, d = matrix.shape
    min_values = np.array([100, 100, 100]) if sentinel == 2 else np.array(
        [-23, -28, 0.2])
    max_values = np.array([3500, 3500, 3500]) if sentinel == 2 else np.array(
        [0, -5, 1])

    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    matrix = (matrix - min_values[None, :]) / (max_values[None, :] -
                                               min_values[None, :])
    matrix = np.reshape(matrix, [w, h, d])

    matrix = matrix.clip(0, 1)
    return matrix


#
BASE_PATH = "/home/HW/Pein/Floodwater/dataset"
TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1
MIN_NORMALIZE = -77
MAX_NORMALIZE = 26
TRAIN_CROP_SIZE = 256


# Generate image and label paths
def generate_paths(base_path, pattern, replacements):
    paths = sorted(glob.glob(f"{base_path}/{pattern}"))
    return [
        list(map(lambda r: path.replace(*r), replacements)) for path in paths
    ]


def split_data(paths, train_perc, val_perc):
    np.random.seed(17)
    np.random.shuffle(paths)
    n = len(paths)
    return {
        "train": paths[:int(train_perc * n)],
        "val": paths[int(train_perc * n):int((train_perc + val_perc) * n)],
        "test": paths[int((train_perc + val_perc) * n):]
    }


label_replacements = [("LabelWater.tif", "LabelWater.tif")]
image_replacements = [("LabelWater.tif", "VV.tif"),
                      ("LabelWater.tif", "VH.tif")]

label_paths = generate_paths(BASE_PATH, "chips/*/s1/*/LabelWater.tif",
                             label_replacements)
image_paths = generate_paths(BASE_PATH, "chips/*/s1/*/LabelWater.tif",
                             image_replacements)

label_splits = split_data(label_paths, TRAIN_PERCENTAGE, VAL_PERCENTAGE)
image_splits = split_data(image_paths, TRAIN_PERCENTAGE, VAL_PERCENTAGE)


# Dataset class
class Sentinel1_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        img_paths,
        mask_paths,
        transforms=None,
        min_normalize=-77,
        max_normalize=26,
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.min_normalize = min_normalize
        self.max_normalize = max_normalize

    def __getitem__(self, idx):
        # Load in image
        arr_x = []
        for path in self.img_paths[idx]:
            arr_x.append(tifffile.imread(path))
        arr_x = np.stack(arr_x, axis=-1)
        # Min-Max Normalization
        arr_x = np.clip(arr_x, self.min_normalize, self.max_normalize)
        arr_x = (arr_x - self.min_normalize) / (self.max_normalize -
                                                self.min_normalize)

        sample = {"image": arr_x}

        # Load in label mask
        sample["mask"] = tifffile.imread(self.mask_paths[idx])

        # Apply Data Augmentation
        if self.transforms:
            sample = self.transforms(image=sample["image"],
                                     mask=sample["mask"])
        if sample["image"].shape[-1] < 20:
            sample["image"] = sample["image"].transpose((2, 0, 1))

        return sample

    def __len__(self):
        return len(self.img_paths)

    def visualize(self, how_many=1, show_specific_index=None):
        for _ in range(how_many):
            rand_int = np.random.randint(len(self.img_paths))
            if show_specific_index is not None:
                rand_int = show_specific_index
            print(self.img_paths[rand_int][0])
            f, axarr = plt.subplots(1, 3, figsize=(30, 9))
            axarr[0].imshow(
                visualize_s1_img(self.img_paths[rand_int][0],
                                 self.img_paths[rand_int][1]))
            sample = self.__getitem__(rand_int)

            img = sample["image"]
            axarr[0].set_title(f"FCC of original S1 image",
                               fontsize=15)  # noqa
            axarr[1].imshow(img[0])  # Just visualize the VV band here
            axarr[1].set_title(
                f"VV band returned from the dataset, Min: {img.min():.4f}, Max: {img.max():.4f}",
                fontsize=15)
            if "mask" in sample.keys():
                axarr[2].set_title(
                    f"Corresponding water mask: {(sample['mask'] == 1).sum()} px",
                    fontsize=15)
                mask = mask_to_img(sample["mask"], {1: (0, 0, 255)})
                axarr[2].imshow(mask)
            plt.tight_layout()
            plt.show()


def mask_to_img(label, color_dict):
    """Recodes a (H,W) mask to a (H,W,3) RGB image according to color_dict"""
    mutually_exclusive = np.zeros(label.shape + (3, ), dtype=np.uint8)
    for key in range(1, len(color_dict.keys()) + 1):
        mutually_exclusive[label == key] = color_dict[key]
    return mutually_exclusive


# Data Augmentation
train_transforms = albumentations.Compose([
    albumentations.RandomCrop(TRAIN_CROP_SIZE, TRAIN_CROP_SIZE),
    albumentations.RandomRotate90(),
    albumentations.HorizontalFlip(),
    albumentations.VerticalFlip()
])

# Dataset initialization
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

# Output
print(len(train_dataset), len(val_dataset), len(test_dataset))
print(
    f'TRAIN_PERCENTAGE: {TRAIN_PERCENTAGE} , VAL_PERCENTAGE: {VAL_PERCENTAGE} , TEST_PERCENTAGE: {TEST_PERCENTAGE}'
)

# Constants
NUM_WORKERS = 8
PIN_MEMORY = False
BATCH_SIZE = 128
EPS = 1e-7
EXPERIMENT_NAME = "Exp1_res34"
BACKBONE = "resnet34"
PATIENCE = 5
N_EPOCHS = 300
LEARNING_RATE = 1e-3
LOG_PATH = f"/home/HW/Pein/Floodwater/trained_models/{EXPERIMENT_NAME}"
DEVICE = 'npu' if torch.npu.is_available() else 'cpu'
EARLY_STOP_THRESHOLD = 5
EARLY_STOP_PATIENCE = PATIENCE * 5


# DataLoaders
def create_data_loader(dataset,
                       batch_size,
                       shuffle,
                       num_workers=NUM_WORKERS,
                       pin_memory=PIN_MEMORY):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)


train_loader = create_data_loader(train_dataset, BATCH_SIZE, True)
val_loader = create_data_loader(val_dataset, BATCH_SIZE, False)
test_loader = create_data_loader(test_dataset, BATCH_SIZE, False)


class XEDiceLoss(nn.Module):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        xe_loss = self.cross_entropy(preds, targets)
        dice_loss = self.calculate_dice_loss(preds, targets)
        return self.alpha * xe_loss + (1 - self.alpha) * dice_loss

    @staticmethod
    def calculate_dice_loss(preds, targets):
        targets = targets.float()
        preds = torch.softmax(preds, dim=1)[:, 1]
        intersection = torch.sum(preds * targets)
        union = torch.sum(preds + targets)
        return 1 - (2.0 * intersection) / (union + EPS)


# Metric Calculation
def tp_fp_fn(preds, targets):
    tp = torch.sum(preds * targets)
    fp = torch.sum(preds) - tp
    fn = torch.sum(targets) - tp
    return tp.item(), fp.item(), fn.item()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py    # noqa: E501
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Setup
os.makedirs(LOG_PATH, exist_ok=True)
model = smp.Unet(encoder_name=BACKBONE,
                 encoder_weights="imagenet",
                 in_channels=2,
                 classes=2).to(DEVICE)

loss_func = XEDiceLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode="max",
                                                       factor=0.1,
                                                       patience=PATIENCE,
                                                       verbose=True)

if torch.npu.is_available() and DEVICE == 'npu':
    print('Using NPU ...')
else:
    print('Using CPU ...')


# Helper function for moving data to device
def to_device(data, device):
    if device == 'npu':
        return {k: v.to(device, non_blocking=True) for k, v in data.items()}
    return data


# Utility functions
def print_epoch_stats(phase,
                      epoch,
                      loss_meter,
                      batch_time,
                      data_time,
                      start_time,
                      iou=None):
    if iou:
        print(
            f"Epoch: [{epoch}] {phase} - TotalT: {(time.time() - start_time) / 60:.1f} min, "  # noqa: E501
            f"Loss: {loss_meter.avg:.4f}, Global IoU: {iou:.4f}")
    else:
        print(
            f"Epoch: [{epoch}] {phase} - TotalT: {(time.time() - start_time) / 60:.1f} min, "  # noqa: E501
            f"BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, Loss: {loss_meter.avg:.4f}"  # noqa: E501
        )


def save_model(epoch, iou):
    global best_metric_epoch, best_metric
    save_dict = {
        "model_name": EXPERIMENT_NAME,
        "epoch_num": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "iou": iou
    }
    old_model_path = glob.glob(f"{LOG_PATH}/best_iou*")
    if old_model_path:
        os.remove(old_model_path[0])
    torch.save(save_dict, f"{LOG_PATH}/best_iou_{epoch}_{iou:.4f}.pt")


# Training and Validation Loop
best_metric, best_metric_epoch = 0, 0
early_stop_counter, best_val_early_stopping_metric = 0, 0
start_time = time.time()

# record starting time
start_time = time.time()
for curr_epoch_num in range(1, N_EPOCHS + 1):
    # Part 1: Training
    # 保存原始stdout
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # 调用 model.train()
    model.train()

    #   恢复原始stdout
    sys.stdout = original_stdout

    train_loss, batch_time, data_time = AverageMeter(), AverageMeter(
    ), AverageMeter()
    end = time.time()
    for data in train_loader:
        data_time.update(time.time() - end)
        data = to_device(data, DEVICE)
        optimizer.zero_grad()
        preds = model(data["image"])
        loss = loss_func(preds, data["mask"].long())
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), data["image"].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    # Print training metrics
    print_epoch_stats('Train', curr_epoch_num, train_loss, batch_time,
                      data_time, start_time)

    # Part 2: Validation
    model.eval()
    val_loss, tps, fps, fns = AverageMeter(), 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            data = to_device(data, DEVICE)
            preds = model(data["image"])
            loss = loss_func(preds, data["mask"].long())
            val_loss.update(loss.item(), data["image"].size(0))
            preds_binary = (torch.softmax(preds, dim=1)[:, 1] > 0.5).long()
            tp, fp, fn = tp_fp_fn(preds_binary, data["mask"])
            tps += tp
            fps += fp
            fns += fn

    iou_global = tps / (tps + fps + fns + EPS)
    print_epoch_stats('Val', curr_epoch_num, val_loss, batch_time, data_time,
                      start_time, iou_global)

    # Scheduler and Model Saving
    if curr_epoch_num > EARLY_STOP_THRESHOLD:
        scheduler.step(iou_global)
    if iou_global > best_metric:
        best_metric, best_metric_epoch = iou_global, curr_epoch_num
        save_model(curr_epoch_num, iou_global)
    if iou_global < best_val_early_stopping_metric:
        early_stop_counter += 1
    else:
        best_val_early_stopping_metric, early_stop_counter = iou_global, 0
    if early_stop_counter > EARLY_STOP_PATIENCE:
        print("Early Stopping")
        break

# record end time
end_time = time.time()
print('finish training')
print('total time is ', end_time - start_time)
print(
    f"Best validation IoU of {best_metric:.5f} in epoch {best_metric_epoch}.")
