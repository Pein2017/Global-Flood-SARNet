import time
import torch
import glob
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch.optim as optim
# flake8: noqa: E501


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


def to_device(data, device):
    if device.startswith('npu'):
        return {k: v.to(device, non_blocking=True) for k, v in data.items()}
    else:
        Warning("NPU Device is not supported")
    return data


def print_epoch_stats(phase, epoch, loss_meter, batch_time, data_time, epoch_start_time, iou=None):
    epoch_duration = time.time() - epoch_start_time
    if iou:
        print(
            f"Epoch: [{epoch}] {phase} - EpochT: {epoch_duration / 60:.1f} min, "
            f"Loss: {loss_meter.avg:.4f}")
        print(f"Global IOU: {iou:.4f}")
    else:
        print(
            f"Epoch: [{epoch}] {phase} - EpochT: {epoch_duration / 60:.1f} min, "
            f"BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, Loss: {loss_meter.avg:.4f}"
        )


def save_model(save_dict, log_path):
    torch.save(save_dict, f"{log_path}/best_model.pt")


def create_data_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)


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


def download_smp_model(model_params):
    model = getattr(smp, model_params["model_name"])(
        encoder_name=model_params["encoder_name"],
        encoder_weights=model_params["encoder_weights"],
        in_channels=model_params["in_channels"],
        classes=model_params["classes"])
    print(
        f"Downloaded and cached {model_params['model_name']} model with {model_params['encoder_name']} encoder."
    )

def initialize_training(checkpoint_path, model, optimizer, scheduler_params, train_mode):
    best_metric = 0
    best_metric_epoch = 0

    if train_mode == 'continue':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        best_metric = checkpoint['iou']
        best_metric_epoch = checkpoint['epoch_num']
        optimizer.param_groups[0]['lr'] = 0.5 * optimizer.param_groups[0]['lr']

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

    return model, optimizer, scheduler, best_metric, best_metric_epoch





def load_model(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    best_metric = checkpoint['iou']
    best_metric_epoch = checkpoint['epoch_num']

    return model, optimizer, best_metric, best_metric_epoch