"""
Usecase 3 Training Script

Usage:
    train.py [options]

Options:
    -h --help          Show this screen
    --summary          Only print model summary and return (Requires the torchsummary package)
    --resume=CKPT      Resume from checkpoint
    --config=CONFIG    Specify run config to use [default: config.yml]
"""
import sys, shutil, random, yaml
from datetime import datetime
from pathlib import Path
from docopt import docopt
from tqdm import tqdm
from data_loading import get_dataset

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from einops import reduce
from torch.utils.tensorboard import SummaryWriter


try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError as e:
    from torch.optim import Adam

from deep_learning import get_loss, get_model, Metrics, flatui_cmap
from deep_learning.utils.data import Augment

writer = None
train_history = []
val_history = []
test_history = []
brightness_label = "Brightness: none"
train_sampler = None


def format_brightness_label(aug_cfg):
    if not aug_cfg:
        return "Brightness: none"
    brightness_range = aug_cfg.get('brightness_range')
    if brightness_range is not None:
        return f"Brightness: range={brightness_range}"
    brightness = aug_cfg.get('brightness', 0.0)
    if brightness and brightness > 0:
        return f"Brightness: +/-{brightness}"
    return "Brightness: none"


def build_balanced_sampler(dataset, indices, augmented_len, cfg):
    if not hasattr(dataset, 'tiles'):
        return None

    eps = cfg.get('balanced_sampling_eps', 1e-4)
    power = cfg.get('balanced_sampling_power', 1.0)
    max_weight = cfg.get('balanced_sampling_max_weight')
    base_weights = []
    for idx in indices:
        data = np.load(dataset.tiles[idx])
        if 'gt' not in data:
            return None
        mask = data['gt']
        if mask.ndim >= 3:
            mask = mask[0]
        pos_frac = float(mask.mean())
        weight = (pos_frac + eps) ** (-power)
        if max_weight is not None:
            weight = min(weight, max_weight)
        base_weights.append(weight)

    if not base_weights:
        return None

    repeat_factor = max(1, augmented_len // len(base_weights))
    weights = np.repeat(base_weights, repeat_factor)
    if len(weights) < augmented_len:
        pad = augmented_len - len(weights)
        weights = np.concatenate([weights, weights[:pad]])
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=augmented_len,
        replacement=True,
    )


def write_split_files(dataset, split_map, output_dir):
    if not hasattr(dataset, 'tiles'):
        print("Dataset has no tile list; skipping split file export.")
        return
    for split_name, split_indices in split_map.items():
        out_path = output_dir / f"{split_name}_split.txt"
        with out_path.open('w') as f:
            for idx in split_indices:
                f.write(str(dataset.tiles[idx]) + "\n")

def showexample(idx, img, target, prediction):
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=0.08, wspace=m)
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), gridspec_kw=gridspec_kw)
    fig.suptitle(brightness_label, fontsize=9)
    heatmap_seg  = dict(cmap='gray', vmin=0, vmax=1)
    heatmap_edge = dict(cmap=flatui_cmap('Clouds', 'Midnight Blue'), vmin=0, vmax=1)
    # Clear all axes
    for axis in ax.flat:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    rgb = (1. + img.cpu().numpy()) / 2.
    ax[0, 0].imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    ax[0, 0].set_title("Input RGB", fontsize=8)
    ax[0, 1].imshow(target[0].cpu(), **heatmap_seg)
    ax[0, 1].set_title("Seg GT", fontsize=8)
    ax[1, 1].imshow(target[1].cpu(), **heatmap_edge)
    ax[1, 1].set_title("Edge GT", fontsize=8)

    seg_pred, edge_pred = torch.sigmoid(prediction)
    ax[0, 2].imshow(seg_pred.cpu(), **heatmap_seg)
    ax[0, 2].set_title("Seg Pred", fontsize=8)
    ax[1, 2].imshow(edge_pred.cpu(), **heatmap_edge)
    ax[1, 2].set_title("Edge Pred", fontsize=8)

    filename = log_dir / 'figures' / f'{idx:03d}_{epoch}.jpg'
    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def init_confusion():
    return dict(tp=0, fp=0, fn=0, tn=0)


def to_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def update_confusion(confusion, seg_tp, seg_fp, seg_fn, seg_tn):
    confusion['tp'] += to_int(seg_tp)
    confusion['fp'] += to_int(seg_fp)
    confusion['fn'] += to_int(seg_fn)
    confusion['tn'] += to_int(seg_tn)


def confusion_matrix_from_counts(confusion):
    return np.array(
        [
            [confusion['tn'], confusion['fp']],
            [confusion['fn'], confusion['tp']],
        ],
        dtype=np.int64
    )

def init_hist(bins):
    return torch.zeros(bins, dtype=torch.float64), torch.zeros(bins, dtype=torch.float64)


def update_hist(pos_hist, neg_hist, seg_prob, seg_true, bins):
    probs = seg_prob.detach().flatten().cpu()
    true = seg_true.detach().flatten().cpu()
    if probs.numel() == 0:
        return
    pos_probs = probs[true == 1]
    neg_probs = probs[true == 0]
    if pos_probs.numel() > 0:
        pos_hist += torch.histc(pos_probs, bins=bins, min=0.0, max=1.0)
    if neg_probs.numel() > 0:
        neg_hist += torch.histc(neg_probs, bins=bins, min=0.0, max=1.0)


def best_threshold_from_hist(pos_hist, neg_hist):
    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()
    if (total_pos + total_neg) == 0:
        return 0.5, dict(tp=0, fp=0, fn=0, tn=0), 0.0, 0.0, 0.0

    cum_pos = torch.flip(torch.cumsum(torch.flip(pos_hist, [0]), 0), [0])
    cum_neg = torch.flip(torch.cumsum(torch.flip(neg_hist, [0]), 0), [0])
    tp = cum_pos
    fp = cum_neg
    fn = total_pos - tp
    tn = total_neg - fp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    best_idx = int(torch.argmax(f1).item())
    bins = pos_hist.numel()
    thresholds = torch.linspace(0.0, 1.0, bins + 1)[:-1]
    best_threshold = float(thresholds[best_idx].item())

    confusion = dict(
        tp=int(tp[best_idx].item()),
        fp=int(fp[best_idx].item()),
        fn=int(fn[best_idx].item()),
        tn=int(tn[best_idx].item()),
    )
    return best_threshold, confusion, float(precision[best_idx].item()), float(recall[best_idx].item()), float(f1[best_idx].item())


def save_confusion_matrix(cm, filename, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
    im = ax.imshow(cm_norm, cmap='RdPu', vmin=0.0, vmax=1.0)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized')
    for (i, j), val in np.ndenumerate(cm):
        pct = cm_norm[i, j] if row_sums[i, 0] != 0 else 0.0
        ax.text(j, i, f'{val}\n{pct:.1%}', ha='center', va='center', color='black')
    fig.tight_layout()
    output = log_dir / 'figures' / filename
    output.parent.mkdir(exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close()


def compute_binary_classification_metrics(tp, fp, fn, tn, eps=1e-6):
    precision_pos = tp / (tp + fp + eps)
    recall_pos = tp / (tp + fn + eps)
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + eps)

    precision_neg = tn / (tn + fn + eps)
    recall_neg = tn / (tn + fp + eps)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + eps)

    support_pos = tp + fn
    support_neg = tn + fp
    total = support_pos + support_neg + eps

    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    weighted_precision = ((precision_pos * support_pos) + (precision_neg * support_neg)) / total
    weighted_recall = ((recall_pos * support_pos) + (recall_neg * support_neg)) / total
    weighted_f1 = ((f1_pos * support_pos) + (f1_neg * support_neg)) / total

    specificity_pos = recall_neg
    specificity_neg = recall_pos
    macro_specificity = (specificity_pos + specificity_neg) / 2
    balanced_accuracy = (recall_pos + recall_neg) / 2
    macro_accuracy = (recall_pos + recall_neg) / 2

    def cohen_kappa(tp_k, fp_k, fn_k, tn_k):
        total_k = tp_k + fp_k + fn_k + tn_k + eps
        po_k = (tp_k + tn_k) / total_k
        pe_k = ((tp_k + fp_k) * (tp_k + fn_k) + (fn_k + tn_k) * (fp_k + tn_k)) / (total_k * total_k)
        return (po_k - pe_k) / (1 - pe_k + eps)

    kappa_pos = cohen_kappa(tp, fp, fn, tn)
    kappa_neg = cohen_kappa(tn, fn, fp, tp)
    macro_kappa = (kappa_pos + kappa_neg) / 2

    return dict(
        precision_pos=float(precision_pos),
        recall_pos=float(recall_pos),
        f1_pos=float(f1_pos),
        precision_neg=float(precision_neg),
        recall_neg=float(recall_neg),
        f1_neg=float(f1_neg),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_precision=float(weighted_precision),
        weighted_recall=float(weighted_recall),
        weighted_f1=float(weighted_f1),
        specificity=float(macro_specificity),
        balanced_accuracy=float(balanced_accuracy),
        macro_accuracy=float(macro_accuracy),
        cohen_kappa=float(macro_kappa),
    )


def get_pyramid(mask):
    with torch.no_grad():
        masks = [mask]
        ## Build mip-maps
        for _ in range(stack_height):
            # Pretend we have a batch
            big_mask = masks[-1]
            small_mask = F.avg_pool2d(big_mask, 2)
            masks.append(small_mask)

        targets = []
        for mask in masks:
            sobel = torch.any(SOBEL(mask) != 0, dim=1, keepdims=True).float()
            if config['model'] == 'HED':
                targets.append(sobel)
            else:
                targets.append(torch.cat([mask, sobel], dim=1))

    return targets


def compute_weighted_loss(y_hat, target, loss_fn, weights):
    if y_hat.shape[1] < 2 or target.shape[1] < 2:
        return loss_fn(y_hat, target)
    seg_weight = weights.get('seg', 1.0)
    edge_weight = weights.get('edge', 1.0)
    seg_loss = loss_fn(y_hat[:, 1], target[:, 0])
    edge_loss = loss_fn(y_hat[:, 0], target[:, 1])
    return seg_weight * seg_loss + edge_weight * edge_loss


def full_forward(model, img, target, metrics, seg_threshold=0.5):
    img = img.to(dev)
    target = target.to(dev)
    y_hat, y_hat_levels = model(img)
    target = get_pyramid(target)
    loss_levels = []
    loss_weights = config.get('loss_weights', {})
    
    for y_hat_el, y in zip(y_hat_levels, target):
        loss_levels.append(compute_weighted_loss(y_hat_el, y, loss_function, loss_weights))
    # Overall Loss
    loss_final = compute_weighted_loss(y_hat, target[0], loss_function, loss_weights)
    # Pyramid Losses (Deep Supervision)
    loss_deep_super = torch.sum(torch.stack(loss_levels))
    loss = loss_final + loss_deep_super

    target = target[0]

    seg_logit = y_hat[:, 1]
    seg_prob = torch.sigmoid(seg_logit)
    seg_pred = (seg_prob > seg_threshold).float() #torch.argmax(y_hat[:, 1:], dim=1)
    seg_true = target[:, 0]
    seg_acc = (seg_pred == seg_true).float().mean()#(seg_pred == target[:, 1]).float().mean()

    # how many pixels are actually crevasse vs predicted as crevasse
    seg_pos_true = seg_true.mean() # fraction of true crevasse pixels
    seg_pos_pred = seg_pred.mean() # fraction predicted as crevasse

    # recall on crevasse pixels (how many of the true vs predicted as crevasse)
    seg_tp = ((seg_pred == 1) & (seg_true == 1)).float().sum()
    seg_fn = ((seg_pred == 0) & (seg_true == 1)).float().sum()
    seg_recall = seg_tp / (seg_tp + seg_fn + 1e-6)
    seg_fp = ((seg_pred == 1) & (seg_true == 0)).float().sum()
    seg_tn = ((seg_pred == 0) & (seg_true == 0)).float().sum()
    seg_precision = seg_tp / (seg_tp + seg_fp + 1e-6)
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall + 1e-6)

    edge_logit = y_hat[:, 0]
    edge_prob = torch.sigmoid(edge_logit)
    edge_pred = (edge_prob > 0.5).float() #(y_hat[:, 0] > 0).float()
    edge_true = target[:, 1]
    edge_acc = (edge_pred == edge_true).float().mean() #(edge_pred == target[:, 0]).float().mean()

    if metrics is not None:
        metrics.step(
            Loss=loss,
            SegAcc=seg_acc,
            EdgeAcc=edge_acc,
            SegPosTrue=seg_pos_true,
            SegPosPred=seg_pos_pred,
            SegRecall=seg_recall,
            SegPrecision=seg_precision,
            SegF1=seg_f1,
        )

    return dict(
        img=img,
        target=target,
        y_hat=y_hat,
        loss=loss,
        loss_final=loss_final,
        loss_deep_super=loss_deep_super,
        seg_tp=seg_tp,
        seg_fp=seg_fp,
        seg_fn=seg_fn,
        seg_tn=seg_tn,
        seg_prob=seg_prob,
        seg_true=seg_true,
        edge_acc=edge_acc,
    )


def train(dataset):
    global epoch, train_history, writer, train_sampler
    # Training step

    data_loader = DataLoader(dataset,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None), num_workers=config['data_threads'],
        sampler=train_sampler,
        pin_memory=True
    )

    epoch += 1
    model.train(True)
    prog = tqdm(data_loader)
    for i, (img, target) in enumerate(prog): 
        for param in model.parameters():
            param.grad = None
        res = full_forward(model, img, target, metrics)
        res['loss'].backward()
        opt.step()

        if (i+1) % 1000 == 0:
            prog.set_postfix(metrics.peek())

    metrics_vals = metrics.evaluate()
    logstr = f'Epoch {epoch:02d} - Train: ' \
           + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)

    if writer is not None:
        for key, val in metrics_vals.items():
            writer.add_scalar(f"train/{key}", val, epoch)

    train_entry = dict(epoch=epoch, **metrics_vals)
    train_history.append(train_entry)

    # Save model Checkpoint
    torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')


@torch.no_grad()
def val(dataset):
    # Validation step
    data_loader = DataLoader(dataset,
        batch_size=config['batch_size'],
        shuffle=False, num_workers=config['data_threads'],
        pin_memory=True
    )

    model.train(False)
    bins = int(config.get('threshold_bins', 256))
    pos_hist, neg_hist = init_hist(bins)
    loss_sum = 0.0
    edge_acc_sum = 0.0
    batch_count = 0
    idx = 0
    for img, target in tqdm(data_loader):
        B = img.shape[0]
        res = full_forward(model, img, target, metrics=None)
        loss_sum += float(res['loss'].item())
        edge_acc_sum += float(res['edge_acc'].item())
        batch_count += 1
        update_hist(pos_hist, neg_hist, res['seg_prob'], res['seg_true'], bins)

        for i in range(B):
            if idx+i in config['visualization_tiles']:
                showexample(idx+i, img[i], res['target'][i], res['y_hat'][i])
        idx += B

    best_threshold, confusion, precision, recall, f1 = best_threshold_from_hist(pos_hist, neg_hist)
    seg_cm = confusion_matrix_from_counts(confusion)
    total = confusion['tp'] + confusion['fp'] + confusion['fn'] + confusion['tn']
    seg_acc = (confusion['tp'] + confusion['tn']) / (total + 1e-6)
    seg_pos_true = (confusion['tp'] + confusion['fn']) / (total + 1e-6)
    seg_pos_pred = (confusion['tp'] + confusion['fp']) / (total + 1e-6)
    metrics_vals = dict(
        Loss=(loss_sum / max(batch_count, 1)),
        SegAcc=seg_acc,
        EdgeAcc=(edge_acc_sum / max(batch_count, 1)),
        SegPosTrue=seg_pos_true,
        SegPosPred=seg_pos_pred,
        SegRecall=recall,
        SegPrecision=precision,
        SegF1=f1,
        SegThreshold=best_threshold,
    )
    return metrics_vals, seg_cm, best_threshold


@torch.no_grad()
def test(dataset, seg_threshold=0.5):
    global test_history, writer
    data_loader = DataLoader(dataset,
        batch_size=config['batch_size'],
        shuffle=False, num_workers=config['data_threads'],
        pin_memory=True
    )

    model.train(False)
    seg_confusion = init_confusion()
    for img, target in tqdm(data_loader):
        res = full_forward(model, img, target, metrics, seg_threshold=seg_threshold)
        update_confusion(seg_confusion, res['seg_tp'], res['seg_fp'], res['seg_fn'], res['seg_tn'])

    metrics_vals = metrics.evaluate()
    tp = seg_confusion['tp']
    fp = seg_confusion['fp']
    fn = seg_confusion['fn']
    tn = seg_confusion['tn']
    seg_cm = confusion_matrix_from_counts(seg_confusion)
    extra_metrics = compute_binary_classification_metrics(tp, fp, fn, tn)
    metrics_vals['SegF1Macro'] = extra_metrics['macro_f1']
    metrics_vals['SegF1Weighted'] = extra_metrics['weighted_f1']
    metrics_vals['SegPrecisionMacro'] = extra_metrics['macro_precision']
    metrics_vals['SegPrecisionWeighted'] = extra_metrics['weighted_precision']
    metrics_vals['SegRecallMacro'] = extra_metrics['macro_recall']
    metrics_vals['SegRecallWeighted'] = extra_metrics['weighted_recall']
    metrics_vals['SegSpecificity'] = extra_metrics['specificity']
    metrics_vals['SegBalancedAcc'] = extra_metrics['balanced_accuracy']
    metrics_vals['SegMacroAcc'] = extra_metrics['macro_accuracy']
    metrics_vals['SegCohenKappa'] = extra_metrics['cohen_kappa']

    logstr = 'Test: ' + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    print(logstr)
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)
        seg_cm_str = f"SegConfusionMatrix: [[{seg_cm[0,0]}, {seg_cm[0,1]}], [{seg_cm[1,0]}, {seg_cm[1,1]}]]"
        print(seg_cm_str, file=f)
    print(seg_cm_str)
    save_confusion_matrix(seg_cm, f'confusion_test_{epoch:02d}.png', 'Test Segmentation Confusion Matrix')

    if writer is not None:
        for key, val in metrics_vals.items():
            writer.add_scalar(f"test/{key}", val, epoch)
        writer.add_text("test/SegConfusionMatrix", seg_cm_str, epoch)

    test_entry = dict(epoch=epoch, **metrics_vals)
    test_history.append(test_entry)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
    config_file = Path(cli_args['--config'])
    config = yaml.load(config_file.open(), Loader=yaml.SafeLoader)
    brightness_label = format_brightness_label(config.get('augmentation', {}))

    modelclass = get_model(config['model'])
    model = modelclass(**config['model_args'])

    if cli_args['--resume']:
        config['resume'] = cli_args['--resume']

    if 'resume' in config and config['resume']:
        checkpoint = Path(config['resume'])
        if not checkpoint.exists():
            raise ValueError(f"There is no Checkpoint at {config['resume']} to resume from!")
        if checkpoint.is_dir():
            # Load last checkpoint in run dir
            ckpt_nums = [int(ckpt.stem) for ckpt in checkpoint.glob('checkpoints/*.pt')]
            last_ckpt = max(ckpt_nums)
            config['resume'] = checkpoint / 'checkpoints' / f'{last_ckpt:02d}.pt'
        print(f"Resuming training from checkpoint {config['resume']}")
        model.load_state_dict(torch.load(config['resume']))

    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")
    print(f'Training on {dev} device')
    model = model.to(dev)

    epoch = 0
    metrics = Metrics()

    SOBEL = nn.Conv2d(1, 2, 3, padding=1, padding_mode='replicate', bias=False)
    SOBEL.weight.requires_grad = False
    SOBEL.weight.set_(torch.Tensor([[
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]],
       [[-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]]).reshape(2, 1, 3, 3))
    SOBEL = SOBEL.to(dev)

    lr = config['learning_rate']
    opt = Adam(model.parameters(), lr)

    if cli_args['--summary']:
        from torchsummary import summary
        summary(model, [(3, 512, 512)])
        sys.exit(0)

    stack_height = 1 if 'stack_height' not in config['model_args'] else \
            config['model_args']['stack_height']

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False, parents=True)

    shutil.copy(config_file, log_dir / 'config.yml')

    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    writer = SummaryWriter(str(log_dir / "tensorboard"))

    trnval = get_dataset('train')
    split_seed = config.get('split_seed', 42)
    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(len(trnval)).tolist()

    test_count = int(len(indices) * 0.2)
    trainval_count = len(indices) - test_count
    val_count = int(trainval_count * 0.1)
    if trainval_count > 1 and val_count == 0:
        val_count = 1

    test_indices = indices[:test_count]
    val_indices = indices[test_count:test_count + val_count]
    trn_indices = indices[test_count + val_count:]

    print(
        f"Split sizes: train={len(trn_indices)} val={len(val_indices)} "
        f"test={len(test_indices)} (seed={split_seed})"
    )
    write_split_files(
        trnval,
        {'train': trn_indices, 'val': val_indices, 'test': test_indices},
        log_dir,
    )

    aug_cfg = config.get('augmentation', {})
    brightness = aug_cfg.get('brightness', 0.0)
    brightness_range = aug_cfg.get('brightness_range')
    trn_dataset = Augment(
        Subset(trnval, trn_indices),
        brightness=brightness,
        brightness_range=brightness_range,
    )
    val_dataset = Subset(trnval, val_indices)
    test_dataset = Subset(trnval, test_indices)

    loss_function = get_loss(config['loss_args'])
    if type(loss_function) is torch.nn.Module:
        loss_function = loss_function.to(dev)

    if config.get('balanced_sampling', False):
        train_sampler = build_balanced_sampler(trnval, trn_indices, len(trn_dataset), config)
        if train_sampler is None:
            print("Balanced sampling requested, but sampler could not be built. Falling back to shuffle.")
            train_sampler = None

    early_stopping_enabled = config.get('early_stopping', False)
    patience = config.get('patience', 0)
    # Allow small fluctuations on the validation loss before counting toward patience
    val_tolerance = config.get('early_stopping_tolerance', 0.0)
    plot_title = "Training metrics"
    best_val_loss = float('inf')
    best_seg_threshold = 0.5
    best_seg_f1 = -1.0
    best_seg_threshold_f1 = 0.5
    patience_counter = 0
    best_checkpoint = None
    best_f1_checkpoint = None

    for _ in range(config['epochs']):
        train(trn_dataset)
        metrics_vals, seg_cm, seg_threshold = val(val_dataset)
        logstr = f'Epoch {epoch:02d} - Val: ' \
               + ', '.join(f'{key}: {value:.3f}' for key, value in metrics_vals.items() if key != 'SegThreshold')
        logstr += f", SegThreshold: {metrics_vals['SegThreshold']:.3f}"
        print(logstr)
        with (log_dir / 'metrics.txt').open('a+') as f:
            print(logstr, file=f)

        if writer is not None:
            for key, value in metrics_vals.items():
                writer.add_scalar(f"val/{key}", value, epoch)

        val_entry = dict(epoch=epoch, **metrics_vals)
        val_history.append(val_entry)

        current_val_loss = metrics_vals.get('Loss')
        current_seg_f1 = metrics_vals.get('SegF1')
        if current_val_loss is not None and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_seg_threshold = seg_threshold
            seg_cm_str = f"SegConfusionMatrix: [[{seg_cm[0,0]}, {seg_cm[0,1]}], [{seg_cm[1,0]}, {seg_cm[1,1]}]]"
            with (log_dir / 'metrics.txt').open('a+') as f:
                print(seg_cm_str, file=f)
            if writer is not None:
                writer.add_text("val/SegConfusionMatrix", seg_cm_str, epoch)
            save_confusion_matrix(seg_cm, f'confusion_val_best_{epoch:02d}.png', 'Best Val Segmentation Confusion Matrix')
        if current_seg_f1 is not None and current_seg_f1 > best_seg_f1:
            best_seg_f1 = current_seg_f1
            best_seg_threshold_f1 = seg_threshold
            best_f1_checkpoint = checkpoints / 'best_f1.pt'
            torch.save(model.state_dict(), best_f1_checkpoint)
        if early_stopping_enabled and patience > 0 and val_history:
            if current_val_loss is None:
                continue
            # Reset patience if loss improves or stays within tolerance margin
            if current_val_loss <= best_val_loss * (1 + val_tolerance):
                patience_counter = 0
                best_checkpoint = checkpoints / 'best.pt'
                torch.save(model.state_dict(), best_checkpoint)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch:02d} (val loss {current_val_loss:.4f})")
                    break

    use_best_f1 = best_f1_checkpoint is not None and best_f1_checkpoint.exists()
    if use_best_f1:
        model.load_state_dict(torch.load(best_f1_checkpoint))
        best_seg_threshold = best_seg_threshold_f1

    if writer is not None:
        writer.close()

    if (not use_best_f1) and early_stopping_enabled and best_checkpoint is not None and best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint))
    if config.get('run_test', True) and len(test_dataset) > 0:
        test(test_dataset, seg_threshold=best_seg_threshold)

    def plot_metric(name, filename):
        if not train_history or not val_history:
            return
        epochs = [h['epoch'] for h in train_history]
        if name not in train_history[0]:
            return

        plt.figure()
        plt.plot(epochs, [h[name] for h in train_history], label='Train', color='forestgreen')
        plt.plot(epochs, [h[name] for h in val_history], label='Val', color='hotpink')
        if test_history and name in test_history[0]:
            test_epoch = test_history[-1]['epoch']
            test_val = test_history[-1][name]
            plt.scatter([test_epoch], [test_val], label='Test', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if name != 'Loss':
            plt.ylim(0, 1)
        plt.title(f"{plot_title}\n{brightness_label}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(log_dir / filename, bbox_inches='tight')
        plt.close()
    
    plot_metric('Loss', 'loss.png')
    plot_metric('SegAcc', 'seg_accuracy.png')
    plot_metric('EdgeAcc', 'edge_accuracy.png')
    plot_metric('SegRecall', 'seg_recall.png')
    plot_metric('SegPrecision', 'seg_precision.png')
    plot_metric('SegF1', 'seg_f1.png')
    plot_metric('SegPosTrue', 'seg_pos_true.png')
    plot_metric('SegPosPred', 'seg_pos_pred.png')
