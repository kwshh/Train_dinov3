import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import timm
import random
import numpy as np

from timm.data import resolve_model_data_config, create_transform
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler


def seed_everything(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 可复现，但可能稍慢
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        # 某些算子严格确定性（可能会报不支持的算子）
        torch.use_deterministic_algorithms(False)
    else:
        # 更快但不完全可复现
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    # 确保每个 dataloader worker 的随机可控性
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed = 42               # 固定seed
seed_everything(seed)

g = torch.Generator()
g.manual_seed(seed)


def get_lrs(optimizer):
    """ 返回每个 param_group 的 lr """
    return [pg["lr"] for pg in optimizer.param_groups]


def create_dataloaders(data_root, model_name, img_size, batch_size, num_workers=4):
    data_root = Path(data_root)

    # 使用 timm 的 data_config 来保证与模型预训练时的归一化等一致
    # 先临时建一个模型，只为了拿 data_config
    tmp_model = timm.create_model(
        model_name,  # 你也可以换成别的 DINOv3 模型
        pretrained=False,
        num_classes=0,
    )
    data_config = resolve_model_data_config(tmp_model)
    # print(data_config)
    # exit(0)
    input_size = (3, img_size, img_size)
    # 训练增强
    train_transform = create_transform(
        input_size=input_size,
        mean=data_config["mean"],
        std=data_config["std"],
        interpolation=data_config["interpolation"],
        crop_pct=data_config.get("crop_pct", 1.0),      # 0.875，相当于resize到293再裁剪（先放大再扣中间），避免边缘噪声，对分类略有好处。
        is_training=True,
        # 你可以在这里额外加一些更强的增强，比如：
        # auto_augment='rand-m9-mstd0.5-inc1',  # 需要 timm>=0.9
        # re_prob=0.25,  # Random Erasing 概率
    )
    # 验证/测试增强（只做 resize + center crop + normalize）
    val_transform = create_transform(
        input_size=input_size,
        mean=data_config["mean"],
        std=data_config["std"],
        interpolation=data_config["interpolation"],
        crop_pct=data_config.get("crop_pct", 1.0),
        is_training=False,
    )

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_dataset.classes), train_dataset.classes


def build_model(model_name: str, num_classes: int, device="cuda"):
    """
    创建 DINOv3 模型，并把分类头改成 num_classes。
    这里用 vit_small_patch16_dinov3.lvd1689m 作为例子。
    """
    model = timm.create_model(
        model_name,
        pretrained=True,          # 加载预训练权重
        num_classes=num_classes,  # 自动替换最后的分类头
    )

    return model


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_BN(model: nn.Module, verbose: bool = True):
    """
    timm ViT 典型结构:
      - model.stem
      - model.blocks (ModuleList，长度一般为4: blocks[0..3])
      - model.head (分类头)
    这里解冻 BN 层，目的是适配分布，不破坏语义结构
    """
    freeze_all(model)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "bn" in name.lower():
            param.requires_grad = True

    if verbose:
        trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
        total_trainable = sum(x[1] for x in trainable)
        print(f"Trainable params: {total_trainable/1e6:.2f}M")
        # 打印确认可训练参数无误
        for n, p in model.named_parameters():
            print(f"{n} trainabel:", p.requires_grad)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # print(images.shape)
        # exit(0)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # 混合精度
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    # ================== 自己按需求改的配置 ================== #
    data_root = r"D:\dataset\xinyongtao_shop_Train"    # data/train, data/val
    # model_name = "vit_small_patch16_dinov3.lvd1689m"
    model_name = "convnext_tiny.dinov3_lvd1689m"
    save_name = f"finetune_BN_{model_name}_best_model.pth"
    batch_size = 32         # 如果显存不够就改小
    num_workers = 4
    num_epochs = 10
    lr       = 5e-4         # 1e-4 ~ 3e-4
    weight_decay = 0.05     # ViT 通常比 CNN 更需要 wd
    warmup_epochs = 0       # 可以 0 或 1~2（不是必须）
    img_size = 224          # DINOv3 常见训练尺寸（跟模型 data_config 保持一致即可）
    # ======================================================= #
    
    output_dir = f"finetune_BN_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据
    train_loader, val_loader, num_classes, class_names = create_dataloaders(
        data_root=data_root,
        model_name=model_name,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print("Num classes:", num_classes)

    # 模型
    model = build_model(model_name, num_classes, device)

    # 解冻最后 BN
    model = unfreeze_BN(model, verbose=True)

    # 损失 & 优化器 & 学习率调度
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model,
        opt="adamw",
        lr=lr,
        weight_decay=weight_decay,
    )

    # # 使用 Cosine scheduler（按 epoch 更新）
    # num_steps = len(train_loader) * num_epochs
    # scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=num_steps,
    #     lr_min=lr * 0.01,
    #     warmup_t=len(train_loader) * warmup_epochs,
    #     warmup_lr_init=lr * 0.1,
    #     t_in_epochs=False,
    # )

    # 混合精度（如果是 CPU 就不用）
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # 手动按 step 更新 scheduler
        for _ in range(len(train_loader)):
            # scheduler.step(global_step)
            global_step += 1

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start

        block_lr, head_lr = get_lrs(optimizer)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"LR(block): {block_lr:.2e} LR(head): {head_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(output_dir, save_name)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "class_names": class_names,
                },
                ckpt_path,
            )
            print(f"  >>> New best model saved to {ckpt_path}")

    print("Training finished. Best Val Acc:", best_val_acc)


if __name__ == "__main__":
    main()
