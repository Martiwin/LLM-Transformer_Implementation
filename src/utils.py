import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch


def set_seed(seed):
    """设置全局随机种子以保证实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_losses(train_losses, val_losses, save_path="./results/plots/loss_curve.png"):
    """绘制训练集和验证集的 Loss 曲线"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(0, len(train_losses))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'b-', label="Train Loss")
    plt.plot(epochs, val_losses, 'r-', label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss 曲线已保存到: {save_path}")


def plot_accuracy(train_accs, val_accs, save_path="./results/plots/accuracy_curve.png"):
    """绘制训练集和验证集的 Accuracy 曲线"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(0, len(train_accs))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, 'b-', label="Train Accuracy")
    plt.plot(epochs, val_accs, 'r-', label="Val Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy 曲线已保存到: {save_path}")


def calculate_accuracy(pred, target, pad_idx):
    """计算序列准确率(忽略 pad)"""
    pred_tokens = pred.argmax(dim=-1)
    mask = target != pad_idx
    correct = (pred_tokens == target) & mask
    
    # 避免除以零 (如果 mask 全为 False)
    mask_sum = mask.sum().float()
    if mask_sum == 0:
        return 0.0
        
    acc = correct.sum().float() / mask_sum
    return acc.item()
