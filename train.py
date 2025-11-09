import torch
from torch.utils.data import DataLoader
from data.dataloader import TranslationDataset, read_lines, load_vocab
from model.transformer import Transformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from pathlib import Path
from tqdm import tqdm
from utils import plot_accuracy, plot_losses, calculate_accuracy, set_seed
from config import parse_args


def train(cfg):
    """训练函数，加载配置"""
    DEVICE = torch.device(cfg['DEVICE'])
    PAD_IDX = 0
    
    # ----------------- 定义保存路径 ----------------- 
    exp_dir = Path(cfg['EXP_DIR'])
    best_model_path = exp_dir / 'weights' / 'best.pth'
    loss_plot_path = exp_dir / 'plots' / 'loss_curve.png'
    acc_plot_path = exp_dir / 'plots' / 'accuracy_curve.png'

    # ----------------- 加载数据 ----------------- 
    print("Loading data...")
    train_en = read_lines(cfg['train_en_path'])
    train_de = read_lines(cfg['train_de_path'])
    val_en = read_lines(cfg['val_en_path'])
    val_de = read_lines(cfg['val_de_path'])

    print(f"训练集句子数: {len(train_en)}, {len(train_de)}")
    print(f"验证集句子数: {len(val_en)}, {len(val_de)}")
    assert len(train_en) == len(train_de), "训练集句子数不匹配"
    assert len(val_en) == len(val_de), "验证集句子数不匹配"

    # ----------------- 加载词表 ----------------- 
    en_vocab = load_vocab(cfg['en_vocab_path'])
    de_vocab = load_vocab(cfg['de_vocab_path'])
    # 验证 <pad> 索引
    assert de_vocab['<pad>'] == PAD_IDX, f"<pad> index is not {PAD_IDX}"

    # ----------------- 构造DataLoader -----------------
    print("Creating datasets and dataloaders...")
    train_dataset = TranslationDataset(train_en, train_de, en_vocab, de_vocab, max_len=cfg['MAX_LEN'])
    val_dataset = TranslationDataset(val_en, val_de, en_vocab, de_vocab, max_len=cfg['MAX_LEN'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False)

    # ----------------- 初始化模型 -----------------
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(de_vocab),
        d_model=cfg['D_MODEL'],
        num_layers=cfg['NUM_LAYERS'],
        num_heads=cfg['NUM_HEADS'],
        d_ff=cfg['D_FF'],
        dropout=cfg['DROPOUT'],
        max_seq_length=cfg['MAX_LEN'],
        use_positional_encoding=cfg.get('USE_POSITIONAL_ENCODING', True)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LR'], weight_decay=cfg.get('L2_WEIGHT', 0.0))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=cfg.get('LABEL_SMOOTHING', 0.0))
    if_scheduler = cfg.get('USE_SCHEDULER', False)  # 是否使用学习率调度器
    if if_scheduler:
        scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=cfg.get('SCHEDULER_FACTOR', 0.5), 
        patience=cfg.get('SCHEDULER_PATIENCE', 3),
        verbose=True
    )

    best_val_loss = float('inf')

    # ----------------- 训练循环 -----------------
    print(f"Starting training for {cfg['EPOCHS']} epochs...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, cfg['EPOCHS'] + 1):
        model.train()
        total_loss, total_acc = 0, 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['EPOCHS']} [Train]", unit="batch")
        for src, tgt in train_iter:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            src_mask, tgt_mask = model.generate_mask(src, tgt[:, :-1])  # 创建掩码
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)  # 前向传播
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            acc = calculate_accuracy(output, tgt[:, 1:], PAD_IDX)
            total_acc += acc
            train_iter.set_postfix(loss=loss.item(), acc=acc)
            if not train_losses:  # 记录初始损失
                train_losses.append(loss.item())
            if not train_accs:
                train_accs.append(acc)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        model.eval()
        val_loss, val_acc = 0, 0
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['EPOCHS']} [Val]", unit="batch")
        with torch.no_grad():
            for src, tgt in val_iter:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                src_mask, tgt_mask = model.generate_mask(src, tgt[:, :-1])
                output = model(src, tgt[:, :-1], src_mask, tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                val_loss += loss.item()
                acc = calculate_accuracy(output, tgt[:, 1:], PAD_IDX)
                val_acc += acc
                val_iter.set_postfix(loss=loss.item(), acc=acc)
                if not val_losses:  # 记录初始损失
                    val_losses.append(loss.item())
                if not val_accs:
                    val_accs.append(acc)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        print(f"Epoch [{epoch}/{cfg['EPOCHS']}] "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        if if_scheduler:  # 使用学习率调度器
            scheduler.step(avg_val_loss)

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print("  -> Save the best model")

    # --------- 绘制训练并保存曲线 ---------
    print("Training complete. Saving plots...")
    plot_losses(train_losses, val_losses, save_path=loss_plot_path)
    plot_accuracy(train_accs, val_accs, save_path=acc_plot_path)
    print("Done.")


if __name__ == "__main__":
    # --cfg, --name, --device, --batch_size 32, --epochs 20
    config = parse_args()
    set_seed(config['SEED'])
    train(config)
