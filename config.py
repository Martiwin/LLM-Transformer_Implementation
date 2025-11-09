import argparse
import yaml
import os
import torch
from pathlib import Path


def get_next_run_dir(base_dir='./results', name='train'):
    """获取下一个运行结果目标"""
    base_path = Path(base_dir)
    
    # 如果指定了名称，且该名称还未使用过
    run_path = base_path / name
    if not run_path.exists():
        return run_path

    # 如果指定名称目录已存在，尝试"name1""name2"
    i = 1
    while True:
        run_path = base_path / f"{name}{i}"
        if not run_path.exists():
            return run_path
        i += 1


def parse_args():
    """解析命令行参数并加载 YAML 配置文件"""
    parser = argparse.ArgumentParser(description="Transformer 训练配置")
    
    parser.add_argument('--cfg', type=str, default='./configs/cfg.yaml',
                        help='指向 .yaml 配置文件路径')
    parser.add_argument('--name', type=str, default=None,
                        help='实验结果保存目录，默认命名为 train, train1...')
    
    # 允许命令行覆盖 YAML 中的特定参数
    parser.add_argument('--device', type=str, default=None,
                        help='覆盖设备设置')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖 BATCH_SIZE')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖 EPOCHS')
    parser.add_argument('--seed', type=int, default=42,
                        help='设置全局随机种子')

    args = parser.parse_args()

    # 加载 YAML 配置
    print(f"Loading config from: {args.cfg}")
    if not os.path.exists(args.cfg):
        raise FileNotFoundError(f"Config file not found: {args.cfg}")
        
    with open(args.cfg, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 命令行覆盖 YAML
    config['SEED'] = args.seed
    
    if args.device:
        config['DEVICE'] = args.device
    if args.batch_size:
        config['BATCH_SIZE'] = args.batch_size
    if args.epochs:
        config['EPOCHS'] = args.epochs

    # 处理设备，检查配置的设备是否可用
    if config['DEVICE'] and 'cuda' in config['DEVICE']:
        if not torch.cuda.is_available():
            print(f"Warning: {config['DEVICE']} not available. Falling back to CPU.")
            config['DEVICE'] = 'cpu'
    elif not config['DEVICE']:
        # 如果 YAML 中未指定，则自动选择
        config['DEVICE'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {config['DEVICE']}")

    # 创建实验结果目录
    base_results_dir = config.get('RESULTS_DIR', './results')
    
    if args.name:
        exp_dir = get_next_run_dir(base_results_dir, name=args.name)
    else:
        exp_dir = get_next_run_dir(base_results_dir, name='train')

    config['EXP_DIR'] = str(exp_dir)
    
    # 创建所有必要的子目录
    (exp_dir / 'weights').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'plots').mkdir(parents=True, exist_ok=True)
    print(f"Experiment results will be saved to: {exp_dir}")

    # 保存本次运行的配置
    save_config_path = exp_dir / 'args.yaml'
    with open(save_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False, indent=4)
    print(f"Saved run configuration to: {save_config_path}")
    
    return config
