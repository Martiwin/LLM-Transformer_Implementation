# Transformer 机器翻译 (EN-DE)

这是一个使用 PyTorch 实现的 Transformer 模型，用于 IWSLT 2017 数据集上的英语德语机器翻译任务。

### 1. 环境设置

**a. 创建虚拟环境**
```bash
conda create -n transformer python=3.10
conda activate transformer
```

**b. 安装依赖**
```bash
pip install -r requirements.txt
# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. 数据预处理

下载 IWSLT 2017 (EN-DE) 数据集并将其解压到 `./data/en-de/` 目录。
然后，运行预处理脚本来提取和清理数据：

```bash
python src/data/preprocess.py
```
这将在 `./data/processed/` 中生成训练/验证 `.txt` 文件，并在 `./data/vocab/` 中创建词汇表。

### 3. 如何运行

#### A：使用 `run.sh` 脚本

该方法会自动执行训练和随后的推理。

```bash
# (首次) 给予脚本执行权限
chmod +x scripts/run.sh

# 运行完整的训练和推理流程
./scripts/run.sh
```
你可以在 `scripts/run.sh` 文件的顶部修改 `RUN_NAME` 来管理你的实验。

#### B：手动执行

**a. 训练模型**

```bash
python src/train.py --cfg configs/cfg.yaml --name "my_experiment" --device cuda:0
```
* `--cfg`: 指定要使用的配置文件，模型训练参数可以在其中修改。
* `--name`: 本次运行的名称，结果将保存在 `results/my_experiment`（若文件夹名重复，会自动编号）。

**b. 执行翻译**

训练完成后，使用 `translate.py` 脚本来生成翻译结果。

```bash
python src/translate.py --run_dir ./results/my_experiment --src_file ./data/processed/dev-en.txt --tgt_file ./data/processed/dev-de.txt --device cuda:0
```
* `--run_dir`: 指向包含 `args.yaml` 和 `best.pth` 的训练结果目录。
* 翻译结果将以三行（输入、翻译、答案）一组的形式保存在 `results/my_experiment/translations.txt` 中。

---

## 重现实验

重现实验时默认随机种子均为42。train3、train4、train6、train7分别为以下四个实验的结果文件夹名。

**a. 训练原始模型**

**重现实验的精确命令行：**

```bash
python src/train.py  --cfg results/train3/args.yaml --name "my_experiment" --device cuda:0
```

**b. 第一次优化训练**

**重现实验的精确命令行：**

```bash
python src/train.py  --cfg results/train4/args.yaml --name "my_experiment" --device cuda:0
```

**c. 第二次优化训练**

**重现实验的精确命令行：**

```bash
python src/train.py  --cfg results/train6/args.yaml --name "my_experiment" --device cuda:0
```

**d. 消融实验训练**

**重现实验的精确命令行：**

```bash
python src/train.py  --cfg results/train7/args.yaml --name "my_experiment" --device cuda:0
```

## 文件结构

```
.
├── configs/
│   └── cfg.yaml               # 默认配置文件
├── data/
│   ├── en-de/                 # (原始 IWSLT 2017 数据)
│   ├── processed/             # (预处理后的 .txt 数据)
│   └── vocab/                 # (生成的词汇表)
├── results/
│   └── (实验结果, 例如 train6)/
│       ├── args.yaml          # 该次运行的配置快照
│       ├── plots/             # 损失和准确率曲线
│       ├── weights/           # best.pth
│       └── translations.txt   # (推理后生成)
├── scripts/
│   └── run.sh                 # 自动化训练和推理脚本
├── src/
│   ├── model/
│   │   ├── transformer.py         # Transformer, Encoder, Decoder
│   │   ├── modules.py             # MultiHeadAttention, FFN
│   │   └── positional_encoding.py # 位置编码
│   ├── data/
│   │   ├── dataloader.py          # Dataset
│   │   ├── preprocess.py          # 数据预处理脚本
│   │   └── vocabulary.py          # (生成的词汇表)
│   ├── config.py                  # 配置解析
│   ├── train.py                   # 训练和验证的主脚本
│   ├── translate.py               # 推理脚本 (生成翻译)
│   └── utils.py                   # 辅助函数 (绘图, set_seed)
├── README.md
└── requirements.txt
```
