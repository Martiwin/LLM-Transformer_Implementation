#!/bin/bash
# Transformer (EN-DE) 训练与推理脚本
# 1. (首次) 给予执行权限: chmod +x scripts/run.sh
# 2. 运行脚本: ./scripts/run.sh


# ================================参数设置============================
# 设置本次运行的实验名称
# ！！！一定不要和现有结果文件夹名称重复！！！一定不要和现有结果文件夹名称重复！！！一定不要和现有结果文件夹名称重复！！！
# 训练结果将保存在 ./results/RUN_NAME
# 推理将自动从该目录加载模型
RUN_NAME="experiment"

# 训练时使用的配置文件
CONFIG_FILE="./configs/cfg.yaml"

# 训练和推理设备
DEVICE="cuda:0"

# 覆盖 YAML 中的 EPOCHS 和 BATCH_SIZE
EPOCHS=20
BATCH_SIZE=32

# 推理时使用的源文件 (英语) 和目标文件 (德语参考答案)
SRC_FILE="./data/processed/dev-en.txt"
TGT_FILE="./data/processed/dev-de.txt"

# ================================训练============================
echo "========================================="
echo "Starting Training..."
echo "Run Name: $RUN_NAME"
echo "Config: $CONFIG_FILE"
echo "========================================="

TRAIN_CMD="python src/train.py --cfg $CONFIG_FILE --name $RUN_NAME --device $DEVICE"

if [ ! -z "$EPOCHS" ]; then
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
fi

if [ ! -z "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
fi

echo "Running command: $TRAIN_CMD"
eval $TRAIN_CMD

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo "Training complete."


# ================================推理============================
# 训练结果的目录
RUN_DIR="./results/$RUN_NAME"

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Training result directory $RUN_DIR does not exist."
    echo "Inference cannot proceed."
    exit 1
fi

echo "========================================="
echo "Starting Inference (Translation)..."
echo "Model Directory: $RUN_DIR"
echo "Source File: $SRC_FILE"
echo "Target File: $TGT_FILE"
echo "========================================="

# 执行推理
python src/translate.py \
    --run_dir $RUN_DIR \
    --src_file $SRC_FILE \
    --tgt_file $TGT_FILE \
    --device $DEVICE

if [ $? -ne 0 ]; then
    echo "Inference failed!"
    exit 1
fi

echo "Inference complete."
echo "Translation results saved in $RUN_DIR/translations.txt"
echo "All done."
