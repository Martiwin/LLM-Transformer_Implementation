import torch
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from model.transformer import Transformer
from data.dataloader import read_lines, load_vocab
from data.vocabulary import SimpleTokenizer


# Mask 创建函数
def create_src_mask(src, pad_idx, device):
    """
    创建源序列的 padding mask
    形状: (batch_size, 1, 1, src_len)
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2).to(device)

def create_tgt_mask(tgt, pad_idx, device):
    """
    创建目标序列的 padding mask 和 look-ahead mask
    形状: (batch_size, 1, tgt_len, tgt_len)
    """
    # 创建 padding mask (B, 1, 1, L)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    
    # 创建 look-ahead mask (1, 1, L, L)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()
    
    # 合并
    tgt_mask = tgt_pad_mask & nopeak_mask
    return tgt_mask


# 模型加载
def load_model_from_config(run_dir, device):
    """从 results 文件夹加载配置和模型权重"""
    run_path = Path(run_dir)
    config_path = run_path / 'args.yaml'
    weights_path = run_path / 'weights' / 'best.pth'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 加载词表
    print("Loading vocabularies...")
    en_vocab = load_vocab(cfg['en_vocab_path'])
    de_vocab = load_vocab(cfg['de_vocab_path'])
    
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(de_vocab),
        d_model=cfg['D_MODEL'],
        num_layers=cfg['NUM_LAYERS'],
        num_heads=cfg['NUM_HEADS'],
        d_ff=cfg['D_FF'],
        dropout=cfg.get('DROPOUT', 0.1), # Dropout 在 eval() 模式下无效
        max_seq_length=cfg['MAX_LEN'],
        use_positional_encoding=cfg.get('USE_POSITIONAL_ENCODING', True)
    ).to(device)
    
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # 设置为评估模式
    
    return model, en_vocab, de_vocab, cfg


# 翻译函数
def translate_sentence(sentence, model, en_vocab, de_vocab, idx2word_de, cfg, device):
    """使用贪婪解码翻译单个句子"""
    model.eval()
    
    PAD_IDX = en_vocab.get('<pad>', 0)
    MAX_LEN = cfg['MAX_LEN']

    # 预处理源句子
    tokens = ['<bos>'] + SimpleTokenizer().tokenize(sentence) + ['<eos>']
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
        
    src_ids = [en_vocab.get(tok, en_vocab['<unk>']) for tok in tokens]
    src = torch.tensor([src_ids], device=device) # 形状: (1, src_len)
    
    # 创建源掩码
    src_mask = create_src_mask(src, PAD_IDX, device)
    
    # 编码器
    with torch.no_grad():
        memory = model.encoder(src, src_mask) # 形状: (1, src_len, d_model)
    
    # 解码器 (自回归)
    # 从<bos>开始
    tgt_input = torch.tensor([[de_vocab['<bos>']]], device=device) # 形状: (1, 1)
    generated_ids = []
    
    for _ in range(MAX_LEN - 1): # -1 因为<bos>已经有了
        with torch.no_grad():
            tgt_mask = create_tgt_mask(tgt_input, PAD_IDX, device)  # 创建目标掩码
            out = model.decoder(tgt_input, memory, src_mask, tgt_mask) # 解码器前向传播，(1, curr_len, d_model)
            logits = model.output_layer(out[:, -1, :]) # 获取最后一个词的 logits，(1, vocab_size)
            pred_token_id = logits.argmax(1).item()  # 贪婪搜索：选择概率最高的词
            if pred_token_id == de_vocab['<eos>']:
                break
                
            generated_ids.append(pred_token_id)

            # 将预测的词添加到下一次的输入中
            new_token = torch.tensor([[pred_token_id]], device=device)
            tgt_input = torch.cat([tgt_input, new_token], dim=1) # (1, curr_len + 1)

    # 将ID转换回单词
    translation = ' '.join([idx2word_de.get(idx, '<unk>') for idx in generated_ids])
    return translation


# 主函数
def main():
    parser = argparse.ArgumentParser(description="Transformer 翻译脚本")
    parser.add_argument('--run_dir', type=str, required=True,
                        help='包含 args.yaml 和 weights/best.pth 的实验结果目录 (例如 ./results/train6)')
    parser.add_argument('--src_file', type=str, required=True,
                        help='要翻译的源语言文件 (英语, .txt)')
    parser.add_argument('--tgt_file', type=str, required=True,
                        help='参考的目标语言文件 (德语答案, .txt)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='翻译结果的保存路径. (默认: [run_dir]/translations.txt)')
    parser.add_argument('--device', type=str, default=None,
                        help='设备')
    
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.output_file is None:
        output_path = Path(args.run_dir) / "translations.txt"
    else:
        output_path = Path(args.output_file)
    print(f"Translations will be saved to: {output_path}")

    # 加载模型和词表
    model, en_vocab, de_vocab, cfg = load_model_from_config(args.run_dir, device)
    
    # 创建反向词表
    idx2word_de = {i: w for w, i in de_vocab.items()}

    print(f"Reading source file: {args.src_file}")
    src_sentences = read_lines(args.src_file)
    print(f"Reading target file: {args.tgt_file}")
    tgt_sentences = read_lines(args.tgt_file)
    
    if len(src_sentences) != len(tgt_sentences):
        print(f"Warning: 源文件 ({len(src_sentences)}) 和目标文件 ({len(tgt_sentences)}) 句子数量不匹配!")

    print("Starting translation...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(src_sentences)), desc="Translating"):
            src_text = src_sentences[i]
            tgt_text = tgt_sentences[i] if i < len(tgt_sentences) else "[N/A]"
            
            # 执行翻译
            translation = translate_sentence(src_text, model, en_vocab, de_vocab, idx2word_de, cfg, device)
            
            f.write(f"Src (EN):  {src_text}\n")
            f.write(f"Pred (DE): {translation}\n")
            f.write(f"Ref (DE):  {tgt_text}\n")
            f.write("="*30 + "\n") # 分隔符

    print(f"Translation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
