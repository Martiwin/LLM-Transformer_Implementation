import re
import os
from vocabulary import SimpleTokenizer


def process_training(input_file, output_file):
    """处理训练集文件：跳过以" <"开头的行，保留其他行"""

    print(f"处理训练集文件: {os.path.basename(input_file)}")
    
    if not os.path.exists(input_file):
        print(f"错误：文件不存在 {input_file}")
        return []
    
    sentences = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            # 跳过以" <"开头的行（XML标签行）
            if line.startswith(" <"):
                continue

            # 跳过以"<"开头的行（XML标签行）
            if line.startswith("<"):
                continue
                
            # 清理文本：移除多余的空白字符
            cleaned_line = re.sub(r'\s+', ' ', line).strip()
            if cleaned_line:
                sentences.append(cleaned_line)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    
    print(f"保存了 {len(sentences)} 个句子到 {output_file}")
    return sentences


def process_validation(input_file, output_file):
    """处理验证集文件：提取<seg>标签内的句子"""
    print(f"处理验证集文件: {os.path.basename(input_file)}")
    
    if not os.path.exists(input_file):
        print(f"错误：文件不存在 {input_file}")
        return []
    
    sentences = []
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式提取<seg>标签内容
    # 匹配模式：<seg id="数字">  句子内容  </seg>
    pattern = r'<seg id="\d+">\s*(.*?)\s*</seg>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        # 清理文本：移除多余的空白字符
        cleaned_text = re.sub(r'\s+', ' ', match).strip()
        if cleaned_text:
            sentences.append(cleaned_text)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    
    print(f"保存了 {len(sentences)} 个句子到 {output_file}")
    return sentences


def main():
    input_dir = "./data/en-de"
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)
    vocab_dir = "./data/vocab"
    os.makedirs(vocab_dir, exist_ok=True)
    
    print("开始处理数据集...")
    
    print("\n处理训练集：")
    train_en = process_training(
        f"{input_dir}/train.tags.en-de.en",
        f"{output_dir}/train-en.txt"
    )
    
    train_de = process_training(
        f"{input_dir}/train.tags.en-de.de", 
        f"{output_dir}/train-de.txt"
    )
    
    print("\n处理验证集：")
    val_en = process_validation(
        f"{input_dir}/IWSLT17.TED.dev2010.en-de.en.xml",
        f"{output_dir}/dev-en.txt"
    )
    
    val_de = process_validation(
        f"{input_dir}/IWSLT17.TED.dev2010.en-de.de.xml",
        f"{output_dir}/dev-de.txt"
    )
    
    # 构建词汇表
    print("\n构建词汇表：")
    en_tokenizer = SimpleTokenizer()
    de_tokenizer = SimpleTokenizer()

    en_tokenizer.build_vocab(train_en)
    de_tokenizer.build_vocab(train_de)

    en_tokenizer.save_vocab(f"{vocab_dir}/vocab-en.txt")
    de_tokenizer.save_vocab(f"{vocab_dir}/vocab-de.txt")

    print(f"\n处理完成！")
    print(f"训练集: {len(train_en)} 个句子对")
    print(f"验证集: {len(val_en)} 个句子对")
    print(f"英语词汇表: {en_tokenizer.get_vocab_size()} 个词")
    print(f"德语词汇表: {de_tokenizer.get_vocab_size()} 个词")
    print(f"输出目录: {output_dir}")
    print(f"词汇表目录: {vocab_dir}")


if __name__ == "__main__":
    main()
