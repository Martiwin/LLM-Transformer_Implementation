import re
from collections import Counter


class SimpleTokenizer:
    def __init__(self, vocab=None, lower=True, vocab_size=None):
        self.lower = lower
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}

        if vocab is not None:
            if isinstance(vocab, list):
                self.word2idx = {w: i for i, w in enumerate(vocab)}
                self.idx2word = {i: w for w, i in self.word2idx.items()}


    def tokenize(self, text):
        """简单分词器：保留单词与标点"""
        if self.lower:
            text = text.lower()
        return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)


    def build_vocab(self, texts, min_freq=1):
        """根据训练集构建词汇表"""
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        vocab = [w for w, f in counter.items() if f >= min_freq]
        vocab = sorted(vocab, key=lambda w: counter[w], reverse=True)
        if self.vocab_size:
            vocab = vocab[:self.vocab_size]

        special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        vocab = special_tokens + vocab

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        print(f"构建完成，词表大小: {len(vocab)}")


    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(tok, self.word2idx['<unk>']) for tok in tokens]
        if add_special_tokens:
            ids = [self.word2idx['<bos>']] + ids + [self.word2idx['<eos>']]
        return ids


    def decode(self, ids, remove_special_tokens=True):
        tokens = [self.idx2word.get(i, '<unk>') for i in ids]
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in ('<pad>', '<unk>', '<bos>', '<eos>')]
        return ' '.join(tokens)


    def save_vocab(self, file_path):
        """保存词表为 .txt，格式：<token>\t<index>"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for word, idx in self.word2idx.items():
                f.write(f"{word}\t{idx}\n")
        print(f"词表已保存到 {file_path}")


    def load_vocab(self, file_path):
        """从 .txt 文件加载词表"""
        self.word2idx = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, idx = parts
                    self.word2idx[word] = int(idx)
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        print(f"词表已加载，共 {len(self.word2idx)} 个词")

    
    def get_vocab_size(self):
        return len(self.word2idx)
