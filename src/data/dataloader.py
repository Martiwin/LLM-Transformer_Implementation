import torch
from torch.utils.data import Dataset
from data.vocabulary import SimpleTokenizer


class TranslationDataset(Dataset):
    def __init__(self, en_sentences, de_sentences, en_vocab, de_vocab, max_len=64):
        if not en_sentences or not de_sentences:
            raise ValueError("句子列表不能为空")
        
        self.en_sentences = en_sentences
        self.de_sentences = de_sentences
        self.en_vocab = en_vocab
        self.de_vocab = de_vocab
        self.max_len = max_len
        self.en_tokenizer = SimpleTokenizer(lower=True)
        self.de_tokenizer = SimpleTokenizer(lower=True)

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        src = self.sentence_to_ids(self.en_sentences[idx], self.en_tokenizer, self.en_vocab)
        tgt = self.sentence_to_ids(self.de_sentences[idx], self.de_tokenizer, self.de_vocab)
        return src, tgt

    def sentence_to_ids(self, sentence, tokenizer, vocab):
        tokens = ['<bos>'] + tokenizer.tokenize(sentence) + ['<eos>']
        ids = [vocab.get(tok, vocab['<unk>']) for tok in tokens]
        if len(ids) < self.max_len:
            ids += [vocab['<pad>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids)
    

def read_lines(file_path):
    """读取训练集验证集文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def load_vocab(file_path):
    """加载词表"""
    vocab = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            vocab[token] = int(idx)
    return vocab
