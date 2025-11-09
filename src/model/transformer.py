import math
from sympy import true
import torch
import torch.nn as nn
from .modules import MultiHeadAttention, PositionWiseFeedForward, SublayerConnection
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask):
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, mask))
        x = self.sublayer2(x, self.feed_forward)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory是编码器的输出
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.cross_attention(x, memory, memory, src_mask))
        x = self.sublayer3(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout,
                 max_seq_length, use_positional_encoding=true):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.use_positional_encoding = use_positional_encoding

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)

        # 是否使用位置编码（消融实验）
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, 
                 max_seq_length, use_positional_encoding=true):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.use_positional_encoding = use_positional_encoding

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)

        # 是否使用位置编码（消融实验）
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, dropout=0.1, max_seq_length=5000, use_positional_encoding=true):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff,
                               dropout, max_seq_length, use_positional_encoding)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, 
                               dropout, max_seq_length, use_positional_encoding)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)
        return self.output_layer(output)

    def generate_mask(self, src, tgt):
        device = src.device  # 获取 src 所在设备
        # 生成源序列掩码
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # 生成目标序列掩码（防止解码器看到未来信息）
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask
