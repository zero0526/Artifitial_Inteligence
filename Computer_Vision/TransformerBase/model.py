import torch 
import torch.nn as nn
import math 

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim= True)
        return self.alpha*(x-mean)/(std+self.eps) + self.bias
    
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ffn:int, rDropout:float=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(d_model, d_ffn)
        self.Dropout =nn.Dropout(rDropout)
        self.Relu = nn.Relu()
        self.Linear2 = nn.Linear(d_ffn, d_model)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x= self.Linear1(x)
        x = self.Relu(x)
        x= self.Dropout(x)
        x = self.Linear2(x)
        return x
    
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.vacab_size = vocab_size
        self.d_model = d_model
        self.embeeding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x,):
        return self.embeeding(x)* torch.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.droupout = nn.Dropout(dropout)
        self.pe= torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # e^((i-1)/D*-log(L))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1],:]).require_grad_(False)
        return self.droupout(x)
        
        
class ResidualConnection(nn.Module):
    def __init__(self, features:int, dropout:float )->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features = features)
        
    def forward(self,x,subLayer):
        return x +self.dropout(subLayer(x))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.dropout =nn.Dropout(dropout)
        assert self.d_model==h == 0, 'h need to be diviable by d_model '
        self.head_dim = d_model//h
        self.heads = h
        self.W_q = nn.Linear(d_model,self.head_dim,bias =False)
        self.W_k = nn.Linear(d_model,self.head_dim,bias =False)
        self.W_v = nn.Linear(d_model,self.head_dim,bias =False)
        self.W_o = nn.Linear(d_model,self.d_model,bias =False)
        self.dropout = nn.Dropout(dropout)
        
    def attention(query,key, value, mask, dropout:nn.Dropout):
        d_k = key.shape[-1]
        # key,query: batch_size, head, seq_len, feature
        attention_scores = query @ key.transpose(-1,-2)/math.sqrt(d_k)
        if mask is not None:
            # padding to deal with multiple sequence
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = nn.Softmax(attention_scores, dim =-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k,v, mask):
        # q: batch_size, seq_len,d_model
        batch_size, seq_len, _= q.shape
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)
        
        query = query.view(batch_size,seq_len,self.heads,self.head_dim).transpose(1,2)
        key = key.view(batch_size,seq_len,self.heads,self.head_dim).transpose(1,2)
        value = value.view(batch_size,seq_len,self.heads,self.head_dim).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttention.attention(query = query, key=key,value=value,dropout=self.dropout, mask=mask)
        x = x.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attetion_block:MultiHeadAttention,cross_attention_block:MultiHeadAttention, feed_forward_block:FeedForward, dropout:float)->None:
        super().__init__()
        self.self_attetion_block = self_attetion_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_block = nn.ModuleList([ResidualConnection(features=features) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x= self.residual_block[0](x,lambda x: self.self_attetion_block(x, x, x,tgt_mask))
        x= self.residual_block[1](x,lambda x: self.cross_attention_block(q= x, k= encoder_output, v= encoder_output, mask = src_mask))
        x= self.residual_block[2](x,self.feed_forward_block)
        
        return x

class EncoderBlock(nn.ModuleList):
    def __init__(self, features:int, self_attention_block:MultiHeadAttention, feed_forward_block:FeedForward, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_block = nn.ModuleList([ResidualConnection(features) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_block(x, lambda x:self.self_attention_block(x, x, x, src_mask))
        x = self.feed_forward_block(x, lambda x: self.feed_forward_block(x))
        return x
    
class Encoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, featuers:int, layers:nn.ModuleList) ->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features= featuers)
        
    def forward(self, x, encoder_out , src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x= x, encoder_output= encoder_out, src_mask = src_mask, tgt_mask =tgt_mask)
        return self.norm(x)
    
class ProjectrionLayer(nn.Module):
    def __init__(self, d_model, vocab_size)->None:
        self.projector = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.projector(x)
    
class Transformer(nn.Moduler):
    def __init__(self, encoder, decoder, src_embed: InputEmbedding, tgt_embed:InputEmbedding,\
        src_pos:PositionalEncoding, tgt_pos:PositionalEncoding,projector:ProjectrionLayer):
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projector = projector
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output:torch.Tensor,src_mask:torch.Tensor,tgt:torch.Tensor,tgt_mask:torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask =src_mask,tgt_mask =tgt_mask)
    
    def project(self, x):
        return self.projector(x)        
        
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer