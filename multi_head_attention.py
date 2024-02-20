import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Multi_Head_Attention(nn.Module):
    def __init__(self, input_dim, d_model, h) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.h = h
        self.h_dim = d_model//h
        self.qkv_layer = nn.Linear(input_dim, 3*d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, sequence_len, _ = x.size()        
        print(f'X shape: \t{x.shape}')
        qkv = self.qkv_layer(x)
        print(f'QKV shape: \t{qkv.shape}')
        qkv = qkv.reshape(batch_size, sequence_len, self.h, 3*self.h_dim)
        qkv = qkv.permute(0,2,1,3)
        print(f'QKV adjust: \t{qkv.shape}')
        q, k, v = qkv.chunk(3, dim=-1)
        print(f'Q: \t\t{q.shape} \nK: \t\t{k.shape}, \nV: \t\t{v.shape}')
        res = self.scaled_dot_product(Q=q, V=v, K=k, Use_Mask=True)
        print(f'SDP: \t\t{res.shape}')
        concat = res.reshape(batch_size, sequence_len, self.h*self.h_dim)
        print(f'Concat: \t{concat.shape}')
        linear_layer = nn.Linear(self.d_model, self.d_model)
        out = linear_layer(concat)
        print(f'Out layer: \t{out.shape}')
        return(out)

    def scaled_dot_product(self, Q, V, K, Use_Mask=True):
        d_k = K.shape[-1]
        scaled = torch.matmul(Q, K.transpose(-1,-2))/math.sqrt(d_k)
        if Use_Mask:
            mask = torch.full(scaled.size(), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            scaled += mask
        attention = F.softmax(scaled, dim=-1)
        out = torch.matmul(attention, V)
        return out

def main():
    sequence_len = 14
    batch_size = 64
    input_dim = 1024
    d_model = 2048
    h=8
    x = torch.randn((batch_size, sequence_len, input_dim))

    mha = Multi_Head_Attention(input_dim=input_dim, d_model=d_model,h=h)
    out = mha(x)
    print(out.shape)

if __name__ == '__main__':
    main()