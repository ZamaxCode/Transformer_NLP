import torch
import torch.nn as nn

class Positional_Encoding(nn.Module):
    def __init__(self, max_sequence_len, d_model) -> None:
        super().__init__()
        self.max_sequence_len = max_sequence_len
        self.d_model = d_model

    def forward(self):
        i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, i/self.d_model)
        position = torch.arange(self.max_sequence_len, dtype=torch.float).reshape(self.max_sequence_len,1)
        PE_even = torch.sin(position/denominator)
        PE_odd = torch.cos(position/denominator)
        stacked = torch.stack([PE_even, PE_odd], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

def main():
    PE = Positional_Encoding(max_sequence_len=10,d_model=6)
    out = PE()
    print(out.shape)
    print(out)

if __name__ == '__main__':
    main()