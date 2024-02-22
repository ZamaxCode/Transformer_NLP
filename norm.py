import torch
from torch import nn

class LayerNormalization():
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y  + self.beta
        return out
    

def main():
    batch_size = 3
    sentence_length = 5
    embedding_dim = 8 
    inputs = torch.randn(sentence_length, batch_size, embedding_dim)
    layer_norm = LayerNormalization(inputs.size()[-1:])
    out = layer_norm.forward(inputs)
    print(out)

if __name__ == '__main__':
    main()