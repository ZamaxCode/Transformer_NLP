import numpy as np 
import math

def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

def self_attention(Q,V,K, Mask=None):
    d_k = K.shape[-1]
    scaled = ( np.matmul(Q, K.T)/math.sqrt(d_k))
    if Mask is not None:
        scaled += Mask
    attention = softmax(scaled)
    out = np.matmul(attention, V)
    return out

def main():
    phrase = "Hola mi nombre es Alex"
    input_phrase = phrase.split(' ')
    in_len = len(input_phrase)

    d_K, d_V = 8, 8
    q = np.random.randn(in_len, d_K)
    k = np.random.randn(in_len, d_K)
    v = np.random.randn(in_len, d_V)
    
    mask = np.tril(np.ones( (in_len, in_len) ))
    mask[mask == 0] = -np.infty
    mask[mask == 1] = 0

    out = self_attention(Q=q,V=v,K=k, Mask=mask)

    print(out)

main()