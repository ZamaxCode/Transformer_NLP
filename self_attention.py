import numpy as np 
import math

def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

def scaled_dot_product(Q,V,K, d_k, Mask=None):
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

    out = scaled_dot_product(Q=q,V=v,K=k, d_k=d_K, Mask=mask)

    print(out)

if __name__ == '__main__':
    main()