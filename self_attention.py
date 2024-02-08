## Attention in transformers

#Q -> Query (What do I am looking for)
#K -> Key (What can I offer)
#V -> Value (What I actually offer)

#d -> dimmention

#Self Attention = softmax((((Q * K.T)/sqrt(d_k)) + M ) * V )

import numpy as np 
import math

