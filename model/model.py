import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Union, Tuple, Iterable

class ModelArgs:
    def __init__(
            self, 
            c_z: int, # pair representation embed dim
            c_m: int, # MSA embed dim
            tf_dim: int, # target feature embed dim
            f_e: int,# extra msa dimension
            c_e: int,
            max_seq_len: int = None,
            **kwargs
            ):
        super().__init__()
        # Evoformer arguments
        self.c_m = c_m
        self.tf_dim = tf_dim
        self.f_e = f_e
        self.c_e = c_e
    


### Model





### Data preprocess


### Interact