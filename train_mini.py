import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import Counter
import numpy as npfrom MHA import init_qkv_proj, self_attention


