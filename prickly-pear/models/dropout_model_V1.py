import torch

try:
    from model_utils import *
except ImportError:
    from .model_utils import *

"""
Uncesary?
"""

class SDN(torch.nn.Module):
    """
    Goal: implement a dropout into a torch module
    """

    def __init__(self):
        super(SDN, self).__init__()

    pass


