"""Fake quantization for QAT: round_ste, grad_scale, FakeQuantLinear (STE and LSQ)."""
import torch
from torch import nn
import torch.nn.functional as F
import math

from src.codealign_distill.quant.config import GROUP_SIZE, Q_MIN, Q_MAX

def round_ste(x: torch.Tensor) -> torch.Tensor: # Straight-Through Estimator
    """Round the tensors to integer numbers to simulate INT4 format. Rounding derivative is 0 which lies on vanishing gradient. We want INT4 in forward and intact gradient of the float number in backward? (it has to act like the identity matrix with derivative 1)"""
    
    delta = (torch.round(x) - x).detach()
    return delta + x


def grad_scale(x: torch.Tensor, g: float) -> torch.Tensor: # LSQ: Learned Step Size Quantization
    """In LSQ (QAT) the model updates the scales as an extra parameter during training. To keep this training stable, the gradient of these scales must be scaled in a factor of g"""
    
    delta = (x - x * g).detach()
    return delta + x * g 

def get_effective_scales():
    """It returns the scales protected by the lower bound. It guarantees that the exported checkpoint is identical to the inference state"""
    pass

class FakeQuantLinear(nn.Module): 
    """Wrapper of a linear standard layer for simulate effects of QAT INT4 during forward maintaining the float weights in memory"""
    def __init__(self, linear_layer: nn.Linear, scales: torch.tensor, learn_scale: bool): 
        super().__init__()
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.learn_scale = learn_scale
        self.scales = nn.Parameter(scales.float().clone(), requires_grad=self.learn_scale)
        if self.scales.shape == 2:
            _, K = self.scales.shape
            if K % 128 != 0:
                raise ValueError("K must be divisible by 128")
        else:
            raise ValueError("Scales must be [N, K // 128]")
    
    def quantized_weight(self):
        """It simulates numerically the rounds and limits of INT4 in each pass"""
        N, K = self.weight.shape
        g = 1.0 / math.sqrt(GROUP_SIZE * Q_MAX)
        
        weights = self.weight.view(N, K // GROUP_SIZE, GROUP_SIZE)
        
        scales = self.scales.unsqueeze(-1)
        if self.learn_scale:
            scales = grad_scale(self.scales, g).unsqueeze(-1) 
        scales = torch.where(scales == 0.0, 1e-8, scales)
        
        scaled_weights = weights / scales
        rounded_weights = round_ste(scaled_weights.clamp(min=Q_MIN, max=Q_MAX)) # [-8, 7]
        
        return (rounded_weights * scales).view(N, K)
        
    def forward(self, x):
        return F.linear(x, self.quantized_weight(), self.bias)

if __name__ == "__main__":
    N = 64; K = 256
    linear_layer = nn.Linear(K, N) 
    scales = torch.randn(N, K // 128)
    fakeQuant = FakeQuantLinear(linear_layer, scales, True)
    x = torch.randn(16, K)
    assert fakeQuant(x).shape == (16, 64)