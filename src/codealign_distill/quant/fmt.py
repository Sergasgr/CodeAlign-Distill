"""The ONLY implementation of CodeAlign-Runtime's INT4 format: quantize, dequantize, pack (contract D1)."""
import torch

from src.codealign_distill.quant.config import GROUP_SIZE, Q_MIN, Q_MAX

def compute_scales(weights: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(weights.to(torch.float32)), dim=-1, keepdim=True) / 7.0 
    return torch.where(scale == 0.0, 1.0, scale)

def quantize_with_scales(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scaled_weights = torch.round(weights / scales)
    return scaled_weights.clamp(min=Q_MIN, max=Q_MAX).to(torch.int32) # [-8, 7]

def pack(weights_int32: torch.Tensor, N: int, K: int, G: int) -> torch.Tensor:
    shifts = (torch.arange(8) * 4).to(torch.int32).to(weights_int32.device) 
    quant_weights = weights_int32.view(N, G, 16, 8)
    
    bitwise = (quant_weights << shifts) & (0xF << shifts)
    
    packed = torch.sum(bitwise, dim=-1, keepdim=False)
    return packed.view(N, K // 8, dtype=torch.int32)
    
def unpack(packed) -> tuple[int, torch.Tensor]:
    shifts = (torch.arange(8) * 4).to(packed.device)
    packed = packed.unsqueeze(-1) # [N, K // 8] -> [N, K // 8, 1]
    weights = (((packed >> shifts) & 0xF) << 28) >> 28
        
    N, K, _ = weights.shape
    G = (K * 8) // 128
    return N, weights.view(N, G, 128)
    
def quantize(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix to symmetric INT4 in groups of 128, packing 8 values per int32 and returning the scales in fp32"""
    N, K = weights.shape
    
    if len(weights.shape) != 2:
        return 
    elif K / 128 == 0:
        return
    
    G = K // 128

    weights3d = weights.view(N, K // 128, 128).to(torch.float32)
    
    scales = compute_scales(weights3d)
    scaled_weights = quantize_with_scales(weights, scales)
    packed = pack(scaled_weights.to(torch.int32), N, K, G)
    if not packed.dtype == torch.float32:
        raise ValueError("Packed must be a float32") 

    return packed, scales.squeeze(-1)

def dequantize(unpacked: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize a packed INT4 matrix and its fp32 scales back to fp32 weights"""
    N, weights = unpack(unpacked)
    dequant = weights * scales.unsqueeze(-1)
    return dequant.view(N, -1).to(torch.float32)