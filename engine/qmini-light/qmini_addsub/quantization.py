from __future__ import annotations
import math
import numpy as np
import torch
WEIGHT_SYMBOLS=(-1.5,-0.5,0.0,0.5,1.5)
SYMBOL_COUNT=5
BITS_PER_WEIGHT=3
LEGACY_FORMAT_NAME="qmini-addsub-v1"
FORMAT_NAME="qmini-addsub-v2-scaled-five-symbol"
WEIGHT_SCALE_MULTIPLIER=1.4
MIN_WEIGHT_SCALE=1e-8

def weight_scale(weight:torch.Tensor)->torch.Tensor:
    """Detached positive per-tensor scale for five-symbol QAT."""
    if not isinstance(weight,torch.Tensor) or not weight.is_floating_point(): raise TypeError("Weight scale requires floating-point weights.")
    if weight.numel()==0: raise ValueError("Cannot scale an empty weight tensor.")
    if not bool(torch.isfinite(weight).all()): raise ValueError("Non-finite weights cannot be quantized.")
    fixed_scale = getattr(weight, "_qmini_fixed_scale", None)
    if fixed_scale is not None:
        value = torch.as_tensor(fixed_scale, device=weight.device, dtype=weight.dtype)
        if value.numel() != 1 or not bool(torch.isfinite(value)) or float(value.item()) <= 0:
            raise ValueError("Frozen quantization scale is invalid.")
        return value
    return (weight.detach().abs().mean()*WEIGHT_SCALE_MULTIPLIER).clamp_min(MIN_WEIGHT_SCALE)

def _quantize_normalized(weight:torch.Tensor)->torch.Tensor:
    x=weight.clamp(-1.5,1.5)
    return torch.where(x<=-1.0,-1.5,torch.where(x<=-0.25,-0.5,torch.where(x<=0.25,0.0,torch.where(x<=1.0,0.5,1.5))))

def quantize_weight_ste(weight: torch.Tensor, scale: torch.Tensor | float | None = None) -> torch.Tensor:
    """Per-tensor scaled five-symbol quantization; identity STE gradient at zero."""
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("Weight quantization requires a floating-point torch.Tensor.")
    if weight.numel() == 0:
        raise ValueError("Cannot quantize an empty weight tensor.")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Non-finite weights cannot be quantized.")

    if scale is None:
        scale_tensor = weight_scale(weight)
    else:
        if isinstance(scale, bool):
            raise TypeError("Scale must be a real scalar, not a boolean.")
        if isinstance(scale, torch.Tensor):
            if scale.numel() != 1 or not scale.is_floating_point():
                raise TypeError("Scale tensor must contain one floating-point value.")
        elif not isinstance(scale, (int, float, np.integer, np.floating)):
            raise TypeError("Scale must be a real scalar or one-element floating tensor.")
        scale_tensor = torch.as_tensor(
            scale, device=weight.device, dtype=weight.dtype
        ).detach()
        if (
            scale_tensor.numel() != 1
            or not bool(torch.isfinite(scale_tensor).all())
            or float(scale_tensor.item()) <= 0.0
        ):
            raise ValueError("Scale must be one finite positive scalar.")

    effective = _quantize_normalized(weight / scale_tensor) * scale_tensor
    return weight + (effective - weight).detach()

def initialize_five_symbol_weight_(weight: torch.Tensor) -> torch.Tensor:
    """Dense Kaiming master weights for scaled five-symbol QAT."""
    if not isinstance(weight, torch.Tensor):
        raise TypeError("Weight initialization requires a torch.Tensor.")
    if weight.ndim < 2 or any(size < 1 for size in weight.shape):
        raise ValueError("Weight must have at least two non-empty dimensions.")
    if not weight.is_floating_point():
        raise TypeError("Weight must be floating point.")
    with torch.no_grad():
        torch.nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="linear")
    return weight

def quantize_to_codes(weight: np.ndarray, scale: float | None = None) -> np.ndarray:
    raw = np.asarray(weight)
    if raw.dtype.kind not in "iuf":
        raise TypeError("Weights must be a real numeric array.")
    with np.errstate(over="ignore", invalid="ignore"):
        values = raw.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite weights cannot be quantized.")
    if scale is not None:
        if isinstance(scale, (bool, np.bool_)):
            raise TypeError("Scale must be a real number, not a boolean.")
        if not isinstance(scale, (int, float, np.integer, np.floating)):
            raise TypeError("Scale must be a real scalar.")
        scale_value = float(scale)
        if not math.isfinite(scale_value) or scale_value <= 0:
            raise ValueError("Scale must be finite and positive.")
        values = values / np.float32(scale_value)
    clipped = np.clip(values, -1.5, 1.5)
    levels = np.asarray(WEIGHT_SYMBOLS, dtype=np.float32)
    return np.argmin(np.abs(clipped[..., None] - levels), axis=-1).astype(np.uint8)

def codes_to_values(codes:np.ndarray)->np.ndarray:
    v=np.asarray(codes)
    if v.dtype.kind not in "iu" or np.any(v<0) or np.any(v>4): raise ValueError("Codes must be integers in [0, 4].")
    return np.asarray(WEIGHT_SYMBOLS,dtype=np.float32)[v]

def pack_3bit_codes(codes:np.ndarray)->bytes:
    v=np.asarray(codes)
    if v.dtype.kind not in "iu" or np.any(v<0) or np.any(v>4): raise ValueError("Only codes 0..4 can be packed.")
    flat=v.reshape(-1); out=bytearray((len(flat)*3+7)//8)
    for i,item in enumerate(flat):
        offset=i*3; byte,shift=offset>>3,offset&7; code=int(item)
        out[byte]|=(code<<shift)&255
        if shift>5: out[byte+1]|=code>>(8-shift)
    return bytes(out)

def unpack_3bit_codes(payload:bytes,shape:tuple[int,...])->np.ndarray:
    if not isinstance(payload,(bytes,bytearray,memoryview)): raise TypeError("Packed weights must be bytes-like.")
    if not isinstance(shape,(tuple,list)) or not shape: raise ValueError("Shape must be non-empty.")
    if any(isinstance(d,bool) or not isinstance(d,(int,np.integer)) or d<0 for d in shape): raise ValueError("Packed weight dimensions must be non-negative integers.")
    count=math.prod(int(d) for d in shape); expected=(count*3+7)//8
    if len(payload)!=expected: raise ValueError(f"Packed size mismatch: expected {expected}, got {len(payload)}.")
    rem=(count*3)%8
    if rem and payload and (int(payload[-1])>>rem): raise ValueError("Packed payload has non-zero unused padding bits.")
    out=np.empty(count,dtype=np.uint8)
    for i in range(count):
        offset=i*3; byte,shift=offset>>3,offset&7; value=int(payload[byte])>>shift
        if shift>5: value|=int(payload[byte+1])<<(8-shift)
        code=value&7
        if code>4: raise ValueError(f"Reserved Add/Sub v1 weight code {code} at index {i}.")
        out[i]=code
    return out.reshape(shape)
