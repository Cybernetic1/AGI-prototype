import torch

def lorentz_inner_product(u, v):
    """
    Computes the Lorentz (Minkowski) inner product.
    <u, v>_L = -u_0*v_0 + u_1*v_1 + ... + u_n*v_n
    
    Args:
        u, v: Tensors of shape (..., L), where L is the embedding dimension.
              The 0-th dimension is the time-like coordinate.
    Returns:
        Tensor of shape (...) containing the inner products.
    """
    xy = u * v
    # Time-like component gets negative sign
    return -xy[..., 0] + xy[..., 1:].sum(dim=-1)

def lorentz_distance(u, v, eps=1e-5):
    """
    Computes the distance between points u and v in the Lorentz model.
    d(u,v) = arcosh(-<u, v>_L)
    
    Args:
        u, v: Tensors of shape (..., L).
        eps: Small value to prevent NaN in arcosh if -<u,v>_L is slightly < 1 due to numerical error.
    Returns:
        Tensor of shape (...) containing the distances.
    """
    inner = lorentz_inner_product(u, v)
    # The inner product for points on the hyperboloid should be <= -1.
    # Therefore, -inner should be >= 1. We clamp to 1 + eps for numerical stability.
    val = torch.clamp(-inner, min=1.0 + eps)
    return torch.acosh(val)

def project_to_hyperboloid(x, eps=1e-5):
    """
    Projects arbitrary vectors in R^L onto the upper sheet of the Lorentz hyperboloid.
    The upper sheet is defined by <x, x>_L = -1 and x_0 > 0.
    
    Given a spatial vector x_s = (x_1, ..., x_n), we compute the time-like coordinate:
    x_0 = sqrt(1 + ||x_s||^2)
    
    Args:
        x: Tensor of shape (..., L). We ignore the 0-th coordinate and replace it.
    Returns:
        Tensor of shape (..., L) on the hyperboloid.
    """
    spatial = x[..., 1:]
    spatial_norm_sq = (spatial ** 2).sum(dim=-1)
    x_0 = torch.sqrt(1.0 + spatial_norm_sq + eps)
    return torch.cat([x_0.unsqueeze(-1), spatial], dim=-1)
