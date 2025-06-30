import torch

def scalar_multiply(matrix, scalar) -> torch.Tensor:
    """
    Multiply each element of a 2D matrix by a scalar using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of the same shape.
    """
    # Convert input to tensor
    m_t = torch.as_tensor(matrix, dtype=torch.float)
    return scalar * m_t

if __name__ == "__main__":
    # Test result
    a = [[1, 2, 3], [4, 5, 6]]
    scalar = 2
    result = scalar_multiply(a, scalar)  # Expected: tensor([[2., 4., 6.], [8., 10., 12.]])
    print(result)