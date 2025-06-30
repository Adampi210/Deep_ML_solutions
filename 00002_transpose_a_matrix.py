import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix `a` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a transposed tensor.
    """
    a_t = torch.as_tensor(a)
    return a_t.T

if __name__ == "__main__":
    # Test result
    a = [[1, 2, 3], [4, 5, 6]]
    result = transpose_matrix(a)
    print(result)  # Expected: tensor([[1, 4], [2, 5], [3, 6]])
    
    # Test with a different shape
    b = [[1, 2], [3, 4], [5, 6]]
    result_b = transpose_matrix(b)
    print(result_b)  # Expected: tensor([[1, 3, 5], [2, 4, 6]])
