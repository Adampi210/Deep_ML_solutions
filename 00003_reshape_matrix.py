import torch

def reshape_matrix(a, new_shape) -> torch.Tensor:
    """
    Reshape a 2D matrix `a` to shape `new_shape` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a tensor of shape `new_shape`, or an empty tensor on mismatch.
    """
    # Dimension check
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return torch.tensor([])
    # Convert to tensor and reshape
    a_t = torch.as_tensor(a, dtype=torch.float)
    return a_t.reshape(new_shape)

if __name__ == "__main__":
    # Test result
    a = [[1, 2, 3], [4, 5, 6]]
    new_shape = (3, 2)
    result = reshape_matrix(a, new_shape)  # Expected: tensor([[1., 2.], [3., 4.], [5., 6.]])
    print(result)
    
    # Test fail case
    new_shape_fail = (2, 4)
    result_fail = reshape_matrix(a, new_shape_fail)  # Expected: tensor([])
    print(result_fail)
