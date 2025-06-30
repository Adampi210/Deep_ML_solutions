import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Dimension mismatch check
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
    # Your implementation here

    return a_t @ b_t

if __name__ == "__main__":
    # Test result
    a = [[1, 2, 3], [4, 5, 6]]
    b = [7, 8, 9]
    result = matrix_dot_vector(a, b) # Expected: tensor([ 50., 122.])
    print(result) 
    
    # Test fail case
    c = [1, 2]
    result_mismatch = matrix_dot_vector(a, c) # Expected: tensor(-1)
    print(result_mismatch)

