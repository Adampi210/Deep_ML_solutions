import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a matrix using PyTorch.
    Input: tensor; Output: 1-D tensor with the two eigenvalues in ascending order.
    """
    # Your implementation here
    eig_vals, _ = torch.linalg.eig(matrix)
    return sorted(torch.tensor(eig_vals, dtype=torch.float), reverse=True)


if __name__ == "__main__":
    # Test result
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    result = calculate_eigenvalues(a)  # Expected: tensor([5.3723, -0.3723])
    print(result)