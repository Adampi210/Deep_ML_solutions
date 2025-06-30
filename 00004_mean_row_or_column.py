import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    if mode == 'column':
        return torch.mean(a_t, dim=0)
    elif mode == 'row':
        return torch.mean(a_t, dim=1)
    else:
        raise ValueError("Mode must be 'row' or 'column'.")
    
if __name__ == "__main__":
    # Test result
    a = [[1, 2, 3], [4, 5, 6]]
    result_column = calculate_matrix_mean(a, 'column')  # Expected: tensor([2.5, 3.5, 4.5])
    print(result_column)

    result_row = calculate_matrix_mean(a, 'row')  # Expected: tensor([2., 5.])
    print(result_row)

    # Test fail case
    try:
        result_fail = calculate_matrix_mean(a, 'invalid_mode')
    except ValueError as e:
        print(e)  # Expected: "Mode must be 'row' or 'column'."