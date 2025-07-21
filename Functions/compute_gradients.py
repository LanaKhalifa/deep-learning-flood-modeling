#%% add terrain gradients 
import torch 
import torch.nn.functional as F


def compute_gradients(elevation, cell_size):
    """
    Compute the gradients of the terrain given the elevation data and cell size.

    Parameters:
    - elevation: 4D PyTorch tensor of elevation values with shape (N, 1, H, W).
    - cell_size: The cell size in the x and y directions (assumed to be equal).

    Returns:
    - gradients_x: 4D PyTorch tensor of gradient values in the x direction with shape (N, 1, H, W).
    - gradients_y: 4D PyTorch tensor of gradient values in the y direction with shape (N, 1, H, W).
    """
    kernel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float64).unsqueeze(0).unsqueeze(0) / (8 * cell_size)
    kernel_y = torch.tensor([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=torch.float64).unsqueeze(0).unsqueeze(0) / (8 * cell_size)

    gradients_x = F.conv2d(elevation, kernel_x, padding=1)
    gradients_y = F.conv2d(elevation, kernel_y, padding=1)

    return gradients_x, gradients_y

