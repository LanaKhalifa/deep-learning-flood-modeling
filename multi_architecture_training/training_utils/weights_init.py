# training_utils/weights_init.py

import torch.nn as nn

def weights_init(m, weight_init='xavier'):
    """
    Apply weight initialization to the model layers.

    Args:
        m: A layer from the model.
        weight_init (str): 'xavier' or 'kaiming'
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if weight_init == 'xavier':
            nn.init.xavier_uniform_(m.weight)
        elif weight_init == 'kaiming':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        else:
            raise ValueError(f"Unsupported weight_init: {weight_init}")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
