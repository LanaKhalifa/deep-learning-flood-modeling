# A_train_all_archs_on_small_set/train.py
"""
Train multiple architectures on the small dataset with per-model hyperparameters.
"""

import torch
import os
import time

from config import DATALOADERS_ROOT
from training_utils.weights_init import weights_init
from training_utils.train_model import train_model

from A_train_all_archs_on_small_set.architecture_configs import architectures

# Load dataloaders
train_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'small_train_val.pt'))
test_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'small_test.pt'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_INIT = 'xavier'

for arch_name, config in architectures.items():
    print(f"\n▶  Training {arch_name}...")

    if "model_class" not in config or "params" not in config:
        raise ValueError(f"{arch_name} is missing 'model_class' or 'params' keys.")

    start_time = time.time()

    downsampler = config["downsampler_class"](**config["downsampler_params"]).to(device)
    model = config["model_class"](downsampler=downsampler, **config["params"]).to(device)

    model.apply(lambda m: weights_init(m, weight_init=WEIGHT_INIT))
    downsampler.apply(lambda m: weights_init(m, weight_init=WEIGHT_INIT))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config["epochs"],
        arch_name=arch_name,
        device=device
    )

    duration = time.time() - start_time
    print(f" Finished training {arch_name} in {duration / 60:.2f} minutes.")
