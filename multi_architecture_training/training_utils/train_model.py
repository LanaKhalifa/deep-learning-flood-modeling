# training_utils/train_model.py

import torch
import os
import pickle


def train_model(model, optimizer, train_loader, test_loader, num_epochs, arch_name, device):
    criterion = torch.nn.L1Loss()
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for terrain, input_data, label in train_loader:
            terrain, input_data, label = terrain.to(device), input_data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(terrain, input_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for terrain, input_data, label in test_loader:
                terrain, input_data, label = terrain.to(device), input_data.to(device), label.to(device)
                output = model(terrain, input_data)
                loss = criterion(output, label)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Create directory for saving
    save_dir = os.path.join("trained_models", arch_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    # Save loss curves
    with open(os.path.join(save_dir, "train_val_losses.pkl"), "wb") as f:
        pickle.dump({"train": train_losses, "val": val_losses}, f)

    return train_losses, val_losses
