# training_utils/train_model.py

import torch
import os
import pickle


def train_model(model, optimizer, train_loader, test_loader, num_epochs, arch_name, device, save_root_dir):
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

    # Save trained model
    model_dir = os.path.join(save_root_dir, "saved_trained_models", arch_name)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

    # Save loss curves
    loss_dir = os.path.join(save_root_dir, "saved_losses", arch_name)
    os.makedirs(loss_dir, exist_ok=True)
    torch.save({'train_losses': train_losses, 'val_losses': val_losses}, os.path.join(loss_dir, "losses.pt"))

    return train_losses, val_losses
