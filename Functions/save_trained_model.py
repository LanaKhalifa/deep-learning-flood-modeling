import torch

def save_trained_model(netG, model_path):
    torch.save(netG.state_dict(), model_path)
