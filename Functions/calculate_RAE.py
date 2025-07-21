import torch
def calculate_RAE_batch(prediction_diffs, true_diffs):
    
    with torch.no_grad():
        # Moneh
        diff = torch.abs(true_diffs - prediction_diffs)    
        diff_sum = torch.sum(diff, dim=(1, 2, 3))
        
        # Mechaneh
        dummy_diff = torch.abs(true_diffs)
        dummy_diff_sum = torch.sum(dummy_diff, dim=(1, 2, 3))
        
        # Calculate
        RAE_batch = diff_sum / dummy_diff_sum
        
        RAE_batch_mean = torch.mean(RAE_batch).item()

    return RAE_batch_mean
        