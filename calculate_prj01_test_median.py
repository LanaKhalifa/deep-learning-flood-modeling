import numpy as np
from pathlib import Path
from config.paths_config import RAE_BOXPLOTS_DIR

def calculate_prj01_test_median():
    """Calculate median RAE for prj_01_test (prj_03_test) excluding infinite values."""
    
    # The actual dataset name in the data is prj_03_test
    dataset_name = 'prj_03_test'
    arch_name = 'arch_05'
    
    # Path to the saved RAE values
    rae_file_path = Path(RAE_BOXPLOTS_DIR) / arch_name / f'{dataset_name}_rae_values.npy'
    
    if not rae_file_path.exists():
        print(f"RAE file not found: {rae_file_path}")
        print("You may need to run the RAE calculation first.")
        return
    
    try:
        # Load the RAE values
        rae_values = np.load(rae_file_path)
        
        print(f"=== RAE Analysis for {dataset_name} (displayed as prj_01_test) ===")
        print(f"File: {rae_file_path}")
        
        # Original statistics
        print(f"\nOriginal data:")
        print(f"  Total samples: {len(rae_values)}")
        print(f"  Infinite values: {np.sum(~np.isfinite(rae_values))}")
        print(f"  Finite values: {np.sum(np.isfinite(rae_values))}")
        
        # Clean data (remove infinite values)
        rae_clean = rae_values[np.isfinite(rae_values)]
        
        print(f"\nAfter removing infinite values:")
        print(f"  Clean samples: {len(rae_clean)}")
        print(f"  Removed samples: {len(rae_values) - len(rae_clean)}")
        
        # Calculate statistics
        median_rae = np.median(rae_clean)
        mean_rae = np.mean(rae_clean)
        std_rae = np.std(rae_clean)
        min_rae = np.min(rae_clean)
        max_rae = np.max(rae_clean)
        
        print(f"\nStatistics (excluding infinite values):")
        print(f"  Median RAE: {median_rae:.6f}")
        print(f"  Mean RAE: {mean_rae:.6f}")
        print(f"  Standard Deviation: {std_rae:.6f}")
        print(f"  Minimum RAE: {min_rae:.6f}")
        print(f"  Maximum RAE: {max_rae:.6f}")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            value = np.percentile(rae_clean, p)
            print(f"  {p}th percentile: {value:.6f}")
        
        return median_rae
        
    except Exception as e:
        print(f"Error loading or processing RAE data: {e}")
        return None

if __name__ == "__main__":
    median_rae = calculate_prj01_test_median()
    if median_rae is not None:
        print(f"\n*** ANSWER: The median RAE for prj_01_test (excluding infinite values) is {median_rae:.6f} ***")
