# simulations_to_samples/scripts/generate_patches.py
"""
Generate patches from HEC-RAS simulation data.
"""
import matplotlib.pyplot as plt
import gc
from simulations_to_samples.scripts.patch_extractor_processor import PatchExtractorProcessor

def process_project(prj_num, prj_name, plans):
    """
    Process a single project to extract patches.
    
    Args:
        prj_num: Project number (str)
        prj_name: Project name (str)
        plans: List of plan numbers (list of str)
    """
    print(f"Processing project {prj_num} ({prj_name}) with {len(plans)} plans...")
    
    # Process each plan
    for plan in plans:
        processor = PatchExtractorProcessor(prj_num, prj_name, plan, plot=True)
        processor.generate_patches()
        plt.close('all')  # Close all figures
        gc.collect()      # Force garbage collection

