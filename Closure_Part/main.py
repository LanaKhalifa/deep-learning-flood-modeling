
"""
main.py

Script to run the DeepClosure model over test and train/val plans.
"""

import os
import numpy as np
from deep_closure import DeepClosure
from config import BASE_PROJECT_PATH

test_plans = ['61', '63', '65', '71', '72']
train_val_plans = ['04', '05', '06', '07', '08', '09', '10', 
                   '13', '14', '15', '16', '17', '18', 
                   '20', '21', '22', '23', '24', '25', '26', 
                   '31', '32', '33', '35', '38', '39', '40', 
                   '51', '52', '53', '56', '57', '58']
all_t = [0, 70, 140, 210]

def run_loop(plan_list, label):
    prj_num = '03'
    prj_name = 'hecras_on_03'
    L1_list = []
    RAE_list = []

    for plan_num in plan_list:
        for t in all_t:
            print(f"Running Plan {plan_num} at t = {t}")
            model = DeepClosure(prj_num=prj_num, prj_name=prj_name, plan_num=plan_num, t=t)
            model.run_closure_loop()
            model.plot_all_matrices()
            L1_list.append(model.L1_val)
            RAE_list.append(model.RAE_val)

    # Save results
    save_dir = os.path.join(BASE_PROJECT_PATH, f"Closure/Closure_Loop/L1_{label}_val")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"L1_{label}.npy"), np.array(L1_list))
    np.save(os.path.join(save_dir, f"RAE_{label}.npy"), np.array(RAE_list))

if __name__ == "__main__":
    run_loop(test_plans, "test")
    run_loop(train_val_plans, "train")
