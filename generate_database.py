# This script processes HEC-RAS simulation outputs into training data for deep learning.
# Each number in the sublists corresponds to a plan ID (plan_num) — a simulation run in HEC-RAS.
# For memory limitation, we process 7 simulations (plan_nums) at a time using the PatchExtractorProcessor.
# The resulting patches from all 7 simulations are concatenated and saved together as a chunk here: ./Database/saved_in_chunks/{prj_num}/.
# Then chunks are loaded and concatenated

import os
import pickle
import numpy as np
from PatchExtractorProcessor import PatchExtractorProcessor  # Update with actual class name if different


prj_03_sublists = [
    ['04', '05', '06', '07', '08', '09', '10'],
    ['13', '14', '15', '16', '17', '18', '19'],
    ['20', '21', '22', '23', '24', '25', '26'],
    ['31', '32', '33', '35', '38', '39', '40'],
    ['51', '52', '53', '55', '56', '57', '58'],
    ['61', '63', '64', '65', '71', '72', '73']]

prj_04_sublists = [
    ['01', '02', '03', '04', '05', '06', '07'],
    ['08', '10', '12', '13', '14', '15', '16'],
    ['17', '18', '20', '21', '22', '23', '24'],
    ['25', '26', '27', '28', '29', '30', '31'],
    ['34', '35', '37'],
    ['38', '39', '40', '41', '42', '44', '45']]

prj_05_sublists = [
    ['01', '02', '03', '04', '05', '06', '07'],
    ['08', '10', '11', '12', '13', '14', '15'],
    ['16', '17', '18', '19', '20', '21', '22'],
    ['23', '24', '25', '26', '27', '28', '29'],
    ['30', '31', '32', '35', '36', '37', '38'],
    ['39', '40', '43', '44', '45', '46', '47'],
    ['48', '49', '50', '51', '52', '53', '54'],
    ['55', '56', '57', '58', '59', '60', '61'],
    ['62', '63', '64', '65', '66', '67', '68'],
    ['69', '70', '71', '72', '73', '74', '75'],
    ['76', '77', '78', '79', '80', '81', '82'],
    ['83'],
    ['84', '85', '86', '87', '88', '89', '90']]

prj_06_sublists = [
    ['01', '02', '03', '04', '05', '06', '07'],
    ['08', '09', '10', '11', '12', '16', '17'],
    ['18', '19', '20', '21', '22', '23', '26'],
    ['27', '28', '29', '30', '31', '32', '33'],
    ['34', '35', '36', '37', '38', '39', '40'],
    ['41', '42', '43'],
    ['44', '45', '46', '47', '48', '49', '50']]

def process_project(prj_num, prj_name, sublists):
    for idx, plan_nums in enumerate(sublists):
        print(f"Processing sublist {idx} of project {prj_num}")

        instances = []
        for plan_num in plan_nums:
            print('plan_num:', plan_num)
            instance = PatchExtractorProcessor(prj_num, prj_name, plan_num)
            instance.generate_training_data()
            instances.append(instance)

        terrains_list = []
        depths_list = []
        depths_next_list = []

        for instance in instances:
            db = instance.database
            terrains_list.append(db['terrain'])
            depths_list.append(db['depth'])
            depths_next_list.append(db['depth_next'])

        cleaned_terrains_list = []
        cleaned_depths_list = []
        cleaned_depths_next_list = []

        for t, d, dn in zip(terrains_list, depths_list, depths_next_list):
            if len(d) > 0 and len(dn) > 0:
                cleaned_terrains_list.append(t)
                cleaned_depths_list.append(d)
                cleaned_depths_next_list.append(dn)

        terrains = np.concatenate(cleaned_terrains_list, axis=0)
        depths = np.concatenate(cleaned_depths_list, axis=0)
        depths_next = np.concatenate(cleaned_depths_next_list, axis=0)

        save_dir = f'./Database/saved_in_chunks/{prj_num}/'
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, f'terrains_{idx}.pkl'), 'wb') as f:
            pickle.dump(terrains, f)

        with open(os.path.join(save_dir, f'depths_{idx}.pkl'), 'wb') as f:
            pickle.dump(depths, f)

        with open(os.path.join(save_dir, f'depths_next_{idx}.pkl'), 'wb') as f:
            pickle.dump(depths_next, f)

        print("Sublist", idx, "saved successfully.")

# Run processing for all four projects
process_project('03', 'hecras_on_03', prj_03_sublists)
process_project('04', 'HECRAS', prj_04_sublists)
process_project('05', 'HECRAS', prj_05_sublists)
process_project('06', 'HECRAS', prj_06_sublists)
