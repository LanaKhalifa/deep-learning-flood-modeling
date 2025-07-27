import pickle
from config.data_config import prjs_metadata
from config.paths_config import PATCHES_DIR, DATASETS_DIR

# Output dataset directories
SMALL_DIR = DATASETS_DIR / 'small_dataset'
BIG_DIR = DATASETS_DIR / 'big_dataset'
PRJ_03_DIR = DATASETS_DIR / 'prj_03_dataset'
# Ensure output directories exist
for path in [SMALL_DIR, BIG_DIR, PRJ_03_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def load_plan_patches(prj_num, plan_num):
    # TEMPORARY: Use existing patches with spaces in directory names
    terrain_path = PATCHES_DIR / f'prj_{prj_num}/plan_{plan_num}' /  'terrain_patches.pkl'
    depth_path = PATCHES_DIR / f'prj_{prj_num}/plan_{plan_num}' / 'depth_patches.pkl'
    depth_next_path = PATCHES_DIR / f'prj_{prj_num}/plan_{plan_num}' / 'depth_next_patches.pkl'

    with open(terrain_path, 'rb') as f:
        terrain = pickle.load(f)
    with open(depth_path, 'rb') as f:
        depth = pickle.load(f)
    with open(depth_next_path, 'rb') as f:
        depth_next = pickle.load(f)

    return terrain, depth, depth_next


def build_dataset(prj_plan_list):
    dataset = {'terrain': [], 'depth': [], 'depth_next': []}
    
    for prj_num, plan_num in prj_plan_list:
        terrain, depth, depth_next = load_plan_patches(prj_num, plan_num)
        dataset['terrain'].extend(terrain)
        dataset['depth'].extend(depth)
        dataset['depth_next'].extend(depth_next)
    
    return dataset

def save_dataset(dataset, directory, name):
    path = directory / f'{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{name} dataset saved successfully!")

def create_and_save_datasets():
    # Prepare dataset lists
    big_train_val_list = []
    big_test_list = []

    for prj_num, (_, plan_list) in prjs_metadata.items():
        big_train_val_list += [(prj_num, p) for p in plan_list[:-7]]
        big_test_list += [(prj_num, p) for p in plan_list[-7:]]

    prj_03_plans = prjs_metadata['03'][1]
    prj_03_train_val_list = [('03', p) for p in prj_03_plans[:-7]]
    prj_03_test_list = [('03', p) for p in prj_03_plans[-7:]]
    small_train_val_list = [('03', p) for p in prj_03_plans[:7]]

    # --- BIG DATASET ---
    save_dataset(build_dataset(big_train_val_list), BIG_DIR, 'big_train_val')
    save_dataset(build_dataset(big_test_list), BIG_DIR, 'big_test')

    # --- PRJ_03 ONLY ---
    save_dataset(build_dataset(prj_03_train_val_list), PRJ_03_DIR, 'prj_03_train_val')
    save_dataset(build_dataset(prj_03_test_list), PRJ_03_DIR, 'prj_03_test')

    # --- SMALL DATASET ---
    save_dataset(build_dataset(small_train_val_list), SMALL_DIR, 'small_train_val')