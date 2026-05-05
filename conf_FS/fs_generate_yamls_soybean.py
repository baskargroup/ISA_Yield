import yaml
import copy
import os

# ✅ modify only here per each round!
SELECTED_WEEKS = [18, 5, 12, 8, 7, 11, 17]  # should be empty for round1
MODALITIES = ['S2L2A', 'S1GRD', 'WEATHER']

selected_str = "_".join(map(str, SELECTED_WEEKS))
round_num = len(SELECTED_WEEKS) + 1
remaining_weeks = [w for w in range(1, 25) if w not in SELECTED_WEEKS]

with open('s12w_24_soybean.yaml', 'r') as f:
    template = yaml.safe_load(f)

output_dir = f'fs_yamls_soybean_round{round_num}'
os.makedirs(output_dir, exist_ok=True)

print(f"🔄 Round {round_num} Soybean YAML Generation")
print(f"✅ Selected weeks: {SELECTED_WEEKS}")
print(f"📋 Remaining weeks: {remaining_weeks}")

for week in remaining_weeks:
    config = copy.deepcopy(template)
    combo = str(week) if not SELECTED_WEEKS else f"{selected_str}_{week}"
    data_root = f"processed_data_fs_{combo}"

    # WandB
    config['trainer']['logger'][0]['init_args']['project'] = 'M4_Soybean_S12W_NEW'
    config['trainer']['logger'][0]['init_args']['name'] = f"FS_Soybean_S12W_week_{combo}"

    # Checkpoint
    config['trainer']['callbacks'][0]['init_args']['dirpath'] = \
        f"output/soybean/FS_S12W/week_{combo}/checkpoints"

    # Data paths
    for split in ['train', 'val', 'test']:
        config['data']['init_args'][f'{split}_data_root'] = {
            mod: f"{data_root}/{mod}" for mod in MODALITIES
        }
        config['data']['init_args'][f'{split}_label_data_root'] = f"{data_root}/yield_geotiffs"
        config['data']['init_args'][f'{split}_split'] = f"{data_root}/{split}_soybean.txt"

    out_path = f"{output_dir}/config_fs_soybean_week_{combo}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ {out_path}")

print(f"🎉 Round {round_num} Soybean YAML generation complete!")