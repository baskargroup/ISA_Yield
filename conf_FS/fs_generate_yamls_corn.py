import yaml
import copy
import os

# ✅ modify only here!
SELECTED_WEEKS = [16, 20, 1, 12, 18, 19, 17, 15, 5]

selected_str = "_".join(map(str, SELECTED_WEEKS))
round_num = len(SELECTED_WEEKS) + 1
remaining_weeks = [w for w in range(1, 25) if w not in SELECTED_WEEKS]

with open('s12cdw_24_corn.yaml', 'r') as f:
    template = yaml.safe_load(f)

output_dir = f'fs_yamls_corn_round{round_num}'
os.makedirs(output_dir, exist_ok=True)

print(f"🔄 Round {round_num} YAML Generation")
print(f"✅ Selected weeks: {SELECTED_WEEKS}")
print(f"📋 Remaining weeks: {remaining_weeks}")

for week in remaining_weeks:
    config = copy.deepcopy(template)
    combo = f"{selected_str}_{week}"
    data_root = f"processed_data_fs_{combo}"

    # WandB name
    config['trainer']['logger'][0]['init_args']['name'] = f"FS_Corn_S12CDW_week_{combo}"

    # Checkpoint path
    config['trainer']['callbacks'][0]['init_args']['dirpath'] = \
        f"output/corn/FS_S12CDW/week_{combo}/checkpoints"

    # Data paths
    for split in ['train', 'val', 'test']:
        for mod in ['S2L2A', 'S1GRD', 'CDL', 'DEM', 'WEATHER']:
            config['data']['init_args'][f'{split}_data_root'][mod] = f"{data_root}/{mod}"
        config['data']['init_args'][f'{split}_label_data_root'] = f"{data_root}/yield_geotiffs"
        config['data']['init_args'][f'{split}_split'] = f"{data_root}/{split}_corn.txt"

    out_path = f"{output_dir}/config_fs_corn_week_{combo}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ {out_path}")

print(f"🎉 Round {round_num} YAML generation complete!")