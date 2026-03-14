# generate_round2_yamls.py
import yaml
import copy
import os

REMAINING = [w for w in range(1, 25) if w != 16]

with open('s12cdw_24_corn.yaml', 'r') as f:
    template = yaml.safe_load(f)

os.makedirs('fs_yamls_corn_round2', exist_ok=True)

for week in REMAINING:
    config = copy.deepcopy(template)
    combo = f"16_{week}"
    data_root = f"processed_data_fs_{combo}"
    
    # WandB name
    config['trainer']['logger'][0]['init_args']['name'] = f"FS_Corn_S12CDW_week_{combo}"
    
    # Checkpoint path
    config['trainer']['callbacks'][0]['init_args']['dirpath'] = f"output/corn/FS_S12CDW/week_{combo}/checkpoints"
    
    # Data paths
    for split in ['train', 'val', 'test']:
        for mod in ['S2L2A', 'S1GRD', 'CDL', 'DEM', 'WEATHER']:
            config['data']['init_args'][f'{split}_data_root'][mod] = f"{data_root}/{mod}"
        config['data']['init_args'][f'{split}_label_data_root'] = f"{data_root}/yield_geotiffs"
        config['data']['init_args'][f'{split}_split'] = f"{data_root}/{split}_corn.txt"
    
    out_path = f"fs_yamls_corn_round2/config_fs_corn_week_{combo}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ {out_path} 생성완료!")

print("🎉 Round 2 YAML 생성 완료!")