import yaml
import copy
import os

# ✅ modify only here per each round!
# START_WEEKS = list(range(1, 21))  # Start from Week 20: [1,2,...,20]
START_WEEKS = list(range(1, 17))  # Start from Week 16: [1,2,...,16]

REMOVED_WEEKS = [11, 12, 6, 14, 10, 4, 2, 8, 3, 15, 9]  # Round 1: [], Round 2: [removed week], ...

MODALITIES = ['S2L2A', 'S1GRD', 'CDL', 'DEM', 'WEATHER']  # corn

# ============================================================
start_str = str(max(START_WEEKS))
removed_str = "_".join(map(str, REMOVED_WEEKS))
round_num = len(REMOVED_WEEKS) + 1

CURRENT_WEEKS = [w for w in START_WEEKS if w not in REMOVED_WEEKS]
CANDIDATE_WEEKS = CURRENT_WEEKS.copy()

with open('s12cdw_24_corn.yaml', 'r') as f:
    template = yaml.safe_load(f)

output_dir = f'bs_yamls_corn_bs{start_str}_round{round_num}'
os.makedirs(output_dir, exist_ok=True)

print(f"🔄 Backward Round {round_num} Corn YAML Generation (from Week {start_str})")
print(f"✅ Current weeks: {CURRENT_WEEKS}")
print(f"📋 Candidate weeks to remove: {CANDIDATE_WEEKS}")

for remove_week in CANDIDATE_WEEKS:
    config = copy.deepcopy(template)

    if REMOVED_WEEKS:
        combo = f"{removed_str}_{remove_week}"
        data_root = f"processed_data_bs{start_str}_rm_{combo}"
    else:
        combo = str(remove_week)
        data_root = f"processed_data_bs{start_str}_rm_{combo}"

    # WandB
    config['trainer']['logger'][0]['init_args']['project'] = 'M4_Corn_S12WDC_BS'
    config['trainer']['logger'][0]['init_args']['name'] = \
        f"BS_Corn_S12WDC_bs{start_str}_round{round_num}_rm_{combo}"

    # Checkpoint
    config['trainer']['callbacks'][0]['init_args']['dirpath'] = \
        f"output/corn/BS_S12WDC/bs{start_str}/round{round_num}/rm_{combo}/checkpoints"

    # Data paths
    for split in ['train', 'val', 'test']:
        config['data']['init_args'][f'{split}_data_root'] = {
            mod: f"{data_root}/{mod}" for mod in MODALITIES
        }
        config['data']['init_args'][f'{split}_label_data_root'] = f"{data_root}/yield_geotiffs"
        config['data']['init_args'][f'{split}_split'] = f"{data_root}/{split}_corn.txt"

    out_path = f"{output_dir}/config_bs_corn_rm_{combo}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ {out_path}")

print(f"🎉 Backward Round {round_num} Corn YAML generation complete!")