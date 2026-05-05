import yaml
import copy
import os

# ✅ modify only here per each round!
START_WEEKS = list(range(1, 17))  # Start from Week 16: [1,2,...,16]

REMOVED_WEEKS = [9, 11, 5, 4, 14, 16, 6, 3, 13, 10, 16]  # Round 1: [], Round 2: [removed week], ...

MODALITIES = ['S2L2A', 'S1GRD', 'WEATHER']  # soybean

# ============================================================
start_str = str(max(START_WEEKS))
removed_str = "_".join(map(str, REMOVED_WEEKS))
round_num = len(REMOVED_WEEKS) + 1

CURRENT_WEEKS = [w for w in START_WEEKS if w not in REMOVED_WEEKS]
CANDIDATE_WEEKS = CURRENT_WEEKS.copy()

with open('s12w_24_soybean.yaml', 'r') as f:
    template = yaml.safe_load(f)

output_dir = f'bs_yamls_soybean_bs{start_str}_round{round_num}'
os.makedirs(output_dir, exist_ok=True)

print(f"🔄 Backward Round {round_num} Soybean YAML Generation (from Week {start_str})")
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
    config['trainer']['logger'][0]['init_args']['project'] = 'M4_Soybean_S12W_BS'
    config['trainer']['logger'][0]['init_args']['name'] = \
        f"BS_Soybean_S12W_bs{start_str}_round{round_num}_rm_{combo}"

    # Checkpoint
    config['trainer']['callbacks'][0]['init_args']['dirpath'] = \
        f"output/soybean/BS_S12W/bs{start_str}/round{round_num}/rm_{combo}/checkpoints"

    # Data paths
    for split in ['train', 'val', 'test']:
        config['data']['init_args'][f'{split}_data_root'] = {
            mod: f"{data_root}/{mod}" for mod in MODALITIES
        }
        config['data']['init_args'][f'{split}_label_data_root'] = f"{data_root}/yield_geotiffs"
        config['data']['init_args'][f'{split}_split'] = f"{data_root}/{split}_soybean.txt"

    out_path = f"{output_dir}/config_bs_soybean_rm_{combo}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ {out_path}")

print(f"🎉 Backward Round {round_num} Soybean YAML generation complete!")