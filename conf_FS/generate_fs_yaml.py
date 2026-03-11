"""
Forward Selection: YAML 24개 자동 생성
기존 S12CDW YAML을 템플릿으로 사용
anchor 문제 방지: train/val/test data_root 각각 따로 설정
"""

import os
import copy
import yaml

# ===================== 설정 (여기만 수정!) =====================
TEMPLATE_YAML = "s12cdw_24_corn.yaml"
OUTPUT_DIR = "fs_yamls_corn"
WEEKS = list(range(1, 25))
# ==============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 기존 YAML 읽기
with open(TEMPLATE_YAML, "r") as f:
    template = yaml.safe_load(f)

for week in WEEKS:
    cfg = copy.deepcopy(template)
    data_dir = f"processed_data_fs_{week}"

    # 1. WandbLogger 이름 변경
    cfg["trainer"]["logger"][0]["init_args"]["name"] = f"FS_Corn_S12CDW_week_{week}"

    # 2. Checkpoint 경로 변경
    cfg["trainer"]["callbacks"][0]["init_args"]["dirpath"] = \
        f"output/corn/FS_S12CDW/week_{week}/checkpoints"

    # 3. 데이터 경로 변경 (각각 독립적으로 설정 → anchor 문제 방지)
    def make_data_root(d):
        return {
            "S2L2A":   f"{d}/S2L2A",
            "S1GRD":   f"{d}/S1GRD",
            "CDL":     f"{d}/CDL",
            "DEM":     f"{d}/DEM",
            "WEATHER": f"{d}/WEATHER",
        }

    cfg["data"]["init_args"]["train_data_root"] = make_data_root(data_dir)
    cfg["data"]["init_args"]["val_data_root"]   = make_data_root(data_dir)
    cfg["data"]["init_args"]["test_data_root"]  = make_data_root(data_dir)

    cfg["data"]["init_args"]["train_label_data_root"] = f"{data_dir}/yield_geotiffs"
    cfg["data"]["init_args"]["val_label_data_root"]   = f"{data_dir}/yield_geotiffs"
    cfg["data"]["init_args"]["test_label_data_root"]  = f"{data_dir}/yield_geotiffs"

    cfg["data"]["init_args"]["train_split"] = f"{data_dir}/train_corn.txt"
    cfg["data"]["init_args"]["val_split"]   = f"{data_dir}/val_corn.txt"
    cfg["data"]["init_args"]["test_split"]  = f"{data_dir}/test_corn.txt"

    # 4. YAML 저장
    out_path = os.path.join(OUTPUT_DIR, f"config_fs_corn_week_{week}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Week {week:2d}: {out_path}")

print(f"\n총 {len(WEEKS)}개 YAML 생성 완료! → {OUTPUT_DIR}/")