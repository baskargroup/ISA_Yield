import os

WEATHER_MEANS = """      WEATHER:
      - 23.818
      - 24.606
      - 12.064
      - 373.494
      - 1482.514
      - 0.547
      - 49955.316"""

WEATHER_STDS = """      WEATHER:
      - 7.991
      - 1.232
      - 1.241
      - 11.317
      - 120.417
      - 1.337
      - 504.205"""

for n in range(1, 25):
    with open(f's12dc_{n}_corn.yaml', 'r') as f:
        content = f.read()

    # project name
    content = content.replace('project: M4_corn', 'project: M4_corn_S12CDW')

    # wandb name
    content = content.replace(f'name: S12DC_seq_week{n}', f'name: S12CDW_seq_week{n}')

    # checkpoint path
    content = content.replace(
        f'dirpath: output/corn/seq/S12DC/week{n}/checkpoints',
        f'dirpath: output/corn/seq/S12CDW/week{n}/checkpoints'
    )

    # modalities 추가
    content = content.replace(
        '    - DEM\n    - CDL\n    rgb_modality:',
        '    - DEM\n    - CDL\n    - WEATHER\n    rgb_modality:'
    )

    # data root에 WEATHER 추가
    content = content.replace(
        f'      CDL: ../processed_data/weekly_24/processed_data_weekly_{n}/CDL',
        f'      CDL: ../processed_data/weekly_24/processed_data_weekly_{n}/CDL\n      WEATHER: ../processed_data/weekly_24/processed_data_weekly_{n}/WEATHER'
    )

    # means에 WEATHER 추가
    content = content.replace(
        '      CDL:\n      - 0\n    stds:',
        f'      CDL:\n      - 0\n{WEATHER_MEANS}\n    stds:'
    )

    # stds에 WEATHER 추가
    content = content.replace(
        '      CDL:\n      - 1\nmodel:',
        f'      CDL:\n      - 1\n{WEATHER_STDS}\nmodel:'
    )

    # backbone_modalities에 WEATHER 추가
    content = content.replace(
        '      - DEM\n      - CDL: 1\n      backbone_merge_method:',
        '      - DEM\n      - CDL: 1\n      - WEATHER: 7\n      backbone_merge_method:'
    )

    fname = f's12cdw_{n}_corn.yaml'
    with open(fname, 'w') as f:
        f.write(content)
    print(f"✅ {fname} created")

print("🎉 Done!")