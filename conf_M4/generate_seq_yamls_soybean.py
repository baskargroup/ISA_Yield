import os

template_yaml = open('s12wsc_24_soybean.yaml').read()

for n in range(1, 25):
    new_yaml = template_yaml

    # data path change
    new_yaml = new_yaml.replace(
        'processed_data_weekly_24',
        f'processed_data_weekly_{n}'
    )

    # wandb name change
    new_yaml = new_yaml.replace(
        'name: S12WSC',
        f'name: S12WSC_seq_week{n}'
    )

    # checkpoint path change
    new_yaml = new_yaml.replace(
        'dirpath: output/soybean/M4/S12WSC_stat/checkpoints',
        f'dirpath: output/soybean/seq/S12WSC/week{n}/checkpoints'
    )

    fname = f's12wsc_{n}_soybean.yaml'
    with open(fname, 'w') as f:
        f.write(new_yaml)
    print(f"✅ {fname} created")

print("🎉 Done!")