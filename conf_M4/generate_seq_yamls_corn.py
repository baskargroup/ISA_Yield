import os

template_yaml = open('s12dc_24_corn.yaml').read()

for n in range(1, 25):
    new_yaml = template_yaml

    # data path change
    new_yaml = new_yaml.replace(
        'processed_data_weekly_24',
        f'processed_data_weekly_{n}'
    )

    # wandb name change
    new_yaml = new_yaml.replace(
        'name: S12DC_seq_week24',
        f'name: S12DC_seq_week{n}'
    )

    # checkpoint path change
    new_yaml = new_yaml.replace(
        'dirpath: output/corn/seq/S12DC/week24/checkpoints',
        f'dirpath: output/corn/seq/S12DC/week{n}/checkpoints'
    )

    fname = f's12dc_{n}_corn.yaml'
    with open(fname, 'w') as f:
        f.write(new_yaml)
    print(f"✅ {fname} created")

print("🎉 Done!")