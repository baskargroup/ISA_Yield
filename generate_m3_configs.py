#!/usr/bin/env python3
"""
Generate YAML configuration files for all modality combinations in conf_M3.
"""

import os
import yaml

# Statistics for each modality
STATS = {
    'S2L2A': {
        'means': [1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975, 2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648],
        'stds': [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311]
    },
    'S1GRD': {
        'means': [-12.599, -20.293],
        'stds': [5.195, 5.890]
    },
    'DEM': {
        'means': [670.665],
        'stds': [951.272]
    },
    'WEATHER': {
        'means': [23.818, 24.606, 12.064, 373.494, 1482.514, 0.547, 49955.316],
        'stds': [7.991, 1.232, 1.241, 11.317, 120.417, 1.337, 504.205]
    },
    'SOIL': {
        'means': [0.812, 0.787, 0.741, 20128.930, 21412.781, 253.445, 87.680, 182.086, 264.623, 335.160],
        'stds': [0.064, 0.071, 0.079, 6099.627, 6499.918, 44.726, 4.492, 15.991, 28.075, 41.569]
    },
    'CDL': {
        'means': [0],
        'stds': [1]
    }
}

# Map modality codes to modality names
MODALITY_MAP = {
    's': 'SOIL',
    'w': 'WEATHER',
    'd': 'DEM',
    'c': 'CDL'
}

# Base modalities always included
BASE_MODALITIES = ['S2L2A', 'S1GRD']

def code_to_modalities(code):
    """Convert a code like 's12wds' to list of modalities."""
    modalities = BASE_MODALITIES.copy()
    
    # Extract the modality letters (after 's12')
    extra_mods = code[3:]  # Skip 's12'
    
    for char in extra_mods:
        if char in MODALITY_MAP:
            mod_name = MODALITY_MAP[char]
            if mod_name not in modalities:
                modalities.append(mod_name)
    
    return modalities

def modalities_to_backbone(modalities):
    """Convert modalities list to backbone_modalities format."""
    backbone_mods = []
    for mod in modalities:
        if mod in ['S2L2A', 'S1GRD', 'DEM']:
            backbone_mods.append(mod)
        else:
            # Add with band count
            if mod == 'WEATHER':
                backbone_mods.append({'WEATHER': 7})
            elif mod == 'SOIL':
                backbone_mods.append({'SOIL': 10})
            elif mod == 'CDL':
                backbone_mods.append({'CDL': 1})
    return backbone_mods

def modalities_to_means_stds(modalities):
    """Convert modalities list to means and stds dicts."""
    means = {}
    stds = {}
    for mod in modalities:
        if mod in STATS:
            means[mod] = STATS[mod]['means']
            stds[mod] = STATS[mod]['stds']
    return means, stds

def modalities_to_data_roots(modalities):
    """Create data root dictionaries for modalities."""
    base_path = '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/processed_data/weekly_24'
    data_root = {}
    for mod in modalities:
        data_root[mod] = f'{base_path}/{mod}'
    return data_root

def generate_yaml(code, crop):
    """Generate YAML content for given code and crop."""
    modalities = code_to_modalities(code)
    backbone_mods = modalities_to_backbone(modalities)
    means, stds = modalities_to_means_stds(modalities)
    data_roots = modalities_to_data_roots(modalities)
    config_name = code.upper()
    
    # Build the config dictionary
    config = {
        'seed_everything': 42,
        'trainer': {
            'accelerator': 'auto',
            'strategy': 'auto',
            'devices': 'auto',
            'num_nodes': 1,
            'precision': '16-mixed',
            'logger': [
                {
                    'class_path': 'WandbLogger',
                    'init_args': {
                        'project': f'M3_{crop}',
                        'name': f'{config_name}_le_5_differentnorm'
                    }
                }
            ],
            'callbacks': [
                {
                    'class_path': 'ModelCheckpoint',
                    'init_args': {
                        'dirpath': f'output/{crop}/M3/differentnorm/{config_name}_le_5/checkpoints',
                        'monitor': 'val/loss',
                        'mode': 'min',
                        'save_top_k': 1,
                        'save_last': True,
                        'filename': 'best-{epoch:03d}'
                    }
                },
                {
                    'class_path': 'EarlyStopping',
                    'init_args': {
                        'monitor': 'val/loss',
                        'patience': 75,
                        'min_delta': 0.0001,
                        'mode': 'min',
                        'verbose': True
                    }
                },
                {
                    'class_path': 'LearningRateMonitor',
                    'init_args': {
                        'logging_interval': 'epoch'
                    }
                },
                {
                    'class_path': 'RichProgressBar'
                }
            ],
            'max_epochs': 1000,
            'log_every_n_steps': 1,
            'accumulate_grad_batches': 8
        },
        'data': {
            'class_path': 'terratorch.datamodules.GenericMultiModalDataModule',
            'init_args': {
                'task': 'regression',
                'batch_size': 4,
                'num_workers': 8,
                'modalities': modalities,
                'rgb_modality': 'S2L2A',
                'rgb_indices': [3, 2, 1],
                'train_data_root': data_roots,
                'train_label_data_root': '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/processed_data/weekly_24/yield_geotiffs',
                'val_data_root': data_roots,
                'val_label_data_root': '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/processed_data/weekly_24/yield_geotiffs',
                'test_data_root': data_roots,
                'test_label_data_root': '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/processed_data/weekly_24/yield_geotiffs',
                'train_split': f'/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/processed_data/weekly_24/train_{crop}.txt',
                'val_split': f'/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/processed_data/weekly_24/val_{crop}.txt',
                'test_split': f'/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/processed_data/weekly_24/test_{crop}.txt',
                'expand_temporal_dimension': False,
                'no_label_replace': 0.0,
                'no_data_replace': 0,
                'means': means,
                'stds': stds
            }
        },
        'model': {
            'class_path': 'terratorch.tasks.PixelwiseRegressionTask',
            'init_args': {
                'model_factory': 'EncoderDecoderFactory',
                'model_args': {
                    'backbone': 'terramind_v1_base',
                    'backbone_pretrained': True,
                    'backbone_modalities': backbone_mods,
                    'backbone_merge_method': 'mean',
                    'necks': [
                        {
                            'name': 'SelectIndices',
                            'indices': [2, 5, 8, 11]
                        },
                        {
                            'name': 'ReshapeTokensToImage',
                            'remove_cls_token': False
                        },
                        {
                            'name': 'LearnedInterpolateToPyramidal'
                        }
                    ],
                    'decoder': 'UNetDecoder',
                    'decoder_channels': [512, 256, 128, 64],
                    'head_dropout': 0.1
                },
                'loss': 'mse',
                'ignore_index': -1,
                'freeze_backbone': False,
                'freeze_decoder': False
            }
        },
        'optimizer': {
            'class_path': 'torch.optim.AdamW',
            'init_args': {
                'lr': 1.e-5
            }
        },
        'lr_scheduler': {
            'class_path': 'ReduceLROnPlateau',
            'init_args': {
                'monitor': 'val/loss',
                'factor': 0.8,
                'patience': 25,
                'min_lr': 0.000001
            }
        }
    }
    
    return config

def main():
    """Generate all YAML configuration files."""
    conf_dir = '/work/mech-ai-scratch/aapowadi/ISA_Yield/conf_M3'
    os.makedirs(conf_dir, exist_ok=True)
    
    # Modality codes from Modality_Config.csv
    codes = [
        's12wd',   # S12wd
        's12',     # S12
        's12d',    # S12d
        's12cdw',  # S12cdw
        's12csd',  # S12csd
        's12sd',   # S12sd
        's12cs',   # S12cs
        's12wc',   # S12wc
        's12s',    # S12s
    ]
    
    crops = ['corn', 'soybean']
    
    created_files = []
    
    for code in codes:
        for crop in crops:
            filename = f'{code}_24_{crop}.yaml'
            filepath = os.path.join(conf_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(filepath):
                print(f'Skipping {filename} (already exists)')
                continue
            
            config = generate_yaml(code, crop)
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            created_files.append(filename)
            print(f'Created {filename}')
    
    if created_files:
        print(f'\nSuccessfully created {len(created_files)} YAML files')
    else:
        print('No new files created (all already exist)')

if __name__ == '__main__':
    main()
