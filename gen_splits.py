import os
import random
import sys

def find_common_tif(root_folder):
    """
    Finds all .npy files that are present (by filename) in every subfolder of the root folder.
    """
    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    if not subfolders:
        print(f"No subfolders found in {root_folder}.")
        return []
    
    # Collect sets of .npy filenames from each subfolder
    file_sets = []
    for sub in subfolders:
        sub_path = os.path.join(root_folder, sub)
        tifs = {f for f in os.listdir(sub_path) if f.lower().endswith('.npy')}
        file_sets.append(tifs)
    
    # Find the intersection of all sets
    if not file_sets:
        return []
    common = set.intersection(*file_sets)
    
    return sorted(list(common))

def split_and_save(common_files, output_dir):
    """
    Strips the .npy extension from common files, shuffles them, splits into train/val/test (0.7/0.2/0.1),
    and saves each list to a line-separated text file in the output directory.
    """
    if not common_files:
        print("No common files to split.")
        return
    
    # Strip .npy extensions to get stems
    stems = [os.path.splitext(f)[0] for f in common_files]
    files_2019 = [s for s in stems if '2019' in s]
    files_2020 = [s for s in stems if '2020' in s]
    files_2021 = [s for s in stems if '2021' in s]
    files_2022 = [s for s in stems if '2022' in s]
    files_2023 = [s for s in stems if '2023' in s]
    # Shuffle the list for random splitting
    random.shuffle(stems)
    
    # Separate Corn and Soybean files
    corn_files = [s for s in stems if 'corn' in s.lower()]
    val_corn_size = int(0.3 * len(corn_files))
    soybean_files = [s for s in stems if 'soybean' in s.lower()]
    val_soybean_size = int(0.3 * len(soybean_files))
    # Calculate split sizes
    total = len(stems)
    train_size = int(0.7 * total)
    val_size = int(0.3 * total)
    # Split the lists
    train = stems[:train_size]
    val = stems[train_size:train_size + val_size]
    test = stems[train_size:train_size + val_size]
    val_corn = corn_files[:val_corn_size]
    train_corn = [f for f in stems if f not in val_corn]
    val_soybean = soybean_files[:val_soybean_size]
    train_soybean = [f for f in stems if f not in val_soybean]
    # Year-based splits
    train_2019 = [f for f in stems if f not in files_2019]
    train_2020 = [f for f in stems if f not in files_2020]
    train_2021 = [f for f in stems if f not in files_2021]
    train_2022 = [f for f in stems if f not in files_2022]
    train_2023 = [f for f in stems if f not in files_2023]
    test_2021_corn = [f for f in files_2021 if 'corn' in f.lower()]
    train_corn_2021 = [f for f in stems if  f not in test_2021_corn]
    test_2021_soybean = [f for f in files_2021 if 'soybean' in f.lower()]
    train_soybean_2021 = [f for f in stems if f not in test_2021_soybean]
    # Function to save list to file
    def save_list(lst, filename):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), 'w') as file:
            file.write('\n'.join(lst) + '\n')
    
    # Save the files
    # save_list(train, 'train.txt')
    # save_list(val, 'val.txt')
    # save_list(test, 'test.txt')
    save_list(train_corn, 'train_corn.txt')
    save_list(train_soybean, 'train_soybean.txt')
    save_list(val_corn, 'val_corn.txt')
    save_list(val_soybean, 'val_soybean.txt')
    save_list(val_corn, 'test_corn.txt')
    save_list(val_soybean, 'test_soybean.txt')
    print(f"{output_dir}: Files saved: train_corn.txt ({len(train_corn)}), val_corn.txt ({len(val_corn)}), test_corn.txt ({len(val_corn)})")
    print(f"{output_dir}: Files saved: train_soybean.txt ({len(train_soybean)}), val_soybean.txt ({len(val_soybean)}), test_soybean.txt ({len(val_soybean)})")
    # save_list(train_2019, 'train_2019.txt')
    # save_list(train_2020, 'train_2020.txt')
    # save_list(train_2021, 'train_2021.txt')
    # save_list(train_2022, 'train_2022.txt')
    # save_list(train_2023, 'train_2023.txt')
    # save_list(files_2019, 'val_2019.txt')
    # save_list(files_2020, 'val_2020.txt')
    # save_list(files_2021, 'val_2021.txt')
    # save_list(files_2022, 'val_2022.txt')
    # save_list(files_2023, 'val_2023.txt')
    # print(f"{output_dir}: Files saved: train.txt ({len(train)}), val.txt ({len(val)}), test.txt ({len(test)})")
    save_list(test_2021_corn, 'test_2021_corn.txt')
    save_list(test_2021_soybean, 'test_2021_soybean.txt')
    save_list(train_corn_2021, 'train_corn_2021.txt')
    save_list(train_soybean_2021, 'train_soybean_2021.txt')
    print(f"{output_dir}: Files saved: test_2021_corn.txt ({len(test_2021_corn)}), test_2021_soybean.txt ({len(test_2021_soybean)})")
    print(f"{output_dir}: Files saved: train_corn_2021.txt ({len(train_corn_2021)}), train_soybean_2021.txt ({len(train_soybean_2021)})")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility

    # Find all processed_data_biweekly_* folders in the current directory
    base_dir = '.'
    biweekly_folders = [f for f in os.listdir(base_dir) if f.startswith('processed_data_biweekly') and os.path.isdir(os.path.join(base_dir, f))]

    if not biweekly_folders:
        print("No processed_data_biweekly* folders found.")
        sys.exit(1)

    for folder in sorted(biweekly_folders):
        root_folder = os.path.join(base_dir, folder)
        output_dir = root_folder
        print(f"Processing {root_folder} ...")
        common_files = find_common_tif(root_folder)
        split_and_save(common_files, output_dir)