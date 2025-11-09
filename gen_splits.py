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
    soybean_files = [s for s in stems if 'soybean' in s.lower()]
    # Calculate split sizes
    total = len(stems)
    train_size = int(0.7 * total)
    val_size = int(0.3 * total)
    # Split Corn files: 180 for val, 180 for test, rest for train
    corn_val = corn_files[:180]
    soybean_val = soybean_files[:180]
    # Split the lists
    train = stems[:train_size]
    val = stems[train_size:train_size + val_size]
    test = stems[train_size:train_size + val_size]
    train_corn = [f for f in stems if f not in corn_val]
    train_soybean = [f for f in stems if f not in soybean_val]
    train_2019 = [f for f in stems if f not in files_2019]
    train_2020 = [f for f in stems if f not in files_2020]
    train_2021 = [f for f in stems if f not in files_2021]
    train_2022 = [f for f in stems if f not in files_2022]
    train_2023 = [f for f in stems if f not in files_2023]
    # Function to save list to file
    def save_list(lst, filename):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), 'w') as file:
            file.write('\n'.join(lst) + '\n')
    
    # Save the files
    save_list(train, 'train.txt')
    save_list(val, 'val.txt')
    save_list(test, 'test.txt')
    save_list(train_corn, 'train_corn.txt')
    save_list(train_soybean, 'train_soybean.txt')
    save_list(corn_val, 'val_corn.txt')
    save_list(soybean_val, 'val_soybean.txt')
    save_list(corn_val, 'test_corn.txt')
    save_list(soybean_val, 'test_soybean.txt')
    save_list(train_2019, 'train_2019.txt')
    save_list(train_2020, 'train_2020.txt')
    save_list(train_2021, 'train_2021.txt')
    save_list(train_2022, 'train_2022.txt')
    save_list(train_2023, 'train_2023.txt')
    save_list(files_2019, 'val_2019.txt')
    save_list(files_2020, 'val_2020.txt')
    save_list(files_2021, 'val_2021.txt')
    save_list(files_2022, 'val_2022.txt')
    save_list(files_2023, 'val_2023.txt')
    print(f"{output_dir}: Files saved: train.txt ({len(train)}), val.txt ({len(val)}), test.txt ({len(test)})")

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