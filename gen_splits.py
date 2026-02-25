# import os
# import random
# import sys
# import numpy as np
# from multiprocessing import Pool, cpu_count

# def check_file_has_zeros(args):
#     """
#     Helper function to check if a file has more than 80% zeros in any timepoint.
#     Returns (filename, has_zeros, message) tuple.
#     """
#     filename, root_folder, dynamic_subfolders = args
#     for sub in dynamic_subfolders:
#         file_path = os.path.join(root_folder, sub, filename)
#         try:
#             data = np.load(file_path)  # Shape: (T, C, H, W)
#             # Check if any timepoint has more than 80% zeros
#             for t in range(data.shape[0]):
#                 total_elements = data[t].size
#                 zero_count = np.sum(data[t] == 0)
#                 zero_percentage = (zero_count / total_elements) * 100
#                 if zero_percentage > 80:
#                     return (filename, True, f"Removing {filename}: {zero_percentage:.1f}% zeros found in {sub} at timepoint {t}")
#         except Exception as e:
#             return (filename, True, f"Warning: Could not load {file_path}: {e}")
#     return (filename, False, None)

# def find_common_tif(root_folder):
#     """
#     Finds all .npy files that are present (by filename) in every subfolder of the root folder.
#     Removes files from the common set if any subfolder's version of that file contains all zeros.
#     """
#     subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
#     if not subfolders:
#         print(f"No subfolders found in {root_folder}.")
#         return []
    
#     # Collect sets of .npy filenames from each subfolder
#     file_sets = []
#     for sub in subfolders:
#         sub_path = os.path.join(root_folder, sub)
#         tifs = {f for f in os.listdir(sub_path) if f.lower().endswith('.npy')}
#         file_sets.append(tifs)
    
#     # Find the intersection of all sets
#     if not file_sets:
#         return []
#     common = set.intersection(*file_sets)
    
#     # # Filter out files that have all zeros in any timepoint in any subfolder
#     # filtered_common = []
#     # dynamic_subfolders = ['S1GRD', 'S2L2A', 'WEATHER']  # To avoid linting issues
    
#     # # Prepare arguments for parallel processing
#     # check_args = [(filename, root_folder, dynamic_subfolders) for filename in common]
    
#     # # Use parallel processing to check files
#     # num_processes = min(cpu_count(), len(check_args))
#     # with Pool(processes=num_processes) as pool:
#     #     results = pool.map(check_file_has_zeros, check_args)
    
#     # # Process results
#     # for filename, has_zeros, message in results:
#     #     if message:
#     #         print(message)
#     #     if not has_zeros:
#     #         filtered_common.append(filename)
    
#     return sorted(common)

# def split_and_save(common_files, output_dir):
#     """
#     Strips the .npy extension from common files, shuffles them, splits into train/val/test (0.7/0.2/0.1),
#     and saves each list to a line-separated text file in the output directory.
#     """
#     if not common_files:
#         print("No common files to split.")
#         return
    
#     # Strip .npy extensions to get stems
#     stems = [os.path.splitext(f)[0] for f in common_files]
#     files_2017 = [s for s in stems if '2017' in s]
#     files_2018 = [s for s in stems if '2018' in s]
#     files_2019 = [s for s in stems if '2019' in s]
#     files_2020 = [s for s in stems if '2020' in s]
#     files_2021 = [s for s in stems if '2021' in s]
#     files_2022 = [s for s in stems if '2022' in s]
#     files_2023 = [s for s in stems if '2023' in s]
#     files_2024 = [s for s in stems if '2024' in s]
#     # Shuffle the list for random splitting
#     random.shuffle(stems)
    
#     # Separate Corn and Soybean files
#     corn_files = [s for s in stems if 'corn' in s.lower()]
#     val_corn_size = int(0.3 * len(corn_files))
#     soybean_files = [s for s in stems if 'soybean' in s.lower()]
#     val_soybean_size = int(0.3 * len(soybean_files))
#     # Calculate split sizes
#     total = len(stems)
#     train_size = int(0.7 * total)
#     val_size = int(0.3 * total)
#     # test_size = total - train_size - val_size
#     # Split the lists
#     train = stems[:train_size]
#     val = stems[train_size:train_size + val_size]
#     # test = stems[train_size + val_size:train_size + val_size + test_size]
#     val_corn = [f for f in val if 'corn' in f.lower()]
#     # test_corn = [f for f in test if 'corn' in f.lower()]
#     train_corn = [f for f in train if 'corn' in f.lower()]
#     val_soybean = [f for f in val if 'soybean' in f.lower()]
#     # test_soybean = [f for f in test if 'soybean' in f.lower()]
#     train_soybean = [f for f in train if 'soybean' in f.lower()]
#     # Year-based splits
#     train_2017 = [f for f in stems if f not in files_2017]
#     train_2018 = [f for f in stems if f not in files_2018]
#     train_2019 = [f for f in stems if f not in files_2019]
#     train_2020 = [f for f in stems if f not in files_2020]
#     train_2021 = [f for f in stems if f not in files_2021]
#     train_2022 = [f for f in stems if f not in files_2022]
#     train_2023 = [f for f in stems if f not in files_2023]
#     train_2024 = [f for f in stems if f not in files_2024]
#     test_2021_corn = [f for f in files_2021 if 'corn' in f.lower()]
#     train_corn_2021 = [f for f in stems if  f not in test_2021_corn]
#     test_2021_soybean = [f for f in files_2021 if 'soybean' in f.lower()]
#     train_soybean_2021 = [f for f in stems if f not in test_2021_soybean]
#     # Function to save list to file
#     def save_list(lst, filename):
#         os.makedirs(output_dir, exist_ok=True)
#         with open(os.path.join(output_dir, filename), 'w') as file:
#             file.write('\n'.join(lst) + '\n')
    
#     # Save the files
#     save_list(train, 'train.txt')
#     save_list(val, 'val.txt')
#     save_list(val, 'test.txt')
#     print(f"{output_dir}: Files saved: train.txt ({len(train)}), val.txt ({len(val)}), test.txt ({len(val)})")
#     save_list(train_corn, 'train_corn.txt')
#     save_list(train_soybean, 'train_soybean.txt')
#     save_list(val_corn, 'val_corn.txt')
#     save_list(val_soybean, 'val_soybean.txt')
#     save_list(val_corn, 'test_corn.txt')
#     save_list(val_soybean, 'test_soybean.txt')
#     print(f"{output_dir}: Files saved: train_corn.txt ({len(train_corn)}), val_corn.txt ({len(val_corn)}), test_corn.txt ({len(val_corn)})")
#     print(f"{output_dir}: Files saved: train_soybean.txt ({len(train_soybean)}), val_soybean.txt ({len(val_soybean)}), test_soybean.txt ({len(val_soybean)})")
#     save_list(train_2017, 'train_2017.txt')
#     save_list(train_2018, 'train_2018.txt')
#     save_list(train_2019, 'train_2019.txt')
#     save_list(train_2020, 'train_2020.txt')
#     save_list(train_2021, 'train_2021.txt')
#     save_list(train_2022, 'train_2022.txt')
#     save_list(train_2023, 'train_2023.txt')
#     save_list(train_2024, 'train_2024.txt')
#     save_list(files_2017, 'val_2017.txt')
#     save_list(files_2018, 'val_2018.txt')
#     save_list(files_2019, 'val_2019.txt')
#     save_list(files_2020, 'val_2020.txt')
#     save_list(files_2021, 'val_2021.txt')
#     save_list(files_2022, 'val_2022.txt')
#     save_list(files_2023, 'val_2023.txt')
#     save_list(files_2024, 'val_2024.txt')
#     # print(f"{output_dir}: Files saved: train.txt ({len(train)}), val.txt ({len(val)}), test.txt ({len(test)})")
#     # save_list(test_2021_corn, 'test_2021_corn.txt')
#     # save_list(test_2021_soybean, 'test_2021_soybean.txt')
#     # save_list(train_corn_2021, 'train_corn_2021.txt')
#     # save_list(train_soybean_2021, 'train_soybean_2021.txt')
#     # print(f"{output_dir}: Files saved: test_2021_corn.txt ({len(test_2021_corn)}), test_2021_soybean.txt ({len(test_2021_soybean)})")
#     # print(f"{output_dir}: Files saved: train_corn_2021.txt ({len(train_corn_2021)}), train_soybean_2021.txt ({len(train_soybean_2021)})")

# if __name__ == "__main__":
#     random.seed(42)  # For reproducibility

#     # Find all processed_data_biweekly_* folders in the current directory
#     base_dir = '.'
#     biweekly_folders = [f for f in os.listdir(base_dir) if f.startswith('processed_data_weekly') and os.path.isdir(os.path.join(base_dir, f))]

#     if not biweekly_folders:
#         print("No processed_data_biweekly* folders found.")
#         sys.exit(1)

#     for folder in sorted(biweekly_folders):
#         root_folder = os.path.join(base_dir, folder)
#         output_dir = root_folder
#         print(f"Processing {root_folder} ...")
#         common_files = find_common_tif(root_folder)
#         split_and_save(common_files, output_dir)
import os
import random
import sys
import numpy as np
from multiprocessing import Pool, cpu_count

def check_file_has_zeros(args):
    """
    Helper function to check if a file has more than 80% zeros in any timepoint.
    Returns (filename, has_zeros, message) tuple.
    """
    filename, root_folder, dynamic_subfolders = args
    for sub in dynamic_subfolders:
        file_path = os.path.join(root_folder, sub, filename)
        try:
            data = np.load(file_path)  # Shape: (T, C, H, W)
            # Check if any timepoint has more than 80% zeros
            for t in range(data.shape[0]):
                total_elements = data[t].size
                zero_count = np.sum(data[t] == 0)
                zero_percentage = (zero_count / total_elements) * 100
                if zero_percentage > 80:
                    return (filename, True, f"Removing {filename}: {zero_percentage:.1f}% zeros found in {sub} at timepoint {t}")
        except Exception as e:
            return (filename, True, f"Warning: Could not load {file_path}: {e}")
    return (filename, False, None)

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
    
    return sorted(common)

def split_and_save(common_files, output_dir):
    """
    Splits files:
    - Train: 2017-2022 (all) - 80%
    - Val = Test: 2023-2024 (same data) - 20%
    """
    if not common_files:
        print("No common files to split.")
        return
    
    # Strip .npy extensions to get stems
    stems = [os.path.splitext(f)[0] for f in common_files]
    
    # Split by year
    train_files = [s for s in stems if any(year in s for year in ['2017', '2018', '2019', '2020', '2021', '2022', '2023'])]
    val_test_pool = [s for s in stems if any(year in s for year in ['2024', '2025'])]
    
    # Split val_test_pool by crop
    val_test_corn = [f for f in val_test_pool if 'corn' in f.lower()]
    val_test_soybean = [f for f in val_test_pool if 'soybean' in f.lower()]
    
    # Val = Test (use same data!)
    val_corn = val_test_corn      # all
    test_corn = val_test_corn     # all (same)
    
    val_soybean = val_test_soybean      # all
    test_soybean = val_test_soybean     # all (same)
    
    # Combine for overall val/test
    val_files = val_corn + val_soybean
    test_files = test_corn + test_soybean  # same as val
    
    # Train by crop
    train_corn = [f for f in train_files if 'corn' in f.lower()]
    train_soybean = [f for f in train_files if 'soybean' in f.lower()]
    
    # Function to save list to file
    def save_list(lst, filename):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), 'w') as file:
            file.write('\n'.join(sorted(lst)) + '\n')
    
    # Save overall splits (ALL crops combined)
    save_list(train_files, 'train.txt')
    save_list(val_files, 'val.txt')
    save_list(test_files, 'test.txt')  # same as val
    
    # Save crop-specific splits
    save_list(train_corn, 'train_corn.txt')
    save_list(train_soybean, 'train_soybean.txt')
    save_list(val_corn, 'val_corn.txt')
    save_list(val_soybean, 'val_soybean.txt')
    save_list(test_corn, 'test_corn.txt')        # same as val_corn
    save_list(test_soybean, 'test_soybean.txt')  # same as val_soybean
    
    # Print summary
    print(f"\n{output_dir}:")
    print(f"{'='*70}")
    print(f"Overall splits (Corn + Soybean):")
    print(f"  - train.txt: {len(train_files)} files (2017-2022)")
    print(f"  - val.txt: {len(val_files)} files (2023-2024)")
    print(f"  - test.txt: {len(test_files)} files (2023-2024) [SAME as val]")
    print(f"\nCrop-specific splits:")
    print(f"  Corn:")
    print(f"    - train_corn.txt: {len(train_corn)} files (2017-2022)")
    print(f"    - val_corn.txt: {len(val_corn)} files (2023-2024)")
    print(f"    - test_corn.txt: {len(test_corn)} files (2023-2024) [SAME as val]")
    print(f"  Soybean:")
    print(f"    - train_soybean.txt: {len(train_soybean)} files (2017-2023)")
    print(f"    - val_soybean.txt: {len(val_soybean)} files (2024-2025)")
    print(f"    - test_soybean.txt: {len(test_soybean)} files (2024-2025) [SAME as val]")
    print(f"{'='*70}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility

    # Find all processed_data_fs_* folders in the current directory
    base_dir = '.'
    weekly_folders = [f for f in os.listdir(base_dir) if f.startswith('processed_data_fs') and os.path.isdir(os.path.join(base_dir, f))]

    if not weekly_folders:
        print("No processed_data_fs* folders found.")
        sys.exit(1)

    for folder in sorted(weekly_folders):
        root_folder = os.path.join(base_dir, folder)
        output_dir = root_folder
        print(f"\n{'='*70}")
        print(f"Processing {root_folder}...")
        print(f"{'='*70}")
        common_files = find_common_tif(root_folder)
        split_and_save(common_files, output_dir)