import os
import random
import sys
def find_common_tif(root_folder):
    """
    Finds all .tif files that are present (by filename) in every subfolder of the root folder.
    """
    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    if not subfolders:
        print("No subfolders found.")
        return []
    
    # Collect sets of .tif filenames from each subfolder
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
    Strips the .tif extension from common files, shuffles them, splits into train/val/test (0.7/0.2/0.1),
    and saves each list to a line-separated text file in the output directory.
    """
    if not common_files:
        print("No common files to split.")
        return
    
    # Strip .tif extensions to get stems
    stems = [os.path.splitext(f)[0] for f in common_files]
    
    # Shuffle the list for random splitting
    random.shuffle(stems)
    
    # Calculate split sizes
    total = len(stems)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size  # Ensures exact split, handling rounding
    
    # Split the lists
    train = stems[:train_size]
    val = stems[train_size:train_size + val_size]
    test = stems[train_size + val_size:]
    
    # Function to save list to file
    def save_list(lst, filename):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), 'w') as file:
            file.write('\n'.join(lst) + '\n')
    
    # Save the files
    save_list(train, 'train.txt')
    save_list(val, 'val.txt')
    save_list(test, 'test.txt')
    
    print(f"Files saved: train.txt ({len(train)}), val.txt ({len(val)}), test.txt ({len(test)})")

if __name__ == "__main__":
    root_folder = f'./processed_data_monthly'
    output_dir = root_folder
    common_files = find_common_tif(root_folder)
    split_and_save(common_files, output_dir)