import os

# Get all directories in the current directory
all_dirs = {d for d in os.listdir('./processed_s2') if os.path.isdir(d)}
print(all_dirs)

# Get directories that contain at least one file starting with "WGS84"
dirs_with_wgs84 = set()

for d in all_dirs:
    for f in os.listdir(d):
        if f.startswith("WGS84"):
            dirs_with_wgs84.add(d)
            break  # No need to check further in this directory
        
print("Number of directories with WGS84 files:", len(dirs_with_wgs84))

# Get directories that do NOT contain any "WGS84" file
dirs_without_wgs84 = all_dirs - dirs_with_wgs84

# Print the result
for d in sorted(dirs_without_wgs84):
    print(d)