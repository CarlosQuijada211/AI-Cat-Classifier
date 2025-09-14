" This script renames all files in a specified folder to a standardized format."

import os

# Set path to desired folder
folder = 'data_test/cleo'

# List all files in the folder
files = os.listdir(folder)

# Rename each file in the folder
for index, file in enumerate(files):
    # Get file extension
    ext = os.path.splitext(file)[1]

    # Create new file name
    new_name = f"cleo_{index}{ext}"

    # Construct full old and new file paths
    old_path = os.path.join(folder, file)
    new_path = os.path.join(folder, new_name)

    # Rename the file
    os.rename(old_path, new_path)