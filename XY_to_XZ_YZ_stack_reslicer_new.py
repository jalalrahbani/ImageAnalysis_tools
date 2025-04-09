import os
import numpy as np
import tifffile as tiff

# Define input and output directories (update these paths accordingly)
input_dir = r"C:\Users\rahba\OneDrive\Desktop\Analysis And Development Tools- Jr\PYTHON CODING PORTFOLIO\Tiff XY stack reslicer to XZ and YZ"
output_dir = r"C:\Users\rahba\OneDrive\Desktop\Analysis And Development Tools- Jr\PYTHON CODING PORTFOLIO\Tiff XY stack reslicer to XZ and YZ\ output"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# If output directory does not exist, create a folder called "output" in input directory
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to rotate the YZ view by 90 degrees counterclockwise
def rotate_yz_view(yz_view):
    return np.rot90(yz_view, k=-1, axes=(1, 2))  # Rotate along (Z, Y) axes
    
def flip_yz_view(yz_view_flipped):
    return np.flip(yz_view_flipped, axis=2)  # Flip horizontally

# Function to reslice and rotate the YZ projection
def reslice_and_rotate_tif_stack(file_path, output_dir):
    # Load the TIF stack
    img_stack = tiff.imread(file_path)  # Shape: (Z, Y, X)

    # Get the base file name
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Generate XZ projection (reshaping: Z remains, Y is horizontal, X is vertical)
    xz_view = np.transpose(img_stack, (1, 0, 2))  # Shape: (Y, Z, X)

    # Generate YZ projection (reshaping: Z remains, X is horizontal, Y is vertical)
    yz_view = np.transpose(img_stack, (2, 0, 1))  # Shape: (X, Z, Y)

    # Rotate YZ view by 90 degrees counterclockwise
    yz_view_rotated = rotate_yz_view(yz_view)

    # Flip the rotated YZ view horizontally
    yz_view_rotated_flipped = flip_yz_view(yz_view_rotated)

    # Save the new projections
    xz_output_path = os.path.join(output_dir, f"{base_name}_XZview.tif")
    yz_output_path = os.path.join(output_dir, f"{base_name}_YZview_rotated.tif")

    tiff.imwrite(xz_output_path, xz_view.astype(np.uint16))
    tiff.imwrite(yz_output_path, yz_view_rotated.astype(np.uint16))
    tiff.imwrite(yz_output_path, yz_view_rotated_flipped.astype(np.uint16))

# Process all TIF files in the input directory
for file in os.listdir(input_dir):
    if file.lower().endswith(".tif"):
        file_path = os.path.join(input_dir, file)
        reslice_and_rotate_tif_stack(file_path, output_dir)

# Confirm processing by listing the output files
print("Processing complete. Resliced files saved in:", output_dir)
