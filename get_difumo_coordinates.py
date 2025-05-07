import numpy as np
import pandas as pd
from nilearn import datasets
from scipy import ndimage
import nibabel as nib
from pathlib import Path

# Load the DiFuMo atlas
print("Loading DiFuMo atlas...")
atlas_data = datasets.fetch_atlas_difumo(dimension=64, resolution_mm=2)
atlas_img = nib.load(atlas_data.maps)
atlas_maps = atlas_img.get_fdata()
affine = atlas_img.affine

# Prepare to store coordinates
mni_coords = []

# Process each ROI
print(f"Calculating centers of mass for {atlas_maps.shape[3]} ROIs...")
for roi_index in range(atlas_maps.shape[3]):
    roi_map = atlas_maps[..., roi_index]
    
    # Use absolute values for center of mass calculation
    abs_roi_map = np.abs(roi_map)
    
    # Calculate center of mass (returns voxel coordinates)
    if abs_roi_map.sum() > 0:  # Check if the ROI has non-zero values
        voxel_coords = ndimage.center_of_mass(abs_roi_map)
        
        # Convert voxel coordinates to MNI coordinates
        mni_coord = nib.affines.apply_affine(affine, voxel_coords)
        mni_coords.append(mni_coord)
    else:
        print(f"  Warning: ROI {roi_index+1} has no non-zero values")
        mni_coords.append(np.array([0, 0, 0]))  # Default to origin if empty

# Load the existing CSV file
csv_path = Path("labels_64_dictionary.csv")
df = pd.read_csv(csv_path)

# Add MNI coordinate columns
df['x_mni'] = [coord[0] for coord in mni_coords]
df['y_mni'] = [coord[1] for coord in mni_coords]
df['z_mni'] = [coord[2] for coord in mni_coords]

# Save the updated CSV
print("Saving updated CSV with MNI coordinates...")
df.to_csv("labels_64_dictionary_with_coords.csv", index=False)
print("Done! Added MNI coordinates to the CSV file.")
