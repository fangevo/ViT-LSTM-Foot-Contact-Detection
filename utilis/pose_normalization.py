import numpy as np
import os

# Define input and output paths
input_dir = r"D:\IPP project\data_long_jump\corrected_2d_poses"  # Path of non-normalized joint point files
output_dir = r"D:\IPP project\data_long_jump\normalized_2d_poses" # Path to save normalized joint files

# Create output folder (if it doesn't exist)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process files from 01 to 26
for file_idx in range(1, 27):
    # Construct input filename
    input_file = os.path.join(input_dir, f"{file_idx:02d}_2D_annotation.npy")
    
    # Load keypoint data
    keypoints = np.load(input_file)
    print(f"Processing file: {input_file}, shape: {keypoints.shape}")
    
    # Get the number of frames and joints
    num_frames, num_joints, _ = keypoints.shape
    
    # Normalization processing
    normalized_keypoints = np.zeros_like(keypoints)  # Create an array of the same shape to store normalized results
    for frame in range(num_frames):
        # Get the hip joint coordinates (index 0) for the current frame
        hip_coords = keypoints[frame, 0, :]  # Shape is (2,), i.e., (x_0, y_0)
        
        # Normalize all joints in the current frame
        for joint in range(num_joints):
            normalized_keypoints[frame, joint, :] = keypoints[frame, joint, :] - hip_coords
    
    # Construct output filename
    output_file = os.path.join(output_dir, f"{file_idx:02d}_2D_annotation_normalized.npy")
    
    # Save the normalized data
    np.save(output_file, normalized_keypoints)
    print(f"Saved normalized file: {output_file}")

print("Normalization completed for all files!")