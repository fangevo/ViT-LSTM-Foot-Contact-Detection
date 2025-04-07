import cv2
import numpy as np
import os

# Define ankle indices
ankle_idx_3 = 3  # Ankle at index 3
ankle_idx_7 = 7  # Ankle at index 7

# Input and output paths
keypoints_dir = 'data/2d_poses'
frames_base_dir = 'data_long_jump/frames'
output_base_dir = 'cropped_ankles'  # Folder to save cropped results

# Create output folder (if it doesn’t exist)
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Process 26 video segments (01 to 26)
for i in range(1, 27):  # 1 to 26
    # Construct paths for keypoints file, frames folder, and output folder
    keypoints_file = os.path.join(keypoints_dir, f'{i:02d}_2D_annotation.npy')
    frames_dir = os.path.join(frames_base_dir, f'{i:02d}')
    output_dir = os.path.join(output_base_dir, f'{i:02d}')
    
    # Create output subfolder for each video segment
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if keypoints file exists
    if not os.path.exists(keypoints_file):
        print(f"Keypoints file {keypoints_file} does not exist, skipping...")
        continue
    
    # Load keypoints data
    keypoints_data = np.load(keypoints_file, allow_pickle=True)
    num_frames = len(keypoints_data)
    print(f"Processing video segment {i:02d}, total {num_frames} frames")
    
    # Iterate through each frame
    for frame_idx in range(num_frames):
        # Read keypoints for the current frame
        keypoints = keypoints_data[frame_idx]
        
        # Read the corresponding frame image
        image_file = os.path.join(frames_dir, f'{frame_idx:05d}.png')
        if not os.path.exists(image_file):
            print(f"Image {image_file} does not exist, skipping frame {frame_idx}")
            continue
        image = cv2.imread(image_file)
        
        # Get coordinates of both ankles
        ankle_3_x, ankle_3_y = keypoints[ankle_idx_3]
        ankle_7_x, ankle_7_y = keypoints[ankle_idx_7]
        
        # Compare y-coordinates to find the lower ankle (larger y is lower)
        if ankle_3_y > ankle_7_y:
            ankle_x, ankle_y = int(ankle_3_x), int(ankle_3_y)
            ankle_idx = ankle_idx_3
        else:
            ankle_x, ankle_y = int(ankle_7_x), int(ankle_7_y)
            ankle_idx = ankle_idx_7
        
        # Define cropping area (224x224 pixels)
        crop_size = 112  # Half of 224
        x_min = max(0, ankle_x - crop_size)
        x_max = min(image.shape[1], ankle_x + crop_size)
        y_min = max(0, ankle_y - crop_size)
        y_max = min(image.shape[0], ankle_y + crop_size)
        
        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        # Pad the image if the cropped area is smaller than 224x224
        if cropped_image.shape[0] < 224 or cropped_image.shape[1] < 224:
            padded_image = np.zeros((224, 224, 3), dtype=np.uint8)
            y_offset = (224 - cropped_image.shape[0]) // 2
            x_offset = (224 - cropped_image.shape[1]) // 2
            padded_image[y_offset:y_offset+cropped_image.shape[0], 
                         x_offset:x_offset+cropped_image.shape[1]] = cropped_image
            cropped_image = padded_image
        
        # Save the cropped result
        output_file = os.path.join(output_dir, f'{frame_idx:05d}.png')
        cv2.imwrite(output_file, cropped_image)
        
        # Print progress
        print(f"Video {i:02d} frame {frame_idx:05d} processed, cropped ankle index: {ankle_idx}, coordinates: ({ankle_x}, {ankle_y})")

print("All video segments processed!")