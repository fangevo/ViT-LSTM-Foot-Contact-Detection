import os
import glob
import numpy as np
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import timm  
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold

# -------------------------------
# 1. Custom Dataset Class
# -------------------------------

class JumpSequenceDataset(Dataset):
    def __init__(self, image_folder, label_file, pose_file, transform=None):
        """
        image_folder: Folder containing video frames
        label_file: Label file for the corresponding video
        pose_file: Keypoint coordinate file for the corresponding video
        transform: Image preprocessing method
        """
        # Load image paths, labels, and pose data
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        self.labels = np.load(label_file)
        self.poses = np.load(pose_file, allow_pickle=True)  # Shape: (seq_len, 19, 2)

        # Ensure the number of images, labels, and poses are consistent by taking the minimum length
        min_len = min(len(self.image_paths), len(self.labels), len(self.poses))
        self.image_paths = self.image_paths[:min_len]
        self.labels = self.labels[:min_len]
        self.poses = self.poses[:min_len]

        self.transform = transform

    def __len__(self):
        return 1  # Each sample is a complete video sequence

    def __getitem__(self, index):
        frames = []
        for img_path in self.image_paths:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        frames = torch.stack(frames, dim=0)  # (seq_len, C, H, W)
        labels = torch.tensor(self.labels, dtype=torch.long)  # (seq_len,)
        poses = torch.tensor(self.poses, dtype=torch.float32)  # (seq_len, 19, 2)
        poses = poses.view(poses.size(0), -1)  # Flatten to (seq_len, 38)
        return frames, poses, labels

# -------------------------------
# 2. Data Preprocessing and Loading
# -------------------------------
def pad_collate_fn(batch):
    frames_list, poses_list, labels_list = zip(*batch)
    max_seq_len = max(frames.shape[0] for frames in frames_list)
    
    padded_frames = []
    padded_poses = []
    padded_labels = []
    for frames, poses, labels in zip(frames_list, poses_list, labels_list):
        seq_len = frames.shape[0]
        if seq_len < max_seq_len:
            pad_frames = torch.zeros((max_seq_len - seq_len, *frames.shape[1:]), dtype=frames.dtype)
            frames = torch.cat([frames, pad_frames], dim=0)
            
            pad_poses = torch.zeros((max_seq_len - seq_len, poses.shape[1]), dtype=poses.dtype)
            poses = torch.cat([poses, pad_poses], dim=0)
            
            pad_labels = torch.full((max_seq_len - seq_len,), -100, dtype=labels.dtype)
            labels = torch.cat([labels, pad_labels], dim=0)
        padded_frames.append(frames)
        padded_poses.append(poses)
        padded_labels.append(labels)
        
    padded_frames = torch.stack(padded_frames, dim=0)  # (batch, seq_len, C, H, W)
    padded_poses = torch.stack(padded_poses, dim=0)    # (batch, seq_len, 38)
    padded_labels = torch.stack(padded_labels, dim=0)  # (batch, seq_len)
    return padded_frames, padded_poses, padded_labels

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Keep preprocessing simple for the validation set
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset Split
all_video_dirs = [os.path.join("data", "train_val", f"{i:02d}") for i in range(1, 26)]
all_label_files = [os.path.join("data", "label", f"{i:02d}.npy") for i in range(1, 26)]
all_pose_files = [os.path.join("data", "normalized_2d_poses", f"{i:02d}_2D_annotation_normalized.npy") for i in range(1, 26)]

# Create a list of all datasets
all_dataset_list = []
for img_dir, label_file, pose_file in zip(all_video_dirs, all_label_files, all_pose_files):
    all_dataset_list.append(JumpSequenceDataset(img_dir, label_file, pose_file, transform=train_transform))

# -------------------------------
# 3. Model Definition
# -------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid() 
        )
        self.norm = nn.LayerNorm(feature_dim)  # Added layer normalization

    def forward(self, vit_features, pose_features):
        combined = torch.cat((vit_features, pose_features), dim=-1)
        weights = self.attention(combined)  # (batch, seq_len, 1)
        fused = vit_features * weights + pose_features * (1 - weights)
        fused = fused + vit_features  # Residual connection
        fused = self.norm(fused)
        return fused, weights  # Return weights for visualization

class ViT_LSTM_with_Pose(nn.Module):
    def __init__(self, num_classes=5, hidden_size=256, num_layers=2, dropout=0.2, pose_dim=38, vit_feature_dim=768):
        super(ViT_LSTM_with_Pose, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
        for param in self.vit.parameters():
            param.requires_grad = False
        for block in self.vit.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.vit.norm.parameters():
            param.requires_grad = True
        
        self.pose_proj = nn.Linear(pose_dim, vit_feature_dim)
        self.fusion = AttentionFusion(vit_feature_dim)
        self.lstm = nn.LSTM(vit_feature_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, frames, poses):
        batch, seq_len, C, H, W = frames.shape
        frames = frames.view(batch * seq_len, C, H, W)
        vit_features = self.vit(frames)
        vit_features = vit_features.view(batch, seq_len, -1)
        poses = self.pose_proj(poses)
        combined_features, attention_weights = self.fusion(vit_features, poses)
        lstm_out, _ = self.lstm(combined_features)
        lstm_out = self.layer_norm(lstm_out)
        out = self.classifier(lstm_out)
        return out, attention_weights  # Return attention weights for visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT_LSTM_with_Pose(num_classes=5, hidden_size=256, num_layers=2, dropout=0.2, pose_dim=38, vit_feature_dim=768)
model.to(device)

# -------------------------------
# 4. Training
# -------------------------------

k_folds = 5  # Set the number of K-folds
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)  # Shuffle the data randomly

# Store performance metrics for each fold
fold_train_losses = []
fold_val_losses = []
fold_accuracies = []

def save_epoch_plots(train_losses, val_losses, accuracies, current_epoch, fold):
    epochs = list(range(1, len(train_losses) + 1))

    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs, val_losses, label="Validation Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train and Validation Loss up to Epoch {current_epoch} Fold {fold+1}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_epoch_{current_epoch}_fold_{fold+1}.png")
    plt.close()

    # Plot accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, label="Validation Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy up to Epoch {current_epoch} Fold {fold+1}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"accuracy_epoch_{current_epoch}_fold_{fold+1}.png")
    plt.close()

    print(f"Plots saved for Epoch {current_epoch} Fold {fold+1}")
    
def train():
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_dataset_list)):
        print(f"\nFold {fold + 1}/{k_folds}")
        
        # Split training and validation datasets based on indices
        train_dataset = ConcatDataset([all_dataset_list[i] for i in train_idx])
        val_dataset = ConcatDataset([all_dataset_list[i] for i in val_idx])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=pad_collate_fn)
        
        # Initialize the model
        model = ViT_LSTM_with_Pose(num_classes=5, hidden_size=256, num_layers=2, dropout=0.2, pose_dim=38, vit_feature_dim=768)
        model.to(device)
        
        # Compute class weights
        all_labels = []
        for _, _, labels in train_dataset:
            all_labels.extend(labels.numpy())
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        class_weights = torch.tensor(class_weights, device=device, dtype=torch.float)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training and validation
        num_epochs = 40
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}")
            for sequences, poses, label_seq in progress_bar:
                sequences = sequences.to(device)
                poses = poses.to(device)
                label_seq = label_seq.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(sequences, poses)
                outputs = outputs.view(-1, outputs.size(-1))
                labels = label_seq.view(-1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                running_loss += loss.item() * sequences.size(0)
                progress_bar.set_postfix(loss=loss.item())
            
            epoch_train_loss = running_loss / len(train_dataset)
            train_losses.append(epoch_train_loss)
            print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            total = 0
            correct = 0
            with torch.no_grad():
                for sequences, poses, label_seq in val_loader:
                    sequences = sequences.to(device)
                    poses = poses.to(device)
                    label_seq = label_seq.to(device)
                    outputs, _ = model(sequences, poses)
                    outputs = outputs.view(-1, outputs.size(-1))
                    labels = label_seq.view(-1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * sequences.size(0)
                    
                    valid_mask = (labels != -100)
                    if valid_mask.sum().item() > 0:
                        _, preds = torch.max(outputs, 1)
                        correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
                        total += valid_mask.sum().item()
            
            epoch_val_loss = val_loss / len(val_dataset)
            accuracy = correct / total if total > 0 else 0
            val_losses.append(epoch_val_loss)
            accuracies.append(accuracy)
            print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            scheduler.step(epoch_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Fold {fold+1}, Epoch {epoch+1}, Current Learning Rate: {current_lr:.8f}")
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_accuracy = accuracy
                best_train_losses = train_losses[-1]
                torch.save(model.state_dict(), f"best_vit_lstm_classifier_fold_{fold+1}.pth")
                print(f"Fold {fold+1}: Best model saved.")
        
        # Save metrics for the current fold
        fold_train_losses.append(best_train_losses)
        fold_val_losses.append(best_val_loss)
        fold_accuracies.append(best_accuracy)
        
        # Save training and validation curves for the current fold
        save_epoch_plots(train_losses, val_losses, accuracies, num_epochs, fold)
    
    # Compute average performance across all folds
    avg_train_loss = np.mean(fold_train_losses)
    avg_val_loss = np.mean(fold_val_losses)
    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nK-Fold Cross-Validation Results:")
    print(f"Average Train Loss: {avg_train_loss:.4f}")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

# -------------------------------
# 5. Prediction
# -------------------------------
def predict_sequence_all_frames(image_folder, pose_file, model, transform, device, output_file):
    model.eval()
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    poses = np.load(pose_file, allow_pickle=True)  # (seq_len, 19, 2)
    
    frames = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        frames.append(image)
    
    if not frames:
        raise ValueError("No images found in the folder.")
    
    frames_tensor = torch.stack(frames, dim=0).unsqueeze(0).to(device)  # (1, seq_len, C, H, W)
    poses_tensor = torch.tensor(poses, dtype=torch.float32).view(1, -1, 38).to(device)  # (1, seq_len, 38)
    
    with torch.no_grad():
        outputs,_ = model(frames_tensor, poses_tensor)  # (1, seq_len, num_classes)
        outputs = outputs.squeeze(0)  # (seq_len, num_classes)
        predictions = outputs.argmax(dim=1).cpu().numpy()  # (seq_len,)
    
    np.save(output_file, predictions)
    print(f"Predicted labels have been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict with ViT-LSTM model")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                        help="Mode to run: 'train' or 'predict'")
    args = parser.parse_args()

    if args.mode == "train":
        train()  

    elif args.mode == "predict":
        test_root = "data/test"
        test_video_dirs = sorted(glob.glob(os.path.join(test_root, "*")))
        model.load_state_dict(torch.load("weight/best_vit_lstm_classifier.pth", map_location=device))
        for video_dir in test_video_dirs:
            video_name = os.path.basename(video_dir) 
            output_file = f"output/predicted_labels_{video_name}.npy"
            predict_sequence_all_frames(
                image_folder=video_dir,
                pose_file=f"data/normalized_2d_poses/{video_name}_2D_annotation_normalized.npy",
                model=model,
                transform=val_transform,
                device=device,
                output_file=output_file
            )

        print("Prediction completed for all test videos.")
