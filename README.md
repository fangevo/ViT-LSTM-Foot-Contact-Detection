# ViT-LSTM-Foot-Contact-Detection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
The project proposes a hybrid Vision Transformer (ViT) and Bidirectional LSTM (BiLSTM) model with an attention-based fusion mechanism to accurately classify the degree of foot-ground contact during the long jump, using video captured at only 25 frames per second.

**Highlights:** Achieved **91.87%** classification accuracy and **8.18 ms/frame** processing speed on a resource-constrained GPU (8G VRAM, 321 TOPS).

## Features

* **Low-Frame-Rate Analysis:** Designed to work effectively with videos captured at standard frame rates, overcoming motion blur issues.
* **Fine-Grained Classification:** Classifies foot contact into 5 distinct labels (0: No Contact, 1-3: Progressive Ground Contact Stages, 4: Sandpit Contact), offering more detail than binary classification.
* **Hybrid ViT-LSTM Architecture:** Combines the spatial feature extraction power of Vision Transformers with the temporal modeling capabilities of LSTMs.
* **Attention-Based Fusion:** Fuses visual features (from cropped ankle images) and 2D pose data using an attention mechanism to focus on relevant information.
* **Efficient Processing:** Optimized for performance, achieving fast processing speeds even on hardware with limited computational resources.
* **Robust Training Strategy:** Employs pose normalization, data augmentation, 5-fold cross-validation, and weighted cross-entropy loss to handle data limitations and class imbalance.

## Method
**Label Definition**
![wechat_2025-04-22_182935_943](https://github.com/user-attachments/assets/222e8b37-5b8d-4318-a0b0-f7a30ae5ea8d)

**Model Architecture**
![s (1)](https://github.com/user-attachments/assets/46e446a2-5b1e-46e1-ad00-76655f493146)
$B$: batch size, $T$: sequence length, $C$: number of channels, $H$: frame height, $W$: frame width, $F_p$: pose feature dimension, $F_v$: ViT feature dimension, $D_h$: hidden size (per direction), $N$: output classes.

## Installation

**Dataset** (Frame sequences extracted from 30 video clips): https://drive.google.com/file/d/13hf_kXzegg2eVV8V31Rg6dn1gqT6wMtb/view?usp=sharing      Put the data folder in ./

**Weight**: https://drive.google.com/file/d/1fAFRAi2CZWLprRo158a0964dfXdIXCnX/view?usp=drive_link    Put it in ./weight/
