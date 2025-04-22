# ViT-LSTM-Foot-Contact-Detection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
The project proposes a hybrid Vision Transformer (ViT) and Bidirectional LSTM (BiLSTM) model with an attention-based fusion mechanism to accurately classify the degree of foot-ground contact during the long jump, using video captured at only 25 frames per second.

**Key Achievement:** Achieved **91.87%** classification accuracy and **8.18 ms/frame** processing speed on a resource-constrained GPU (8G VRAM, 321 TOPS).

## Features

* **Low-Frame-Rate Analysis:** Designed to work effectively with videos captured at standard frame rates (e.g., 25 fps), overcoming motion blur issues.
* **Fine-Grained Classification:** Classifies foot contact into 5 distinct labels (0: No Contact, 1-3: Progressive Ground Contact Stages, 4: Sandpit Contact), offering more detail than binary classification.
* **Hybrid ViT-LSTM Architecture:** Combines the spatial feature extraction power of Vision Transformers with the temporal modeling capabilities of LSTMs.
* **Attention-Based Fusion:** Intelligently fuses visual features (from cropped ankle images) and 2D pose data using an attention mechanism to focus on relevant information.
* **Efficient Processing:** Optimized for performance, achieving fast processing speeds even on hardware with limited computational resources.
* **Robust Training Strategy:** Employs pose normalization, data augmentation, 5-fold cross-validation, and weighted cross-entropy loss to handle data limitations and class imbalance.

Dataset (Frame sequences extracted from 30 video clips): https://drive.google.com/file/d/13hf_kXzegg2eVV8V31Rg6dn1gqT6wMtb/view?usp=sharing      Put the data folder in ./

Weight: https://drive.google.com/file/d/1fAFRAi2CZWLprRo158a0964dfXdIXCnX/view?usp=drive_link    Put it in ./weight/
