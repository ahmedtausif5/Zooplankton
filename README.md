# Hierarchical Zooplankton Classification

## Project Overview
The primary goal of my project is to develop a robust, hierarchy-aware machine learning pipeline for the automated classification of zooplankton images. While standard flat classifiers have been previously implemented for this dataset by my collaborator, Dr. Sofia, this project explores a multi-head hierarchical neural network. By structuring the model this way, the goal is to capture the natural ecological relationships in the data, learning shared visual features at broader taxonomic levels before predicting specific variations.

## Dataset Details
The dataset initially consisted of large mosaic images containing multiple specimens and artifacts. Using a preprocessing notebook provided by Dr. Sofia, I extracted fine cut-outs of individual subjects. The resulting dataset consists of these cropped images, categorized into a uniquely unbalanced three-level hierarchical taxonomy:

* **Level 0 (Biological vs. Non-Biological):** Binary classification indicating if the subject is Zooplankton (Yes) or an artifact (No).
* **Level 1 (Order/Broad Category):** Includes classes such as Cladocera, Copepoda, Rotifer, Bubble, Exoskeleton, and Fiber.
* **Level 2 (Family/Genus/Life Stage):** Includes specific classes such as Bosminidae, Daphnia, Nauplius, Cyclopoid, Harpacticoid, and Calanoid.

A key characteristic of this dataset is its unbalanced structure. Several Level 1 classes (specifically Rotifer, Bubble, Exoskeleton, and Fiber) are terminal nodes and do not possess Level 2 children. To handle this mathematically, the PyTorch Dataset class assigns a `-1` integer value to represent "Not Applicable" for missing deeper levels. 

The data is rigorously cleaned and split into a 70% Training, 15% Validation, and 15% Testing distribution. Images are preprocessed and resized to 224x224 pixels.

## Methodology and Architecture

### 1. Model Architecture
The pipeline utilizes a pre-trained Vision Transformer (ViT-B/16) as a core feature extractor. The standard classification head is replaced with three independent, parallel linear heads corresponding to Level 0, Level 1, and Level 2 of the taxonomy.

### 2. Custom Loss Function
To train the multi-head architecture, I engineered a custom multi-task Cross-Entropy loss function. It calculates the loss for each hierarchical level independently but incorporates an `ignore_index=-1` mechanism. This ensures the model safely skips gradient penalties for terminal nodes, allowing specialized layers to learn without being skewed by missing data.

### 3. Strict Top-Down Inference
To completely prevent biological contradictions during evaluation, a cascading logit-masking inference function enforces strict top-down taxonomy rules. The model predicts the parent node first, and mathematically masks (sets to negative infinity) any child node logits that do not biologically belong to that predicted parent.

## Current Project Status
* **Data Engineering:** Completed. Hierarchical parsing and DataLoader classes are fully functional.
* **Prototyping:** Currently conducting Phase 1 training using linear probing. The ViT backbone is frozen to allow rapid iteration and debugging on the classification heads and masking logic.
* **Upcoming Steps:** Full model fine-tuning (unfreezing the backbone), rigorous evaluation using Strict Path Accuracy, and deployment via a Streamlit web application.
