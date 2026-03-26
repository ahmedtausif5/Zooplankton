# Hierarchical Classification of Zooplankton Images

A deep learning project for **hierarchical image classification of zooplankton images** using a **Vision Transformer (ViT-B/16)** backbone, **multi-head prediction**, and **strict top-down hierarchical masking** during inference.

## Live Demo

Hosted Streamlit app: [https://zooplankton-classifier.streamlit.app/](https://zooplankton-classifier.streamlit.app/)

## Project Overview

Most image classification projects treat all classes as flat and independent. In this project, I model zooplankton recognition as a **hierarchical classification problem**, where predictions are made across multiple levels instead of a single flat label.

The model predicts:

- **Level 0**: `Zooplankton` vs `Not-Zooplankton`
- **Level 1**: broader parent categories
- **Level 2**: finer child classes

This design makes predictions more biologically meaningful and allows the model to enforce valid parent-child relationships during inference.

---

## Hierarchy Used in This Project

### Level 0
- `Zooplankton`
- `Not-Zooplankton`

### Level 1

Under `Zooplankton`:
- `Cladocera`
- `Rotifer`
- `Copepoda`

Under `Not-Zooplankton`:
- `Bubble`
- `Exoskeleton`
- `Fiber`
- `Plant_Matter`

### Level 2

Under `Cladocera`:
- `Bosmina`
- `Daphnia`

Under `Copepoda`:
- `Nauplius`
- `Cyclopoid`
- `Harpacticoid`
- `Calanoid`

Some Level 1 classes are terminal nodes and do not have a valid Level 2 child. In those cases, the Level 2 target is masked.

---

## Repository Structure

```text
root/
├── data/
│   ├── CSV_Data/
│   ├── Classified_Data/
│   ├── Processed_Data/
│   └── Inference_Test_images/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── pre_process_data.ipynb
│   └── train_test_prototype.ipynb
│
├── streamlit_app/
│   ├── app.py
│   ├── inference.py
│   ├── model.py
│   └── requirements.txt
│
└── models/                  # not included in GitHub, created locally when needed
```

---

## Folder and File Description

### `data/`

This folder contains the raw, processed, and testing data used in the project.

- **`Classified_Data/`**  
  Contains the original **mosaic images** provided by **Dr. Sofia from MNR**.

- **`CSV_Data/`**  
  Contains the metadata files associated with the mosaic images.

- **`Processed_Data/`**  
  Contains the cropped individual specimen images organized into class-specific folders after preprocessing.

- **`Inference_Test_images/`**  
  Contains random images used for quick inference and testing.

### `notebooks/`

This folder contains the main research and experimentation workflow.

- **`pre_process_data.ipynb`**  
  Used to preprocess the mosaic images by cutting them into individual organism images and organizing them into folders inside `Processed_Data/`. This notebook was originally provided by **Dr. Sofia**.

- **`eda.ipynb`**  
  Used for exploratory data analysis on the processed dataset, including class distribution and hierarchy inspection.

- **`train_test_prototype.ipynb`**  
  Contains the training, validation, testing, and prototype inference workflow for the hierarchical ViT model.

### `streamlit_app/`

This folder contains the deployment code for the web app.

- **`model.py`**  
  Defines the hierarchical class mappings, valid parent-child constraints, and the `HierarchicalZooplanktonViT` model architecture.

- **`inference.py`**  
  Contains the inference pipeline, including image preprocessing and strict top-down prediction logic with hierarchical masking.

- **`app.py`**  
  The Streamlit frontend. It loads the model, accepts uploaded images, runs inference, and displays predictions and confidence scores.

- **`requirements.txt`**  
  Lists the dependencies required to run the Streamlit app locally.

---

## Workflow

The project was developed in the following sequence:

1. I received the raw **mosaic dataset** from **MNR**, provided by **Dr. Sofia**.
2. The raw mosaic images were stored in `data/Classified_Data/`.
3. I used `notebooks/pre_process_data.ipynb` to cut the mosaic images into **individual specimen images** and save them into species/class folders under `data/Processed_Data/`.
4. I performed exploratory data analysis using `notebooks/eda.ipynb`.
5. I built and evaluated the hierarchical classification prototype in `notebooks/train_test_prototype.ipynb`.
6. I then created a Streamlit app for interactive inference using the files in `streamlit_app/`.

---

## Methodology

### 1. Problem Formulation

This project frames zooplankton image classification as a **3-level hierarchical classification task** rather than a flat classification task.

The model predicts:
- a coarse root-level label at **Level 0**
- a broader parent category at **Level 1**
- a finer child category at **Level 2**

This allows the model to better reflect the underlying structure of the dataset and produce more consistent predictions.

---

### 2. Data Source and Preprocessing

The original data was provided by **Dr. Sofia from MNR**.

The raw data consists of:
- mosaic images stored in `data/Classified_Data/`
- metadata stored in `data/CSV_Data/`

Since each mosaic may contain multiple organisms, preprocessing is necessary. Using `notebooks/pre_process_data.ipynb`, the raw mosaic images are cropped into **individual specimen images**, and these cropped samples are then organized into their respective folders inside `data/Processed_Data/`.

So the preprocessing pipeline is:

**Raw mosaic images -> cropped individual images -> class-specific folders -> model-ready dataset**

---

### 3. Exploratory Data Analysis

The notebook `notebooks/eda.ipynb` is used to inspect the processed dataset and understand its structure.

This includes:
- scanning the processed folder structure
- identifying classes and hierarchical levels
- counting images per class
- visualizing class imbalance

This step is important because the dataset is not evenly distributed across categories.

---

### 4. Hierarchical Label Construction

In `train_test_prototype.ipynb`, the folder-level labels are parsed into a 3-level hierarchy:

- `Target_L0`
- `Target_L1`
- `Target_L2`

These are then converted from string labels to integer targets so they can be used for model training.

For classes that do not have a valid child class at deeper levels, the target is masked using a sentinel value such as `-1`.

---

### 5. Dataset and Dataloader Design

A custom PyTorch dataset is created to return:
- the image tensor
- the hierarchical labels in dictionary format

```python
{
    "L0": int,
    "L1": int,
    "L2": int
}
```

All images are resized to **224 x 224**, which matches the input size expected by the Vision Transformer backbone.

#### Data augmentation used for training
- random horizontal flip
- random vertical flip
- random rotation
- color jitter
- conversion to tensor

Validation and test images use standard deterministic transforms without augmentation.

---

### 6. Train, Validation, and Test Split

The processed dataset is split into:
- **70% training**
- **15% validation**
- **15% test**

This split is used in `train_test_prototype.ipynb` for model development and final evaluation.

---

### 7. Model Architecture

The classifier is built using a **Vision Transformer (ViT-B/16)** backbone with **three separate classification heads**, one for each hierarchy level.

#### Architecture summary
- shared ViT backbone for feature extraction
- one classifier head for **Level 0**
- one classifier head for **Level 1**
- one classifier head for **Level 2**

This makes the model a **multi-head hierarchical classifier**.

In the current implementation, the ViT backbone is frozen and only the classifier heads are trained. This makes fine-tuning more practical for the available dataset size.

---

### 8. Handling Class Imbalance

Because some classes have much fewer samples than others, the project computes **inverse-frequency class weights** separately for each hierarchy level.

These weights are used in the loss function so that minority classes contribute more strongly during training.

---

### 9. Loss Function

Training uses a **hierarchical multi-task loss** composed of three weighted cross-entropy losses:

- one for Level 0
- one for Level 1
- one for Level 2

These are combined into a single total loss for backpropagation.

An important implementation detail is the use of:

```python
ignore_index = -1
```

This allows terminal nodes or missing deeper-level targets to be ignored safely during loss computation.

---

### 10. Optimization and Training Setup

The prototype training setup uses:

- **AdamW** optimizer
- learning rate: `1e-3`
- weight decay: `1e-4`
- epochs: `10`

The notebook tracks training and validation loss throughout the training process.

---

### 11. Strict Top-Down Inference

One of the key ideas in this project is **strict top-down hierarchical inference**.

Instead of predicting all hierarchy levels independently, predictions are made sequentially:

1. predict **Level 0**
2. use the Level 0 prediction to mask invalid **Level 1** classes
3. predict **Level 1**
4. use the Level 1 prediction to mask invalid **Level 2** classes
5. predict **Level 2**

#### Example

If the model predicts:

- `Level 0 = Not-Zooplankton`

then valid Level 1 options are restricted to:

- `Bubble`
- `Exoskeleton`
- `Fiber`
- `Plant_Matter`

If the model predicts:

- `Level 1 = Cladocera`

then valid Level 2 options are restricted to:

- `Bosmina`
- `Daphnia`

This prevents invalid combinations such as:
- `Not-Zooplankton -> Copepoda`
- `Cladocera -> Calanoid`

As a result, the final prediction is always hierarchically valid.

---

### 12. Evaluation Strategy

The project is evaluated at multiple levels:

- **Level 0 Accuracy**
- **Level 1 Accuracy**
- **Level 2 Accuracy**
- **Strict Path Accuracy**

#### Strict Path Accuracy
This is the hardest metric. A prediction is counted as correct only if the entire prediction path is correct across all levels simultaneously.

This gives a stronger measure of true hierarchical performance than level-wise accuracy alone.

---

## Results

Final test set performance:

- **Level 0 Accuracy:** `99.63%`
- **Level 1 Accuracy:** `96.34%`
- **Level 2 Accuracy:** `92.53%`
- **Strict Path Accuracy:** `91.74%`

These results show strong performance across the hierarchy while maintaining high full-path consistency.

---

## Streamlit App Explanation

The Streamlit app is the deployment layer of the project. It does not train the model. It simply loads the already trained model and makes it easy to test new images through a web interface.

### What the Streamlit files do

#### `model.py`
This file defines:
- the class mappings for each hierarchy level
- valid parent-child relationships
- the `HierarchicalZooplanktonViT` model

#### `inference.py`
This file handles:
- image preprocessing
- top-down prediction logic
- hierarchical masking
- confidence score generation
- conversion of numeric predictions back into readable class names

#### `app.py`
This file creates the web app using Streamlit. It:
- loads the model
- checks whether model weights are available locally
- downloads the model from Google Drive if needed
- accepts uploaded images
- runs inference
- displays Level 0, Level 1, and Level 2 predictions with confidence scores

---

## Model Weights

The trained model file is **not included in this GitHub repository** because the `.pth` file is too large to upload.

Instead, the model weights are hosted on Google Drive:

[Download model weights](https://drive.google.com/file/d/1NlvYztsQ9156BEuSBkl1uGMDTKe-EJc-/view?usp=drive_link)

---

## Important Note About the `models/` Folder

There is **no `models/` folder in the GitHub repo** because it is excluded using `.gitignore`.

However, if someone wants to run the project locally, there should be a folder called `models/` under the project root.

Expected local structure:

```text
root/
├── data/
├── notebooks/
├── streamlit_app/
└── models/
```

The trained model is expected to be downloaded into this `models/` folder.

For reproducibility, `train_test_prototype.ipynb` includes:

### `download_model_from_drive(file_id, output_path)`
This function:
- checks whether the model already exists in `root/models/`
- downloads it from Google Drive if it is missing
- saves it into the correct local folder

### `load_trained_model(filepath, device)`
This function:
- reconstructs the model architecture
- loads the trained weights
- moves the model to the target device

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

#### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Streamlit app dependencies

```bash
pip install -r streamlit_app/requirements.txt
```

---

## Running the Project

### Option 1: Run the Streamlit app locally

Run this from the **project root**:

```bash
streamlit run streamlit_app/app.py
```

### Option 2: Explore the notebooks

Open the notebooks in Jupyter and run them in this order:

1. `notebooks/pre_process_data.ipynb`
2. `notebooks/eda.ipynb`
3. `notebooks/train_test_prototype.ipynb`

---

## Notes

- The raw dataset was provided by **Dr. Sofia from MNR**
- The preprocessing notebook was also originally provided by **Dr. Sofia**
- `Inference_Test_images/` contains random test images for quick inference checks
- The GitHub repo does not include the large trained model file
- A local `models/` folder is expected when running the project outside GitHub

---

## Future Improvements

Potential future directions include:

- unfreezing and fine-tuning the ViT backbone
- experimenting with alternative backbones
- trying more hierarchy-aware loss functions
- improving class imbalance handling further
- adding batch inference support in the Streamlit app
- expanding the dataset with more labeled examples

---

## Acknowledgments

- **Dr. Sofia** for providing the dataset and the preprocessing notebook
- **MNR** for the zooplankton data used in this project

---

## Contact

Feel free to connect through GitHub or LinkedIn if you would like to discuss the project.