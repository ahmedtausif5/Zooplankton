# Hierarchical Classification of Zooplankton Images

A deep learning project for **hierarchical image classification of zooplankton images** using a **Vision Transformer (ViT-B/16)** backbone, **multi-head prediction**, and **strict top-down hierarchical masking** during inference.

## Live Demo

- **Streamlit App:** [zooplankton-classifier.streamlit.app](https://zooplankton-classifier.streamlit.app/)

## Overview

This project classifies zooplankton images using a **3-level hierarchy** instead of a flat label space. Rather than predicting a single class directly, the model predicts progressively from coarse to fine categories, which makes the output more structured and biologically meaningful.

The pipeline starts from raw **mosaic images** provided by **Dr. Sofia from MNR**, preprocesses them into cropped individual specimen images, trains a hierarchical ViT-based classifier, and serves predictions through a Streamlit web app.

## Hierarchy

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

Some Level 1 classes are terminal nodes, so they do not have valid Level 2 children.

## Results

Final test set performance:

- **Level 0 Accuracy:** `99.63%`
- **Level 1 Accuracy:** `96.34%`
- **Level 2 Accuracy:** `92.53%`
- **Strict Path Accuracy:** `91.74%`

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
└── models/   # created locally, not tracked in GitHub
```

## Workflow

1. Raw mosaic images and metadata were provided by **Dr. Sofia from MNR**.
2. `pre_process_data.ipynb` crops the mosaic images into individual specimen images.
3. Cropped images are organized into class folders inside `data/Processed_Data/`.
4. `eda.ipynb` is used for exploratory data analysis.
5. `train_test_prototype.ipynb` handles training, validation, testing, and prototype inference.
6. The trained model is deployed through a Streamlit app in `streamlit_app/`.

## Method Summary

The model uses a **ViT-B/16 backbone** with **three classifier heads**, one for each hierarchy level. During inference, predictions are made in a strict top-down manner:

1. predict Level 0  
2. mask invalid Level 1 classes based on the predicted parent  
3. predict Level 1  
4. mask invalid Level 2 classes based on the predicted parent  
5. predict Level 2  

This prevents invalid class combinations and ensures that the final prediction always follows a valid hierarchy path.

## Streamlit App

The `streamlit_app/` folder contains the deployment code:

- `model.py` defines the hierarchical label mappings and model architecture
- `inference.py` handles preprocessing and top-down hierarchical inference
- `app.py` creates the Streamlit UI for uploading images and viewing predictions
- `requirements.txt` lists the dependencies needed to run the app locally

## Model Weights

The trained `.pth` file is **not included in the GitHub repository** because it is too large.

Model weights are hosted on Google Drive:

- **Download link:** [Model Weights](https://drive.google.com/file/d/1NlvYztsQ9156BEuSBkl1uGMDTKe-EJc-/view?usp=drive_link)

### Note about `models/`
There is no `models/` folder in the GitHub repo because it is excluded via `.gitignore`.  
If you want to run the project locally, create a `models/` folder under the project root so the trained model can be downloaded there.

Expected local structure:

```text
root/
├── data/
├── notebooks/
├── streamlit_app/
└── models/
```

For reproducibility, `train_test_prototype.ipynb` includes helper functions to:
- download the model from Google Drive if missing
- load the trained model from the local `models/` folder

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

Create a virtual environment and activate it:

### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r streamlit_app/requirements.txt
```

## Run Locally

Run the Streamlit app from the project root:

```bash
streamlit run streamlit_app/app.py
```

## Notes

- The raw dataset was provided by **Dr. Sofia from MNR**
- `pre_process_data.ipynb` was originally provided by **Dr. Sofia**
- `Inference_Test_images/` contains random images for quick testing
- The full training and experimentation workflow is in the notebooks

## Acknowledgments

- **Dr. Sofia** for providing the dataset and preprocessing notebook
- **MNR** for the zooplankton data used in this project