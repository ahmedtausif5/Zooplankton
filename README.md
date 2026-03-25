# Hierarchical Zooplankton Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zooplankton-classifier.streamlit.app/)

A computer vision pipeline utilizing a custom multi-head Vision Transformer (ViT) to classify microscopic images of zooplankton and aquatic debris. The model enforces strict biological taxonomy through top-down hierarchical masking to ensure biologically impossible predictions are systematically prevented.

## Live Web Application
The model is fully deployed and accessible via Streamlit Community Cloud:
**[Test the Live Application Here](https://zooplankton-classifier.streamlit.app/)**

***

## Project Workflow & Data Pipeline

This project was built systematically from raw data processing to web deployment:

1. **Raw Data Acquisition:** The initial dataset consisted of raw mosaic images provided by the Ministry of Natural Resources (MNR) via Dr. Sofia. These were stored in `data/Classified_Data`.
2. **Preprocessing:** Using the `notebooks/pre_process_data.ipynb` script provided by Dr. Sofia, the raw mosaic images were systematically cropped into individual specimen images and sorted into specific taxonomic folders within `data/Processed_Data`.
3. **Exploratory Data Analysis (EDA):** Conducted in `notebooks/eda.ipynb` to understand class distributions, image dimensions, and biological hierarchies.
4. **Model Prototyping:** The core deep learning research, including training, validation, and local inference testing of the Vision Transformer, was developed in `notebooks/train_test_prototype.ipynb`.
5. **Web Deployment:** The finalized inference logic and model architecture were abstracted into modular Python scripts to power a live Streamlit web application.

***

## Repository Structure

```text
Zooplankton-Data/
├── data/
│   ├── Classified_Data/      # Original raw mosaic images from MNR
│   ├── CSV_Data/             # Tabular metadata and structured labels
│   └── Processed_Data/       # Cropped and sorted individual specimen images
├── notebooks/
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── pre_process_data.ipynb        # Cropping and sorting script
│   └── train_test_prototype.ipynb    # Model training and validation pipeline
└── streamlit_app/            # Lightweight production deployment code
    ├── app.py                # Streamlit frontend and UI logic
    ├── inference.py          # Image processing and strict masking logic
    ├── model.py              # PyTorch ViT class and mapping dictionaries
    └── requirements.txt      # Production dependencies