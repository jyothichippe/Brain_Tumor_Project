<<<<<<< HEAD
# Brain Tumor MRI Image Classification

## Project Overview
This project uses deep learning and computer vision to classify brain MRI images into the categories provided by the dataset. It compares a custom Convolutional Neural Network (CNN) with MobileNetV2 transfer learning.

## Objectives
- Preprocess and augment MRI images.
- Build a custom CNN model.
- Apply transfer learning using MobileNetV2.
- Evaluate and compare model performance.
- Develop an interactive Streamlit application for image classification.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Pillow

## Project Structure
- `train.py` — Data loading, training, evaluation and model comparison.
- `app.py` — Streamlit application.
- `models/` — Saved models and class labels.
- `results/` — Evaluation metrics and visualizations.
- `data/` — Training and testing images.

## Methodology
1. Dataset exploration and preprocessing.
2. Image resizing and normalization.
3. Training data augmentation.
4. Custom CNN development.
5. MobileNetV2 transfer learning.
6. Model evaluation using accuracy, precision, recall and F1-score.
7. Confusion matrix and training-curve visualization.
8. Streamlit application development.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the models:

```bash
python train.py
```

Launch the web application:

```bash
streamlit run app.py
```

## Results
The model comparison metrics are available in `results/model_comparison.csv`. Actual results should be reported after completing training and testing.

## Disclaimer
This project is an educational research demonstration. Its predictions are not validated for clinical diagnosis or treatment decisions.
=======
# Brain_Tumor_Project
Brain Tumor Detection using CNN and MobileNetV2 to classify MRI images with deep learning, TensorFlow, and Streamlit. Developed as an AI/ML internship project.
>>>>>>> 2ab36fb667b5c68d487faebae96c78c9733cd6b4
