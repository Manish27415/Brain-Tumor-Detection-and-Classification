# Brain-Tumor-Detection-and-Classification
# Brain Tumor Classification from MRI using CNN

This project uses a Convolutional Neural Network (CNN) to automatically detect and classify brain tumors from MRI scan images into three types: **Meningioma**, **Glioma**, and **Pituitary tumors**.

---

## 🧠 Problem Statement

Manual diagnosis of brain tumors from MRI images can be time-consuming and error-prone, especially in areas lacking medical experts. This project aims to build a deep learning model that automates this classification process using labeled MRI data stored in `.mat` files. The model classifies images into one of the three tumor types with high accuracy and can also detect the presence or absence of a tumor in a new MRI scan.

---

## 📁 Dataset

- Format: `.mat` files (MATLAB format)
- Each file contains:
  - `image`: 512x512 grayscale MRI scan
  - `label`: Tumor type (1: Meningioma, 2: Glioma, 3: Pituitary)
- Total images: 3064

---

## 🛠️ Preprocessing

- Loaded `.mat` files using `h5py`
- Normalized input data to shape `(512, 512, 1)`
- Split into 80% training and 20% testing data
- Labels converted to range `[0, 2]` for compatibility with softmax output

---

## 🧪 Model Architecture

- Convolutional layers with increasing filter sizes (64 → 128 → 256 → 512)
- MaxPooling and Batch Normalization after each block
- Global Average Pooling and Fully Connected layers
- Dropout for regularization
- Final Softmax layer for 3-class classification

---

## ⚙️ Training

- Loss Function: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: 11
- Batch Size: 32
- Validation accuracy and loss tracked using `matplotlib`

---

## 📊 Evaluation

- Confusion Matrix and Classification Report
- Accuracy Score
- ROC Curve for class-wise prediction
- Visualization with Seaborn heatmaps

---

## 🔍 Predicting New Images

- Accepts `.mat` file as input
- Preprocesses and feeds image into the trained model
- Displays whether a tumor is present and the class prediction

---

---

## ✅ Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- h5py
- scikit-learn
- OpenCV (optional)

---

## 🚀 How to Run

1. Clone the repository or run in Google Colab.
2. Place `.mat` files in the `/BRAIN_DATA` folder.
3. Run the complete notebook script.
4. Visualize training results and make predictions.

---

## 🤝 Acknowledgements

- Dataset credits: Figshare Brain MRI dataset
- Inspired by the need to assist early tumor diagnosis using deep learning tools.

---

