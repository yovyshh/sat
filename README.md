# 🛰️ Satellite Image Classifier & Deforestation Detector

This project is a deep learning-based tool that uses PyTorch to train a ResNet-50 classifier on the EuroSAT dataset to detect various types of land cover and identify segments of deforestation. A Streamlit web application provides an interactive dashboard where users can upload "Before" and "After" satellite images to run the deforestation detection engine and visualize results.

## 🚀 Features

- **Model Training**: A custom training pipeline (`train_classifier.py`) built on `resnet50` with fine-tuning, data augmentation, test evaluation, and confusion matrix visualization.
- **Interactive UI**: A Streamlit dashboard (`app.py`) to upload side-by-side temporal images and run patch-by-patch forest cover change analysis.
- **Diagnostics**: An expandable diagnostic log directly in the UI to see the raw classification predictions for both scenes.

## 📂 Project Structure

- `app.py`: Streamlit application file.
- `train_classifier.py`: Script to train the PyTorch model on the EuroSAT dataset.
- `confusion_matrix.png`: Model evaluation visual (generated after training).
- `requirements.txt`: Python package requirements.
- `.gitignore`: Standard rules to exclude virtual environments, datasets, and large weights from version control.

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd sat
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv sat_env
   sat_env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**:
   Download the **EuroSAT RGB** dataset and extract it to a directory. Update the `DATASET_ROOT` variable in `train_classifier.py` to point to your local dataset folder.

## 🏋️ Training the Classifier

To train the ResNet-50 classifier:
```bash
python train_classifier.py
```
This script will:
- Train the head of the model (Phase 1).
- Unfreeze all layers and fine-tune with a low learning rate (Phase 2).
- Save the trained model parameters to `local_resnet50_eurosat.pth`.
- Output evaluation results and save a `confusion_matrix.png` plot.

## 🖥️ Running the Web App

Once the model weights (`local_resnet50_eurosat.pth`) are generated, start the Streamlit app:
```bash
streamlit run app.py
```
Upload two images (representing before and after states of the same area) to inspect forest change.
