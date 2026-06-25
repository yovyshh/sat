# 🛰️ Satellite Image Classifier & Deforestation Detector

An end-to-end deep learning project built to classify land cover types and monitor regional deforestation trends using Sentinel-2 satellite imagery.

## 🚀 Technical Architecture
* **Model Engine:** PyTorch transfer learning via a customized `ResNet50` pipeline.
* **Optimization:** Fine-tuned cross-entropy classification reaching **98% test accuracy** locally on an NVIDIA RTX 5070.
* **Frontend:** Interactive web application dashboard deployed via `Streamlit`.

## 📂 Project Structure
- `app.py`: Streamlit application file.
- `train_classifier.py`: Script to train the PyTorch model on the EuroSAT dataset.
- `confusion_matrix.png`: Model evaluation visual (generated after training).
- `requirements.txt`: Python package requirements.
- `.gitignore`: Standard rules to exclude virtual environments, datasets, and large weights from version control.

## 📦 Project Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yovyshh/sat.git
   cd sat
   ```

2. **Create your virtual environment**:
   ```bash
   python -m venv sat_env
   sat_env\Scripts\activate
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
   Or install them manually:
   ```bash
   pip install torch torchvision scikit-learn streamlit matplotlib seaborn pillow
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
