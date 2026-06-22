# Dog Vision — Multi-class Dog Breed Classifier

An end-to-end deep learning project that identifies dog breeds from images using **TensorFlow 2.x** and **Transfer Learning** with MobileNetV2.

---

##  Problem Statement

Given an image of a dog, predict its breed from **120 possible classes**.

---

##  Dataset

- Source: [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data)
- ~10,000 labeled training images across 120 breeds
- ~10,000 unlabeled test images

> **Note:** The dataset is not included in this repo. Download it from Kaggle and place it in a folder named `dog-breed-identification/` at the root of the project.

Expected folder structure:
```
dog-breed-identification/
├── train/          # training images (.jpg)
├── test/           # test images (.jpg)
└── labels.csv      # image IDs and breed labels
```

---

## Model Architecture

| Layer | Details |
|---|---|
| Base model | MobileNetV2 (pretrained on ImageNet via TF Hub) |
| Output layer | Dense(120, activation='softmax') |
| Loss | Categorical Crossentropy |
| Optimizer | Adam |

- The MobileNetV2 backbone is **frozen** during initial training (only the 120-class head is trained).
- Training uses **EarlyStopping** (patience=3) and **TensorBoard** logging.
- Two training runs: first on a 1,000-image subset, then on the full dataset.

---

##  Results

| Training Set | Val Accuracy (approx.) |
|---|---|
| 1,000 images (subset) | ~65% |
| Full dataset (~10k images) | Higher (see TensorBoard logs) |

---

##  Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/Dog-vision_project.git
cd Dog-vision_project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Go to the [Kaggle page](https://www.kaggle.com/c/dog-breed-identification/data), download and unzip the data into `dog-breed-identification/`.

### 5. Run the notebook
```bash
jupyter notebook dog-vision.ipynb
```

---

## 📁 Project Structure

```
Dog-vision_project/
├── dog-vision.ipynb               # Main notebook
├── labels.csv                     # Breed labels (from Kaggle)
├── requirements.txt               # Python dependencies
├── models/                        # Saved model files (generated after training)
├── logs/                          # TensorBoard logs (generated during training)
└── README.md
```

---

##  Technologies Used

- Python 3.12
- TensorFlow 2.18
- TensorFlow Hub 0.16
- tf_keras (Keras 2 compatibility layer)
- NumPy, Pandas, Matplotlib, scikit-learn

---

## Acknowledgements

- [Kaggle Dog Breed Identification competition](https://www.kaggle.com/c/dog-breed-identification)
- [TensorFlow Hub](https://tfhub.dev/)
- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381)
- [ZeroToMastery TensorFlow course](https://zerotomastery.io/)
