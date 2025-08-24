# 🎵 Music Genre Classification

This project classifies songs into **10 genres** (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using the **GTZAN dataset**.  
It applies **audio feature extraction (MFCCs, chroma, spectral contrast, tonnetz)** with `librosa`, then trains machine learning models for **multi-class classification**.

---

## 📂 Dataset
- **GTZAN Dataset** (available on [Kaggle](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection))  
- Contains **1,000 audio files** (30 sec each, 10 genres × 100 files).

---

## 🛠️ Tools & Libraries
- **Python 3**
- [Librosa](https://librosa.org/) – audio feature extraction  
- [Scikit-learn](https://scikit-learn.org/) – ML models  
- [Matplotlib / Seaborn](https://matplotlib.org/) – visualization  
- [Pickle](https://docs.python.org/3/library/pickle.html) – model saving  

---

## ⚙️ Features Extracted
From each audio file:
- **MFCCs (Mel Frequency Cepstral Coefficients)**
- **Delta & Delta² of MFCCs**
- **Chroma features**
- **Spectral contrast**
- **Tonnetz features**

All features are standardized using `StandardScaler`.

---

## 🧠 Models Trained
1. **Random Forest Classifier** (baseline)  
2. **MLP Neural Network (Scikit-learn)** – improved accuracy  

📊 Current best accuracy: **~70%**  

---

## 📈 Evaluation
- **Classification Report (Precision, Recall, F1-score)**
- **Confusion Matrix (heatmap)**
- **Accuracy Score**
