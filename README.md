---

# 🌌 ExoMiner-like Exoplanet Classifier

This repository contains an implementation of a deep learning classifier inspired by **NASA’s ExoMiner**, designed to identify exoplanet candidates from Kepler/TESS light curves and auxiliary stellar features.

The model uses **CNNs** for light-curve views (global + local) combined with **MLP layers** for stellar parameters and vetting metrics.

---

## 🚀 Features

* Preprocesses light curves into **global and local folded views**
* Incorporates **stellar parameters & vetting features**
* Outputs **probability of planet vs. false positive**
* Saves model in both **`.pkl` (pipeline)** and **`.keras` (TensorFlow format)**
* `.keras` format is recommended for:

  * TensorFlow Serving
  * TensorFlow Lite (mobile)
  * TensorFlow.js (web)
  * Transfer learning
  * Visualization tools

---

## 🛠️ Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/exominer-classifier.git
cd exominer-classifier
```

### 2. Create and activate virtual environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 Training the Model

Run the training script:

```bash
python exominer.py
```

This will:

* Train the model on provided/preprocessed dataset
* Save results to:

  * `model.pkl` → includes full pipeline (preprocessing + model) for predictions
  * `model.keras` → standard TensorFlow format, most reliable for long-term storage

If you only need predictions with the trained pipeline, the `.pkl` file is enough.
If you want to use TensorFlow tools or extend the model, keep the `.keras` file.

---
