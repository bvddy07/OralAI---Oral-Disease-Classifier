Got it 👍 Here’s a cleaner **README.md** without the project structure section — you can copy-paste this directly:

---

````markdown
# 🦷 OralAI — Oral Disease Classifier

OralAI is a deep learning project built with **PyTorch** that classifies oral diseases from images using a fine-tuned **ResNet50** model.  
The goal is to provide a tool that can automatically identify different oral conditions from patient images.

---

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/bvdyy07/OralAI---Oral-Disease-Classifier.git
   cd OralAI---Oral-Disease-Classifier
````

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 🔹 Training (if you want to retrain)

```bash
python scripts/run_training.py
```

### 🔹 Evaluation

```bash
python scripts/run_evaluation.py
```

### 🔹 Random Prediction

```bash
python scripts/predict_random.py
```

---

## 🧠 Model Details

* **Base Model:** ResNet50 (pretrained on ImageNet)
* **Image Size:** 128x128
* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss
* **Epochs:** 10
* **Batch Size:** 32

---

## 📊 Results

* Training Accuracy: \~XX% (replace with your results)
* Test Accuracy: \~XX% (replace with your results)

---

## 📌 Notes

* Trained model is saved in `models/OralAI.pth` (not included in repo due to GitHub’s 100 MB file size limit).
* Update `config.py` to set dataset paths before training.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

```

---

✅ This version is short, clean, and GitHub-ready.  

Do you also want me to add **badges at the top** (Python, PyTorch, License) to make it more professional?
```
