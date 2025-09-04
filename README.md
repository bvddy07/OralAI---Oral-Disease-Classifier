# 🦷 OralAI — Oral Disease Classifier

OralAI is a deep learning project built with PyTorch that classifies oral diseases from images using a fine-tuned ResNet50 model.
The goal is to provide a tool that can automatically identify different oral conditions from patient images.

## Setup

1. Clone the repository
   git clone [https://github.com/bvdyy07/OralAI---Oral-Disease-Classifier.git](https://github.com/bvdyy07/OralAI---Oral-Disease-Classifier.git)
   cd OralAI---Oral-Disease-Classifier

2. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate   (Linux/Mac)
   venv\Scripts\activate      (Windows)

3. Install dependencies
   pip install -r requirements.txt

## Usage

Training (if you want to retrain)
python scripts/run\_training.py

Evaluation
python scripts/run\_evaluation.py

Random Prediction
python scripts/predict\_random.py

## Model Details

* Base Model: ResNet50 (pretrained on ImageNet)
* Image Size: 128x128
* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Epochs: 10
* Batch Size: 32

## Results

* Test Accuracy: 96 %

## Notes

* Trained model is saved in `models/OralAI.pth` (not included in repo due to GitHub’s 100 MB file size limit).
* Update `config.py` to set dataset paths before training.

## License

This project is licensed under the MIT License.

