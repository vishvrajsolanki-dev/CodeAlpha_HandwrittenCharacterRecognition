# LetterLens — Handwritten Digit Recognition

A CNN-powered Streamlit app that recognizes handwritten digits (0–9) drawn directly on a canvas or uploaded as an image.

Built as **CodeAlpha ML Internship — Task 3**.

## Demo

Draw a digit on the canvas (or upload an image) and the model predicts it instantly, showing the top-3 most likely digits with confidence scores.

## Model

- **Dataset:** MNIST (60,000 training images, 10,000 test images)
- **Architecture:** Convolutional Neural Network — Conv2D ×4, BatchNormalization, MaxPooling, Dropout, Dense layers
- **Test Accuracy:** 99.52%
- **Input:** 28×28 grayscale images
- **Output:** 10 classes (digits 0–9)

## Tech Stack

- **TensorFlow / Keras** — model training and inference
- **Streamlit** — web app UI
- **streamlit-drawable-canvas** — interactive drawing canvas
- **Plotly** — confidence visualization
- **Pillow / NumPy** — image preprocessing

## Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
├── app.py                  # Streamlit application
├── train.py                # CNN training script
├── utils/
│   └── preprocess.py       # Canvas & upload image preprocessing
├── model/
│   └── letterlens_cnn.keras
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

1. User draws a digit on the canvas or uploads an image
2. Image is converted to grayscale, inverted, resized to 28×28, and normalized
3. The trained CNN predicts the digit and returns confidence scores
4. Top-3 predictions are displayed with a bar chart

## Notes

The model is trained specifically on MNIST and supports single handwritten digits (0–9). For best results, write large, centered, with thick strokes.

---
Built by Vishvraj · CodeAlpha ML Internship · 2026