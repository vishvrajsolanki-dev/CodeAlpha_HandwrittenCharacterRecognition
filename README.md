# LetterLens — Handwritten Character Recognition

A Streamlit app that recognizes handwritten digits drawn directly on a canvas, powered by a CNN trained on MNIST.

## What It Does

Draw a digit on the canvas, get a real-time prediction with confidence score.

## Approach

Originally targeted EMNIST for full alphanumeric recognition, but EMNIST's cursive/dataset handwriting style didn't transfer well to canvas-drawn input — the model performed poorly on real user strokes despite good validation accuracy. Pivoted to MNIST digits after systematic root-cause debugging of the mismatch, prioritizing a model that actually works on real input over one that scores well on a mismatched benchmark.

- **Architecture:** CNN with Keras built-in augmentation layers (`RandomRotation`, `RandomZoom`, `RandomTranslation`) to generalize better to imperfect canvas strokes
- **Result:** 99.24% test accuracy

## Stack

Python · TensorFlow/Keras · Streamlit · CNN

## UI

Dark theme with gold accents, live drawable canvas.

---

*CodeAlpha Machine Learning Internship — Task 3*
