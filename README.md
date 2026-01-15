🏏 Next Word Prediction using LSTM (RNN)
📖 Overview

This project demonstrates a Next Word Prediction system using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM).
The entire workflow is implemented within a single Jupyter Notebook, starting from reading a text article about cricket, preprocessing the text, training the model, and finally predicting the next word.

The model learns word sequences from the cricket article and predicts the most probable next word for a given input phrase.

📘 Dataset

Source: A text article related to cricket

Format: Plain text

The text is loaded directly inside the notebook and used for training.

🔄 Workflow (Implemented in One Notebook)

All steps are implemented sequentially in one .ipynb file:

1️⃣ Text Loading

The cricket article text is read into the notebook.

2️⃣ Text Preprocessing

Converted text to lowercase

Tokenized text using Keras Tokenizer

Created input sequences

Encoded target variable (y)

Applied padding to ensure uniform sequence length

3️⃣ Model Building

The neural network architecture consists of:

Embedding Layer – to learn word representations

LSTM Layer – to capture sequence dependencies

Dense Layer with Softmax activation – for multi-class word prediction

4️⃣ Model Compilation

The model is compiled using:

Loss function: categorical_crossentropy

Optimizer: Adam

Metric: accuracy

5️⃣ Model Training

The model is trained on word sequences generated from the cricket article.

Training is performed directly within the notebook.

6️⃣ Prediction Pipeline

A prediction pipeline is implemented to:

Accept a seed text

Tokenize and pad the input

Predict the next word using the trained model

🧠 Example

Input:

The cricket team


Predicted Next Word:

won

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Jupyter Notebook

📂 Project Structure
Next-Word-Prediction-LSTM/
│
├── Next_Word_Prediction_LSTM.ipynb
├── README.md
├── requirements.txt
└── .gitignore

📄 requirements.txt
tensorflow
numpy
jupyter

📌 Key Notes

Entire implementation is contained in a single Jupyter Notebook

Uses word-level prediction

Dataset is a cricket-related article

Designed for educational and learning purposes
