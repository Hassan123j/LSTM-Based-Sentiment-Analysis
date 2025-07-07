# Social Media Sentiment Analyzer

A basic sentiment analysis web app for classifying social media posts as positive or negative. Built with Flask and a pre-trained LSTM model (TensorFlow/Keras). There's also a Jupyter notebook showing the full training pipeline (RNNs, GRU, and LSTM).

---

## What it does

* Predicts sentiment (positive or negative) from text input
* Outputs a confidence score
* Includes basic NLP preprocessing (lowercasing, stopword removal, stemming)
* Runs as a simple Flask web app

---

## Files & Structure

```
.
├── main.py                         # Flask app
├── lstm_model.h5                   # Trained LSTM model (Keras)
├── tokenizer.pkl                   # Tokenizer used to preprocess input
├── templates/
│   └── index.html                  # Web UI (form + output)
├── sentiment-analysis-on-social-media-posts-with-lstm-rnn-and-gru.ipynb
                                    # Notebook with model training + evaluation
```

---

## Setup

### 1. Clone this repo

```bash
git clone <repo-url>
cd <repo-dir>
```

### 2. (Optional) Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install Flask tensorflow numpy nltk scikit-learn pandas matplotlib bz2file
```

Note: `main.py` will auto-download NLTK stopwords if needed.

### 4. Make sure the model files are in place

Ensure these two files are in the project root:

* `lstm_model.h5`
* `tokenizer.pkl`

If they’re missing, you’ll need to run the notebook to train the model and save them.

### 5. Run the app

```bash
python main.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## How to use

1. Open the app in your browser
2. Paste or type some text (e.g., a social media post)
3. Click the button
4. You’ll see the sentiment result and confidence score

---

## Training the model (Notebook)

The notebook (`sentiment-analysis-on-social-media-posts-with-lstm-rnn-and-gru.ipynb`) walks through the full training process:

* Loads data (Amazon reviews, `.bz2` format) from [this Kaggle dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
* Preprocesses text (cleaning, stemming, etc.)
* Tokenizes and pads input
* Trains and compares RNN, GRU, LSTM, and Bidirectional LSTM models
* Evaluates accuracy, confusion matrix, etc.
* Saves final model (`lstm_model.h5`) and tokenizer (`tokenizer.pkl`)

The final model used in `main.py` is a standard LSTM.

---

## Dependencies

* Python 3.x
* Flask
* TensorFlow / Keras
* NLTK
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* bz2file

---

## License

MIT License. See `LICENSE` for details.
