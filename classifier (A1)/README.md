# Wikipedia Text Classifier

A simple tool that tells you whether a piece of text is about geography (like cities, rivers, countries) or something else (like science, art, technology).

## What it does

This program reads text and decides if it's talking about:
- **Geographic stuff**: places, cities, rivers, mountains, countries, etc.
- **Non-geographic stuff**: science, art, music, technology, philosophy, etc.

## How it works

1. **Gets training data**: Downloads Wikipedia articles about different topics
2. **Cleans the text**: Removes common words like "the", "is", "at" and converts words to their basic form
3. **Learns patterns**: Figures out which words are common in geographic vs non-geographic articles
4. **Makes predictions**: Uses what it learned to classify new text

## NLP Methods Used

The program uses several natural language processing techniques:

### Text Preprocessing
- **Tokenization**: Breaks text into individual words (e.g., "New York City" becomes ["New", "York", "City"])
- **Stopword Removal**: Removes common words that don't add meaning (e.g., "the", "is", "at", "which")
- **Lemmatization**: Converts words to their base form (e.g., "cities" becomes "city", "running" becomes "run")
- **Case Normalization**: Converts all text to lowercase for consistency

### Feature Extraction
- **Bag of Words**: Counts how many times each word appears in the text
- **Word Frequency Analysis**: Identifies which words are most common in geographic vs non-geographic articles

### Machine Learning
- **Logistic Regression**: The main classifier that learns to distinguish between the two categories
- **Train-Test Split**: Uses 80% of data for training, 20% for testing

## How to use it

### First time setup
```bash
# From the root of the repository
pip install -r requirements.txt

# Or install packages individually
pip install nltk scikit-learn wikipedia
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

### Train the model
```bash
python classifier.py --train
```
This downloads Wikipedia articles and teaches the program how to classify them. It takes a few minutes.

### Test how well it works
```bash
python classifier.py --test
```
This tests the model on some new articles it hasn't seen before.

### Classify your own text
```bash
python classifier.py --predict "Your text here"
```

Examples:
```bash
python classifier.py --predict "Paris is the capital of France"
python classifier.py --predict "Quantum physics is a branch of science"
```

## What's in the files

- `classifier.py` - The main program
- `geo_classifier_model.pkl` - The trained model (created after training)
- `geo_vectorizer.pkl` - The word processor (created after training)
- `README.md` - This file

## How it was built

I used:
- **NLTK** for cleaning and processing text
- **scikit-learn** for the machine learning part
- **Wikipedia API** to get training data
- **Logistic Regression** as the main classifier

The program downloads articles about different topics (cities, rivers, mountains for geography; math, art, science for non-geography) and learns to tell them apart based on the words they use.

## Results

On test data, the model gets about 90% accuracy, which means it's usually right about whether text is geographic or not.

## Limitations

- It works best with longer texts (like Wikipedia articles)
- It might struggle with very short sentences
- It depends on having seen similar words during training
- It's a simple approach - there are more advanced methods available

This was built for an NLP assignment to demonstrate basic text classification techniques. 