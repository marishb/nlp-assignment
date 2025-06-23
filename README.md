# NLP Assignment Repository

**This repository belongs to Maral Sheikhi Biglari, master student of linguistics at the University of Verona.**

This repository contains solutions for two major NLP assignments:

- **A1: Text Classifier**
- **A2: Hierarchical Style-Based Summarizer**

GitHub link: [https://github.com/marishb/nlp-assignment/tree/main/](https://github.com/marishb/nlp-assignment/tree/main/)

---

## Assignment 1: Text Classifier (A1)

**Folder:** `classifier (A1)`

- Implements a text classification pipeline using NLP techniques.
- Uses NLTK and scikit-learn for feature extraction and model training.
- Supports training, evaluation, and prediction on new data.
- Includes a clear README with setup and usage instructions.

**Key Features:**
- Preprocessing: tokenization, stopword removal, stemming/lemmatization
- Feature extraction: Bag-of-Words, TF-IDF
- Model: Logistic Regression (or similar)
- Evaluation: Accuracy, precision, recall, F1-score

**How to use:**
- See the `classifier (A1)/README.md` for detailed instructions.

---

## Assignment 2: Hierarchical Style-Based Summarizer (A2)

**Folder:** `summarizer (A2)`

- Summarizes an input document in the style of another document (style transfer summarization).
- Handles long documents using a hierarchical chunking and summarization approach.
- Uses NLTK for tokenization and style analysis.
- Includes automated test cases in nested folders for easy validation.

**Key Features:**
- Style analysis: average sentence length, vocabulary overlap
- Hierarchical summarization for long texts
- Extractive, style-aware sentence selection
- Automated test runner for batch testing

**How to use:**
- See the `summarizer (A2)/README.md` for setup, usage, and test automation details.

---

## Repository Structure

```
classifier (A1)/         # Assignment 1: Text classification
summarizer (A2)/         # Assignment 2: Hierarchical style-based summarization
requirements.txt         # Python dependencies
README.md                # This file
```

---

## License & Contact

This repository is for educational purposes as part of an NLP course. For questions or contributions, see the GitHub page:

[https://github.com/marishb/nlp-assignment/tree/main/](https://github.com/marishb/nlp-assignment/tree/main/) 