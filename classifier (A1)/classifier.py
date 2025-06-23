import argparse
import pickle
import sys

import nltk
import wikipedia
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


MODEL_PATH = "geo_classifier_model.pkl"
VECTORIZER_PATH = "geo_vectorizer.pkl"


def save_model(model, vectorizer):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def fetch_sample_wikipedia_data():
    # Example topics: Geographic and Non-Geographic
    geo_titles = ["Paris", "Amazon River", "Mount Everest"]
    non_geo_titles = ["Quantum mechanics", "Photosynthesis", "Artificial intelligence"]
    texts = []
    labels = []
    for title in geo_titles:
        try:
            text = wikipedia.page(title).content
            texts.append(text)
            labels.append("geographic")
        except Exception as e:
            print(f"Error fetching {title}: {e}")
    for title in non_geo_titles:
        try:
            text = wikipedia.page(title).content
            texts.append(text)
            labels.append("non-geographic")
        except Exception as e:
            print(f"Error fetching {title}: {e}")
    return texts, labels


# Define separate keyword sets for training and testing
TRAIN_GEO_KEYWORDS = [
    "city",
    "river",
    "mountain",
    "country",
    "continent",
    "ocean",
    "lake",
    "island",
    "desert",
    "valley",
]
TRAIN_NON_GEO_KEYWORDS = [
    "mathematics",
    "philosophy",
    "biology",
    "physics",
    "music",
    "art",
    "technology",
    "psychology",
    "economics",
    "literature",
]
TEST_GEO_KEYWORDS = ["peninsula", "archipelago", "plateau", "fjord", "volcano"]
TEST_NON_GEO_KEYWORDS = ["sociology", "linguistics", "chemistry", "painting", "robotics"]


def fetch_articles(keywords, label, max_per_keyword=100):
    articles = []
    labels = []
    for keyword in keywords:
        try:
            search_results = wikipedia.search(keyword, results=max_per_keyword)
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    articles.append(page.content)
                    labels.append(label)
                except wikipedia.DisambiguationError as e:
                    for option in e.options[:1]:
                        try:
                            page = wikipedia.page(option, auto_suggest=False)
                            articles.append(page.content)
                            labels.append(label)
                            break
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            continue
    return articles, labels


def load_train_data():
    geo_texts, geo_labels = fetch_articles(TRAIN_GEO_KEYWORDS, "geographic")
    non_geo_texts, non_geo_labels = fetch_articles(TRAIN_NON_GEO_KEYWORDS, "non-geographic")
    texts = geo_texts + non_geo_texts
    labels = geo_labels + non_geo_labels
    return texts, labels


def load_test_data():
    geo_texts, geo_labels = fetch_articles(TEST_GEO_KEYWORDS, "geographic", max_per_keyword=5)
    non_geo_texts, non_geo_labels = fetch_articles(TEST_NON_GEO_KEYWORDS, "non-geographic", max_per_keyword=5)
    texts = geo_texts + non_geo_texts
    labels = geo_labels + non_geo_labels
    return texts, labels


def preprocess_texts(texts):
    processed = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens if t.isalpha() and t.lower() not in stop_words]
        processed.append(' '.join(tokens))
    return processed


def extract_features(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb))
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))


def train_and_save():
    texts, labels = load_train_data()
    if not texts or not labels:
        print("No training data loaded.")
        sys.exit(1)
    processed_texts = preprocess_texts(texts)
    X, vectorizer = extract_features(processed_texts)
    model = LogisticRegression(max_iter=200)
    model.fit(X, labels)
    save_model(model, vectorizer)
    print("Model and vectorizer saved.")


def predict_label(text):
    model, vectorizer = load_model()
    processed = preprocess_texts([text])
    print(processed)
    X = vectorizer.transform(processed)
    print(X)
    label = model.predict(X)[0]
    return label


def test_model():
    texts, labels = load_test_data()
    if not texts or not labels:
        print("No test data loaded.")
        sys.exit(1)
    processed_texts = preprocess_texts(texts)
    model, vectorizer = load_model()
    X = vectorizer.transform(processed_texts)
    y_pred = model.predict(X)
    print("Test set classification report:")
    print(classification_report(labels, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia Geographic Text Classifier")
    parser.add_argument('--train', action='store_true', help='Train and save the model')
    parser.add_argument('--test', action='store_true', help='Test the saved model on a test set')
    parser.add_argument('--predict', type=str, help='Predict the class of a given text')
    args = parser.parse_args()
    if args.train:
        train_and_save()
    elif args.test:
        test_model()
    elif args.predict:
        label = predict_label(args.predict)
        print(f"Predicted label: {label}")
    else:
        parser.print_help()
