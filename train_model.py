import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")
nltk.download("punkt")

def clean_text(text):
    text = re.sub(r'\W', ' ', text).lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stopwords.words("english")]
    return " ".join(text)

def train_model():
    df = pd.read_csv("dataset/cyberbullying_data.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, "models/cyberbullying_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_model()
