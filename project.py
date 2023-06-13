import string

import mlflow
import mlflow.sklearn
import nltk
import numpy as np
import pandas as pd
import plac
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
wnl = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = "".join([wnl.lemmatize(word) for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def vectorize(sentence, w2v_model: Word2Vec):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(100)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)


def main():
    data = pd.read_csv("data/train.tsv", sep="\t")

    X_train, X_test, y_train, y_test = train_test_split(
        data["Phrase"], data["Sentiment"], test_size=0.2
    )

    X_train = X_train.apply(preprocess)
    X_test = X_test.apply(preprocess)
    sentences = [sentence.split() for sentence in X_train]
    w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

    X_train = np.array([vectorize(sentence, w2v_model) for sentence in X_train])
    X_test = np.array([vectorize(sentence, w2v_model) for sentence in X_test])

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = clf.predict(X_test)

        # calculate metrics
        # Accuracy, Precision, Recall, F1 score,
        # from sklearn.metrics import precision_recall_curve
        # eewentualnie ROC AUC
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1score = f1_score(y_test, y_pred, average="macro")

        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", recall)
        print("F1-score:", f1score)

        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)

        print(
            f"LogisticRegression: RMSE={rmse}, MAE={mae}, R2={r2score}, acc={acc}, f1-score: {f1score}"
        )

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2score", r2score)

        mlflow.log_metric("acc", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1score)

        mlflow.sklearn.log_model(clf, "model")
        print(f"Model saved in run {mlflow.active_run().info.run_uuid}")


if __name__ == "__main__":
    plac.call(main)
