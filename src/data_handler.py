import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import string
from sentence_transformers import SentenceTransformer

from . import config

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def wget_data():
    dataset_url = config.DATASET_URL
    file_destination = config.RAW_DATASET_PATH
    res = requests.get(dataset_url)
    if res.status_code == 200:  # http 200 means success
        with open(file_destination, 'wb') as file_handle:  # wb means Write Binary
            file_handle.write(res.content)


def get_cleared_text(text):
    table = text.maketrans(
        dict.fromkeys(string.punctuation))

    words = word_tokenize(
        text.lower().strip().translate(table))

    words = [word for word in words if word not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]
    return " ".join(lemmed)


def clean_data():
    raw_dataset_path = config.RAW_DATASET_PATH
    if not os.path.isfile(raw_dataset_path):
        wget_data()
    dataset = pd.read_csv(raw_dataset_path, sep=";")
    dataset['isHate'] = np.round(dataset.isHate).astype('Int64')
    dataset['comment'] = dataset['comment'].apply(lambda x: get_cleared_text(x))
    dataset.to_csv(config.CLEAN_DATASET_PATH, index=False, sep=";")


def preprocess_data():
    clean_dataset_path = config.CLEAN_DATASET_PATH
    if not os.path.isfile(clean_dataset_path):
        clean_data()
    dataset = pd.read_csv(clean_dataset_path, sep=";")
    encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    encoded_texts = []
    for text in dataset["comment"]:
        encoded_texts.append(encoder_model.encode(text))
    dataset["comment"] = encoded_texts
    train, test = train_test_split(dataset, test_size=0.3)
    val, test = train_test_split(dataset, test_size=0.5)

    train.to_csv(config.TRAIN_DATASET_PATH, index=False, sep=";")
    test.to_csv(config.TEST_DATASET_PATH, index=False, sep=";")
    val.to_csv(config.VAL_DATASET_PATH, index=False, sep=";")


def get_embedded_data(path):
    if not os.path.isfile(path):
        preprocess_data()
    dataset = pd.read_csv(path, sep=";")
    X = dataset["comment"].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
    return np.vstack(X.values), dataset['isHate'].to_numpy()
