import os
import pandas as pd


def test_dataset_existeix():
    assert os.path.exists("data\\ciclistes.csv")


def test_columnes():
    df = pd.read_csv("data\\ciclistes.csv")
    assert "tp" in df.columns and "tb" in df.columns


def test_model_guardat():
    assert os.path.exists("model\\clustering_model.pkl")
