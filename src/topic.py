import typer
import pandas as pd
import numpy as np
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from bertopic import BERTopic

def main(train_path: str, val_path: str, test_path: str):
    ## Load train data
    train_data = pd.read_csv(train_path, header=0)
    # print(train_data.head())

    ## Load model
    model = BERTopic()

    ## Fit Transform BERTopic model
    model = model.fit(train_data.caption.values.tolist())

    ## Load Val data
    val_data = pd.read_csv(val_path, header=0)
    # print(val_data.head())

    topics, probs = model.transform(val_data.caption.values.tolist())
    val_data["topic"] = [model.get_topic_info(i)["Name"] for i in topics]

    print(val_data.head())

if __name__ == "__main__":
    typer.run(main)