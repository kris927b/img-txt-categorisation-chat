import typer
import pandas as pd
import numpy as np
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_cm(cm):
  classes = ['negative','positive']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sns.heatmap(df_cm, annot = True, fmt='g')
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')


def main(text_path: str, img_path: str = ""):
    ## Load text data
    data = pd.read_csv(text_path, header=0)
    print(data.head())

    ## Load sentiment analysis pipeline
    senti = pipeline("text-classification")

    ## Run texts through pipeline
    res = senti(data.caption.values.tolist())
    
    ## Convert labels to ints
    preds = [1 if r["label"].lower() == "positive" else 0 for r in res]
    data["preds"] = preds
    data["validation"] = data["preds"] == data["is_positive_sentiment"]

    ## Plot confusion matrix
    cm = confusion_matrix(data["is_positive_sentiment"], data["preds"])
    plot_cm(cm)
    plt.show()


if __name__ == "__main__":
    typer.run(main)