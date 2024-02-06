import typer
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import ViltProcessor, DefaultDataCollator, ViltForQuestionAnswering
from datasets import Dataset
import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}

def plot_cm(cm):
  classes = ['negative','positive']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sns.heatmap(df_cm, annot = True, fmt='g')
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')


def finetune_multimodal(train_path: str, val_path: str):
    train = pd.read_csv(train_path, header=0).rename({"is_positive_sentiment": "label"}, axis=1)
    train = train[~train["filename"].isin(["COCO_val2014_000000064332.jpg", "COCO_val2014_000000566596.jpg"])] # Balck and White image, NOT SUPPORTED BY ViLT
    train.filename = train.filename.apply(lambda x: f"/home/kristian/Documents/img-txt-categorisation-chat/img_data/coco_val2014/val2014/{x}")
    train_data = Dataset.from_pandas(train.sample(n=1000))
    val = pd.read_csv(val_path, header=0).rename({"is_positive_sentiment": "label"}, axis=1)
    val = val[~val["filename"].isin(["COCO_val2014_000000064332.jpg", "COCO_val2014_000000566596.jpg"])]
    val.filename = val.filename.apply(lambda x: f"/home/kristian/Documents/img-txt-categorisation-chat/img_data/coco_val2014/val2014/{x}")
    val_data = Dataset.from_pandas(val.sample(n=250))

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    def preprocess(data):
        img_paths = data["filename"]
        images = [Image.open(path) for path in img_paths]

        # images = Image.open(data["filename"])
        captions = data["caption"]

        try:
            embedding = processor(images, captions, padding="max_length", truncation=True, return_tensors="pt")
        except ValueError as e:
            print(data["filename"])
            exit()

        for k,v in embedding.items():
            embedding[k] = v.squeeze()

        targets = []
        for label in data["label"]:
            target = [0.0,0.0]
            target[label] = 1.0
            targets.append(target)
        
        embedding["labels"] = targets

        return embedding

    processed_train = train_data.map(preprocess, batched=True, remove_columns=["filename", "caption"])
    processed_val = val_data.map(preprocess, batched=True, remove_columns=["filename", "caption"])

    data_collator = DefaultDataCollator()

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.argmax(labels, axis=1)
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)

    for name, param in model.named_parameters():
        if name.startswith("vilt"):
            param.requires_grad = False
        

    training_args = TrainingArguments(
        output_dir="senticap-multimodal",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def finetune_unimodal(train_path: str, val_path: str):
    train = pd.read_csv(train_path, header=0).rename({"is_positive_sentiment": "label"}, axis=1)
    train_data = Dataset.from_pandas(train)
    val = pd.read_csv(val_path, header=0).rename({"is_positive_sentiment": "label"}, axis=1)
    val_data = Dataset.from_pandas(val)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    for name, param in model.named_parameters():
        if name.startswith("distilbert"):
            param.requires_grad = False

    def preprocess(data):
        return tokenizer(data["caption"], truncation=True)

    tokenized_train = train_data.map(preprocess, batched=True)
    tokenized_val = val_data.map(preprocess, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def compute_sentiment(text: str, img: bool = False):
    classifier = pipeline("sentiment-analysis", model="/home/kristian/Documents/img-txt-categorisation-chat/senticap-model/checkpoint-2770")

    result = classifier(text)

    print(result)


def evaluate_bulk(test_path: str, img: bool = False):
    
    pass

if __name__ == "__main__":
    typer.run(finetune_multimodal)