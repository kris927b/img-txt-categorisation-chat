import typer
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import ViltProcessor, DefaultDataCollator, ViltForQuestionAnswering
from datasets import load_dataset, Dataset
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


def finetune_multimodal(train_path: str, val_path: str, img_path: str):
    train = pd.read_csv(train_path, header=0).rename({"is_positive_sentiment": "label"}, axis=1)
    train = train[~train["filename"].isin(["COCO_val2014_000000064332.jpg", "COCO_val2014_000000566596.jpg"])] # Balck and White image, NOT SUPPORTED BY ViLT
    train.to_csv(f"{train_path}.edit", index=False)
    val = pd.read_csv(val_path, header=0).rename({"is_positive_sentiment": "label"}, axis=1)
    val = val[~val["filename"].isin(["COCO_val2014_000000064332.jpg", "COCO_val2014_000000566596.jpg"])]
    val.to_csv(f"{val_path}.edit", index=False)
    del val, train

    dataset = load_dataset("csv", data_files={"train": f"{train_path}.edit", "val": f"{val_path}.edit"}, cache_dir="/content")

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    def preprocess(data):
        img_paths = data["filename"]
        images = [Image.open(f"{img_path}/{path}") for path in img_paths]
        captions = data["caption"]

        embedding = processor(images, captions, padding=True, truncation=True, return_tensors="pt")#.to('cuda:0')

        for k,v in embedding.items():
            embedding[k] = v.squeeze()

        targets = []
        for label in data["label"]:
            target = torch.zeros(2)
            target[label] = 1.0
            targets.append(target)

        embedding["label"] = targets

        return embedding

    processed_data = dataset.map(preprocess, batched=True, remove_columns=["filename", "caption"], batch_size=16, load_from_cache_file=True) # , cache_file_names={"val": "", "train": ""})
    
    # data_collator = DefaultDataCollator() # tokenizer=processor)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.argmax(labels, axis=1)
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)
    #model = model.to('cuda:0')
    for name, param in model.named_parameters():
        if name.startswith("vilt"):
            param.requires_grad = False


    training_args = TrainingArguments(
        output_dir="senticap-multimodal",
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
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["val"],
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
        output_dir="unimodal-senticap",
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
    test = pd.read_csv(test_path, header=0).rename({"is_positive_sentiment": "label"}, axis=1)

    accuracy = evaluate.load("accuracy")

    classifier = pipeline("sentiment-analysis", model="/home/kristian/Documents/img-txt-categorisation-chat/senticap-model/checkpoint-2770")

    results = classifier(test.caption.values.tolist())
    preds = [LABEL2ID[pred["label"]] for pred in results]

    acc = accuracy.compute(predictions=preds, references=test.label.values.tolist())

    print(acc)
    pass

if __name__ == "__main__":
    typer.run(evaluate_bulk)