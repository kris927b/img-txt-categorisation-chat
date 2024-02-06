import typer
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from bertopic.backend import MultiModalBackend

def unimodal_topics(train_path: str, val_path: str): # , test_path: str):
    ## Load train data
    train_data = pd.read_csv(train_path, header=0)
    # print(train_data.head())

    ## Load model
    model = BERTopic()

    ## Fit Transform BERTopic model
    topics, _ = model.fit_transform(train_data.caption.values.tolist())
    train_data["topic"] = [model.get_topic_info(i)["Name"].values for i in topics]

    train_data.to_csv(f"{train_path}.topics")

    ## Load Val data
    val_data = pd.read_csv(val_path, header=0)
    # print(val_data.head())

    topics, _ = model.transform(val_data.caption.values.tolist())
    val_data["topic"] = [model.get_topic_info(i)["Name"].values for i in topics]
    val_data.to_csv(f"{val_path}.topics")


def multimodal_topics(train_path: str, val_path: str): # , test_path: str):
    ## Load train data
    train_data = pd.read_csv(train_path, header=0)
    train_data_img = train_data.filename.apply(lambda x: f"/home/kristian/Documents/img-txt-categorisation-chat/img_data/coco_val2014/val2014/{x}").values.tolist()
    train_data_txt = train_data.caption.values.tolist()
    # print(train_data.head())
    
    ## Load embedder
    embedder = MultiModalBackend('clip-ViT-B-32')

    ## Generate img_txt train embeddings
    img_txt_embeds = embedder.embed(train_data_txt, train_data_img)

    ## Load model
    model = BERTopic()

    ## Fit Transform BERTopic model
    topics, _ = model.fit_transform(train_data_txt, img_txt_embeds)
    train_data["topic"] = [model.get_topic_info(i)["Name"].values for i in topics]

    train_data.to_csv(f"{train_path}.multi.topics")

    ## Load Val data
    val_data = pd.read_csv(val_path, header=0)
    val_data_img = val_data.filename.apply(lambda x: f"/home/kristian/Documents/img-txt-categorisation-chat/img_data/coco_val2014/val2014/{x}").values.tolist()
    val_data_txt = val_data.caption.values.tolist()
    # print(val_data.head())

    ## Generate img_txt val embeddings
    val_img_txt_embeds = embedder.embed(val_data_txt, val_data_img)

    ## Create topics for val data
    topics, _ = model.transform(val_data_txt, val_img_txt_embeds)
    val_data["topic"] = [model.get_topic_info(i)["Name"].values for i in topics]
    val_data.to_csv(f"{val_path}.multi.topics")

if __name__ == "__main__":
    typer.run(multimodal_topics)