# img-txt-categorisation-chat
Combining SOTA ML Models for categorising images and their captions, so the user can better search through and organise them guided by a LangChain + StreamLit ChatBot interface.


## Goal:
As a user I wish to get an overview over all the pictures and their descriptions that I have posted on Social Media. Therefore I need to know the sentiment of the image-caption pair (whether it’s negative, neutral or positive) and preferably also the topics, that they belong to, so I can sort and view them based on this. Ideally I could also have conversation with my personal assistant about the pictures, so I can quickly go through them. 

### Goal breakdown:
All models available can be used to solve the task. These could be found on HuggingFace, Spacy, LangChain or other ML Libraries.  

1. Sentiment: Show in practice how you would apply a machine learning model on the captions to extract the sentiment (negative or positive) of them.

2. Topic: Show in practice how you would apply a ML model on the captions to extract the topic(s) that they belong to.

3. Repeat step 1+2 with a ML model extracting information from the images valuable for determining the topic and sentiment (a pretrained Vision Language Model, VLM, could be beneficial)

4. Compare the results from having captions only and then adding the image understanding. 

5. With an LLM in Langchain and Streamlit as the UI, write a Chat User Interface, where the user can have a dialog about their images, where the sentiment and topic is helping them and the chatbot to understand the images. 

## Setup and Data

The data used in this project is a combination of two sources:
- **Images**: [MS COCO 2014: A Step-by-Step Guide to Loading and Visualizing](https://machinelearningspace.com/coco-dataset-a-step-by-step-guide-to-loading-and-visualizing/) - Only validation set is needed for SentiCap.
- **Captions with sentiment**: [SentiCap: Generating Image Descriptions with Sentiments](http://users.cecs.anu.edu.au/~u4534172/senticap.html) and the [Paper with Code and Dataset Description](https://paperswithcode.com/dataset/senticap) 

Example of two COCO images from the 2014 validation set and their captions and sentiment from SentiCap:
![SentiCap Example](example_img_sent_cap.png)

### Setup

Go to your respective folder for the project and run the following commands to download the data and set up the project:

First clone this repo
```bash
git clone https://github.com/maximillian91/img-txt-categorisation-chat.git
cd img-txt-categorisation-chat
```

Then prepare the image data by running the following commands in the terminal:
```bash
mkdir img_data
cd img_data
```

The COCO 2014 validation images and annotations are downloaded and unfolded to the folder `img_data/coco_val2014` and `img_data/coco_ann2014` respectively by running the following commands in the terminal for images:
```bash
wget http://images.cocodataset.org/zips/val2014.zip -O coco_val2014.zip
unzip coco_val2014.zip -d coco_val2014
rm coco_val2014.zip
```
and for annotations:
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -O coco_ann2014.zip
unzip coco_ann2014.zip -d coco_ann2014
rm coco_ann2014.zip
```

### Locate and understand data

#### SentiCap
is already downloaded in the repo, as the original JSON file `txt_data/data/senticap_dataset.json`, which was unfolded to a CSV in `txt_data/data/senticap_dataset.csv`. We suggest using the provided classes in `src/senticap_reader.py` to organise and read the data and get an understanding of it. How to do this is and connect the senticap data to the COCO data is shown in the `src/main.py` file, where you can also inspect two random images and their captions and sentiment by running it. 

#### COCO 2014
The COCO 2014 validation images and annotations are downloaded and unfolded to the folder `img_data/coco_val2014` and `img_data/coco_ann2014` respectively.
The images are in `jpg` format and annotations are in `JSON` format. In the `src/main.py` you can also see how you load the annotations and connect them to the images, even though the coco captions are more used for model prediction in this project. We get the annotations to train and evaluate a sentiment model from the Senticap data if this is needed.
You can get more familiar with the COCO 2014 dataset by checking out this guide for the 2017 dataset - [MS COCO 2017: A Step-by-Step Guide to Loading and Visualizing](https://machinelearningspace.com/coco-dataset-a-step-by-step-guide-to-loading-and-visualizing/).

### Requirements
You need at least Python 3.8 to run this project. [Download it here](https://www.python.org/downloads/).
We suggest using a virtual environment to install the requirements. Find out how [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or just go to the repo folder and run the following commands in the terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
