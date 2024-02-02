# img-txt-categorisation-chat
Combining SOTA ML Models for categorising images and their captions, so the user can better search through and organise them guided by a LangChain + StreamLit ChatBot interface.


## Goal:
As a user I wish to get an overview over all the pictures and their descriptions that I have posted on Social Media. Therefore I need to know the sentiment of the image-caption pair (whether it’s negative, neutral or positive) and preferably also the topics, that they belong to, so I can sort and view them based on this. Ideally I could also have conversation with my personal assistant about the pictures, so I can quickly go through them. 

### Goal breakdown:
All models available can be used to solve the task. These could be found on HuggingFace, Spacy, LangChain or other ML Libraries.  

1. Sentiment: Show in practice how you would apply a machine learning model on the captions to extract the sentiment (negative, neutral or positive) of them.

2. Topic: Show in practice how you would apply a ML model on the captions to extract the topic(s) that they belong to.

3. Repeat step 1+2 with a ML model extracting information from the images valuable for determining the topic and sentiment (a pretrained Vision Language Model, VLM, could be beneficial)

4. Compare the results from having captions only and then adding the image understanding. 

5. With an LLM in Langchain and Streamlit as the UI, write a Chat User Interface, where the user can have a dialog about their images, where the sentiment and topic is helping them and the chatbot to understand the images. 

## Setup and Data

### Data:
The data used in this project is a combination of two sources:
- **Images**: [MS COCO 2014: A Step-by-Step Guide to Loading and Visualizing](https://machinelearningspace.com/coco-dataset-a-step-by-step-guide-to-loading-and-visualizing/) - Only validation set is needed for SentiCap.
- **Captions with sentiment**: [SentiCap: Generating Image Descriptions with Sentiments](http://users.cecs.anu.edu.au/~u4534172/senticap.html) and the [Paper with Code and Dataset Description](https://paperswithcode.com/dataset/senticap) 

Example of two COCO images from the 2014 validation set and their captions and sentiment from SentiCap:
![SentiCap Example](example_img_sent_cap.png)

#### Data loading and handling:
- SentiCap is already downloaded and unfolded to CSV in the repo under `txt_data/data/senticap_dataset.json` and `txt_data/data/senticap_dataset.csv`
- Follow this tutorial to get the MS COCO dataset from 2014 (Only validation is needed for SentiCap) - COCO Dataset: A Step-by-Step Guide to Loading and Visualizing 
- Connect the captions and labels from SentiCap and MSCOCO to the final dataset we are gonna use. References to the img file names are in the senticap_dataset.json and a reading script in the senticap_dataset folder.

#### Bash code for downloading validation images in COCO 2014:
```bash
wget http://images.cocodataset.org/zips/val2014.zip -O coco_val2014.zip
unzip coco_val2014.zip -d coco_val2014
rm coco_val2014.zip
```

#### Bash code for downloading the annotations for COCO 2014:
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -O coco_ann2014.zip
unzip coco_ann2014.zip -d coco_ann2014
rm coco_ann2014.zip
```