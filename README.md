# img-txt-categorisation-chat
Combining SOTA ML Models for categorising images and their captions, so the user can better search through and organise them guided by a LangChain + StreamLit ChatBot interface.


## Goal:
As a user I wish to get an overview over all the pictures and their descriptions that I have posted on Social Media. Therefore I need to know the sentiment of the image-caption pair (whether it’s negative, neutral or positive) and preferably also the topics, that they belong to, so I can sort and view them based on this. Ideally I could also have conversation with my personal assistant about the pictures, so I can quickly go through them. 

### Goal breakdown:
1. Sentiment: Show in practice how you would apply a machine learning model on the captions to extract the sentiment (negative, neutral or positive) of them.

2. Topic: Show in practice how you would apply a ML model on the captions to extract the topic(s) that they belong to.

3. Repeat step 1+2 with a ML model extracting information from the images valuable for determining the topic and sentiment (pretrained VLM could be beneficial)

4. Compare the results from having captions only and then adding the image understanding. 

5. With an LLM in Langchain and Streamlit as the UI, write a Chat User Interface, where the user can have a dialog about their images, where the sentiment and topic is helping them and the chatbot to understand the images. 


## Setup and Data

### Data:
SentiCap + MSCOCO_2014_val: Papers with Code - SentiCap Dataset + COCO Dataset: A Step-by-Step Guide to Loading and Visualizing  

#### Data loading and handling:
- Download SentiCap here: SentiCap: Generating Image Descriptions with Sentiments 
- Follow this tutorial to get the MS COCO dataset from 2014 (Only validation is needed for SentiCap) - COCO Dataset: A Step-by-Step Guide to Loading and Visualizing 
- Connect the captions and labels from SentiCap and MSCOCO to the final dataset we are gonna use. References to the img file names are in the senticap_dataset.json and a reading script in the senticap_dataset folder.

#### Bash code for downloading MSCOCO_2014_val:
```bash
wget http://images.cocodataset.org/zips/val2017.zip -O coco_val2017.zip
unzip coco_val2017.zip -d coco_val2017
```