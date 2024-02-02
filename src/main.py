import sys
import os
from senticap_reader import SenticapSentence, SenticapReader, SenticapImage

import pandas as pd
import json

from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random

VERBOSE = True

# A function for loading and inspecting the data, img, captions and sentiments
def load_and_inspect_data(
		coco_data_dir,
		senticap_data_dir,
		senticap_filename="senticap_dataset.json",
		num_imgs_show=2,
		pick_random=False,
		show_img=True):

	# Load data
	senticap_data_json_path = os.path.join(senticap_data_dir, senticap_filename)
	senticap_data = SenticapReader(senticap_data_json_path)

	list_of_image_data_objects = senticap_data.images

	# Get the first image data object and first sentence
	first_img_sentence = list_of_image_data_objects[0].getSentences()[0]

	pprint(first_img_sentence.getTokens())

	# Define sentinment polarity to string dict
	sentiment_polarity_dict = {
		first_img_sentence.NEGATIVE_SENTIMENT: "Negative",
		first_img_sentence.POSITIVE_SENTIMENT: "Positive"
	}

	# Load csv data
	# senticap_df = pd.read_csv(senticap_data_csv_path)

	# print data
	# print(senticap_df.head())
	# print(senticap_df.describe())
	# print(senticap_df.info())
	# print(senticap_df.columns.values)

	# Read a single image from img_data
	# img_filename = senticap_df["filename"][0]
	# print("img_filename: ", img_filename)

	# Load image
	# img_path = os.path.join(coco_data_dir, img_filename)
	# img = mpimg.imread(img_path)
	# imgplot = plt.imshow(img)
	# plt.show()

	if not show_img:
		return senticap_data
	else:
		fig, ax = plt.subplots(1, num_imgs_show, figsize=(10*num_imgs_show, 12))
		[a.axis('off') for a in ax.flatten()]

		# Either pick `num_imgs_show` random images or just from the top
		num_imgs_total = len(list_of_image_data_objects)
		if pick_random:
			picked_img_inds = random.sample(range(num_imgs_total), num_imgs_show)
		else:
			picked_img_inds = range(num_imgs_show)

		# Loop over and show all picked images
		cnt = 0
		for i, img_ind in enumerate(picked_img_inds):
			print(f"Showing image {img_ind}, which is {i+1} out of {num_imgs_show}")
			img_obj = list_of_image_data_objects[img_ind]
			img_filename = img_obj.getFilename()
			img_path = os.path.join(coco_data_dir, img_filename)
			img = mpimg.imread(img_path)
			ax[i].imshow(img)

			# Get the sentences
			sentences = img_obj.getSentences()
			txt_list = [sent.getRawsentence() for sent in sentences]
			sentiment_list = [sentiment_polarity_dict[sent.getSentimentPolarity()] for sent in sentences]
			txt_sentiment_list = [f"{sentiment}: {txt}" for txt, sentiment in zip(txt_list, sentiment_list)]
			txt_sentiment_str = "\n".join(txt_sentiment_list)
			
			# Show title on top of image
			# ax[i].set_title(img_filename)

			# Show sentiment and captions on bottom of image
			ax[i].text(0, -0.5, txt_sentiment_str, ha="left", wrap=True, color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

			if VERBOSE:
				for sent in sentences:
					print(sent.getTokens())
					print(sent.getSentimentPolarity())
					print(sent.getSentimentPolarity()==sent.POSITIVE_SENTIMENT)
					print(sent.getSentimentPolarity()==sent.NEGATIVE_SENTIMENT)
					print("")

		plt.show()

		return senticap_data

def main():
	# handle arguments

	# read data
	current_dir = os.path.dirname(os.path.realpath(__file__))
	senticap_data_dir = os.path.join(current_dir, "..", "txt_data", "data")
	coco_img_data_dir = os.path.join(current_dir, "..", "img_data", "coco_val2014", "val2014")
	coco_ann_data_dir = os.path.join(current_dir, "..", "img_data", "coco_ann2014", "annotations")

	senticap_data_json_path = os.path.join(senticap_data_dir, "senticap_dataset.json")
	senticap_data_csv_path = os.path.join(senticap_data_dir, "senticap_dataset.csv")

	coco_cap_data_path = os.path.join(coco_ann_data_dir, "captions_val2014.json")

	# Load and inspect data
	senticap_data = load_and_inspect_data(
		coco_img_data_dir,
		senticap_data_dir,
		senticap_filename="senticap_dataset.json",
		num_imgs_show=2,
		pick_random=True
	)

	# Load the captions from the coco dataset
	with open(coco_cap_data_path, "r") as f:
		coco_cap_data = json.load(f)
	
	coco_cap_data_ann = coco_cap_data["annotations"]
	coco_cap_data_img = coco_cap_data["images"]
	
	# Create a dataframe from the coco captions
	coco_cap_df = pd.DataFrame(coco_cap_data_ann)

	print(coco_cap_df.head())
	print("...")
	print(coco_cap_df.tail())
	print(coco_cap_df.describe())
	print(coco_cap_df.info())
	print(coco_cap_df.columns.values)


	# Connecting senticap captions to sentiments with an ML model



	# write data
	# count

if __name__ == "__main__":
	main()