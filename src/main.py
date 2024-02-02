import sys
import os
from senticap_reader import SenticapSentence, SenticapReader, SenticapImage

import pandas as pd

from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

VERBOSE = True

def main():
	# handle arguments

	# read data
	current_dir = os.path.dirname(os.path.realpath(__file__))
	senticap_data_dir = os.path.join(current_dir, "..", "txt_data", "data")
	coco_data_dir = os.path.join(current_dir, "..", "img_data", "coco_val2014", "val2014")

	senticap_data_json_path = os.path.join(senticap_data_dir, "senticap_dataset.json")
	senticap_data_csv_path = os.path.join(senticap_data_dir, "senticap_dataset.csv")

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
	senticap_df = pd.read_csv(senticap_data_csv_path)

	# print data
	print(senticap_df.head())
	print(senticap_df.describe())
	print(senticap_df.info())
	print(senticap_df.columns.values)

	# Read a single image from img_data
	img_filename = senticap_df["filename"][0]
	print("img_filename: ", img_filename)

	# Load image
	img_path = os.path.join(coco_data_dir, img_filename)
	img = mpimg.imread(img_path)
	# imgplot = plt.imshow(img)
	# plt.show()

	num_img_show = 2

	fig, ax = plt.subplots(1, num_img_show, figsize=(10*num_img_show, 12))
	[a.axis('off') for a in ax.flatten()]

	for i in range(num_img_show):
		img_filename = list_of_image_data_objects[i].getFilename()
		img_path = os.path.join(coco_data_dir, img_filename)
		img = mpimg.imread(img_path)
		ax[i].imshow(img)

		# Get the sentences
		sentences = list_of_image_data_objects[i].getSentences()
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



	# write data
	# count

if __name__ == "__main__":
	main()