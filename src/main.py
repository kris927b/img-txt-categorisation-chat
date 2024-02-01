import sys
import os
from senticap_reader import SenticapSentence, SenticapReader, SenticapImage

import pandas as pd

from pprint import pprint

def main():
	# handle arguments

	# read data
	current_dir = os.path.dirname(os.path.realpath(__file__))
	senticap_data_dir = os.path.join(current_dir, "..", "txt_data", "data")
	
	senticap_data_json_path = os.path.join(senticap_data_dir, "senticap_dataset.json")
	senticap_data_csv_path = os.path.join(senticap_data_dir, "senticap_dataset.csv")

	senticap_data = SenticapReader(senticap_data_json_path)

	list_of_image_data_objects = senticap_data.images

	pprint(list_of_image_data_objects[0].getSentences()[0].getTokens())

	# Load csv data
	senticap_df = pd.read_csv(senticap_data_csv_path)

	# print data
	print(senticap_df.head())
	print(senticap_df.describe())
	print(senticap_df.info())
	print(senticap_df.columns.values)

	# Read a single image from img_data
	img_filename = senticap_df["filename"][0]
	

	# write data
	# count

if __name__ == "__main__":
	main()