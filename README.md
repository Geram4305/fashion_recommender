## Fashion Recommender Model with VGG16 and Cosine Similarity

## Overview

This fashion recommender model is designed to recommend similar attires based on an input image. 
It utilizes the VGG16 deep learning model for feature extraction and calculates similarity scores 
between images using cosine similarity. 


## Features

- Extracts high-level features from input fashion images using a pre-trained VGG16 model.
- Calculates similarity scores between the input image and the entire dataset using cosine similarity.
- Recommends fashion attires that are visually similar to the input image.

## Dataset

The dataset is taken from Kaggle . The dataset contains a variety of images of different attires like pants, shirts,
etc. A subset of these images are used to train and provide results in this project due space and time constraints.

## Usage

## How to run:
1. Install docker 
2. In cmd, run the command 'docker build -t <image_name> .' after replacing <image_name> with
your own image name. This creates an image that can start up the application
3. Then run 'docker run --rm -it <image_name>.
