import os
import numpy as np
from keras.applications import vgg16
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model,load_model
from keras.preprocessing.image import load_img, img_to_array


class SimiliarityModel:
    '''
  Class to handle methods which help in creating similarity model
  '''

    def __init__(self, model_path:str,img_ht: int = 224, img_wd: int = 224) -> None:
        self.img_height = img_ht
        self.image_width = img_wd
        self.feat_model = self.__load_vgg_model(model_path)

    # Set parameters for img processing and extracts vgg features for images in given paths.
    def extract_vgg_features(self, image_paths: list) -> list:
        all_images = []
        for img in image_paths:
            # Convert the PIL image from (width, height, channel formant) to a numpy array((height, width, channel))
            raw_inputs = load_img(img, target_size=(self.img_height, self.image_width))
            numpy_image = img_to_array(raw_inputs)

            # Prepare the image to be used with the VGG model. Add an extra dimension to the NumPy array to make it
            # compatible with the expected input shape of the VGG model. The resulting shape is typically (
            # batch_size, height, width, channels)
            image_batch = np.expand_dims(numpy_image, axis=0)
            all_images.append(image_batch)
            

        # Vertically stack (concatenate) the images stored in the all_images list. The result is a single NumPy array
        # where each row corresponds to an image. The np.vstack function is used to stack arrays vertically along
        # their first axis (rows). This is useful to process multiple images together, such as when preparing a batch
        # of images for the VGG model.
        images = np.vstack(all_images)

        # The preprocess_input function is typically used to preprocess images before feeding them into the model.
        # These preprocessing steps often include mean centering, scaling, or other transformations that are specific
        # to the model's training data.
        processed_imgs = preprocess_input(images.copy())

        # extract the images features
        imgs_features = self.feat_model.predict(processed_imgs)
        print("features successfully extracted!")
        return imgs_features

    def __load_vgg_model(self,model_path:str):
        # loading the VGG model
        # vgg_model = vgg16.VGG16(weights='imagenet')
        vgg_model = load_model(model_path)

        # Eliminate the final layers to obtain image features rather than prediction outcomes
        return Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
