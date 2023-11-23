import os
import glob
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import load_img

class ImageUtil:
    '''
  Utility class
  '''
    def plot_img(img_path: str, img_ht: int = 224, img_wd: int = 224) -> None:
        try:
            raw_input = load_img(img_path, target_size=(img_ht, img_wd))
            plt.imshow(raw_input)
            plt.show()
        except Exception as e:
            raise (e)

    def get_path_list_from_root(root_directory: str) -> list:
        '''
    Given a root directory returns list of paths of image files
    :return: image file paths (List)
    '''
        try:
            image_paths = []
            # Recursively traverse the directory and its subdirectories
            for root, _, files in os.walk(root_directory):
                files = files[:10]
                for file in files:
                    if ((file.lower().endswith('.jpg')) or (file.lower().endswith('.jpeg')) or (
                            file.lower().endswith('.png'))):
                        image_paths.append(os.path.join(root, file))
            return image_paths
        except Exception as e:
            print('Error in walking root directory')
            raise e

    def retrieve_most_similar_products(given_img: str, similarity_df: pd.DataFrame, img_ht: int = 224,
                                       img_wd: int = 224,
                                       n_similar_imgs: int = 3) -> None:
        '''
        Given an image and similarity matrix returns requested similar images
      :param similarity_df: Similarity score dataframe
      :param img_ht: height of image
      :param img_wd: width of given image
      :param n_similar_imgs: number of results to be returned
      '''
        # Drop the input image from column list if exists
        if given_img in similarity_df.columns.values:
            similarity_df.drop(columns=[given_img], inplace=True)

        # Sort the closest images decreasing order of similarity score
        closest_imgs = similarity_df.loc[given_img, :].sort_values(ascending=False).head(n_similar_imgs)

        for img, score in closest_imgs.items():
            ImageUtil.plot_img(img, img_ht, img_wd)
            print("similarity score:", score)

        return list(closest_imgs.keys())

    def unzip_images(dir_name: str, extension: str = '.zip') -> None:
        '''
        Given a directory with zip file, unzips the contents(images) to same folder.
        Will search for file(s) with extension given and unzip all of them.
        Make sure there are only zip files that are needed and nothing else.
        :param extension: zip is default extension
        :param dir_name: Full path to directory containing zip file
        '''
        try:
            os.chdir(dir_name)  # change directory from working dir to dir with files
            for item in os.listdir(dir_name):  # loop through items in dir
                if item.endswith(extension):  # check for ".zip" extension
                    file_name = os.path.abspath(item)  # get full path of files
                    zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
                    zip_ref.extractall(dir_name)  # extract file to dir
                    zip_ref.close()  # close file
        except Exception as e:
            print("Error in unzipping images")
            raise e
        
    def get_filepath_from_dir(dir:str,extension:list=[])->str:
        """Given a directory returns first image in the directory

        Args:
            dir (str): Directory to search

        Returns:
            str: Full path of image in dir
        """
        files = glob.glob(os.path.join(dir, "*.*"))
        for file in files:
            if os.path.splitext(file)[1] in extension:
                return file
