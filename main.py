from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from similarity_model import *
from utils import *
from configs import *


def predict_img_outside_dataset(test_img_dir:str, imgs_dir:str,model_dir:str,n_similar_imgs:int=3):
    #Get image full path from given dir
    new_image_path = ImageUtil.get_filepath_from_dir(test_img_dir,extension=[".jpg",".jpeg",".png"])
    
    #Unzip the all images in image directory
    ImageUtil.unzip_images(imgs_dir,extension=".zip")

    # Instantiate the similarity model class with the dimension parameters
    model_path = ImageUtil.get_filepath_from_dir(model_dir,extension=[".h5"])
    sim_inst = SimiliarityModel(model_path=model_path,img_ht=224, img_wd=224)
    new_img_features = sim_inst.extract_vgg_features([new_image_path]).flatten()

    # Create features for our dataset to compare with
    image_paths = ImageUtil.get_path_list_from_root(imgs_dir)
    train_img_features = sim_inst.extract_vgg_features(image_paths)

    # Calculate cosine similarity with existing images
    similarities = cosine_similarity(new_img_features.reshape(1, -1), train_img_features)

    # Convert the similarities to a DataFrame
    similarities_df = pd.DataFrame(similarities, columns=image_paths, index=[new_image_path])

    # Plot the inpout image
    print('Input image:')
    ImageUtil.plot_img(new_image_path, img_ht=224, img_wd=224)

    # Retrieve and display the most similar products
    print('-------------------------------------------------------------------------------')
    print('Similar results:')
    return ImageUtil.retrieve_most_similar_products(new_image_path, similarities_df,n_similar_imgs=n_similar_imgs)


if __name__ == '__main__':
    img_dir = os.path.join(os.getcwd(),"data")
    test_img_dir = os.path.join(os.getcwd(),"test_image")
    model_dir = os.path.join(os.getcwd(),"vgg_model")
    image_file_paths = predict_img_outside_dataset(test_img_dir, img_dir, model_dir, n_similar_imgs=NUMBER_SIMILAR_RESULTS)
    for file in image_file_paths:
        img = Image.open(file)
        img.show()

