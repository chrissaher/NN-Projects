import numpy as np
from skimage.color import rgba2rgb
from skimage.transform import resize
import scipy.misc


def generate_noise_image(content_image, img_height, img_width, img_channel, noise_ratio = 0.6):
    
    noise_image = np.random.uniform(-20, 20, (1, img_height, img_width, img_channel)).astype('float32')
    
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image

def reshape_and_normalize_image(image, img_width, img_height):

    if image.shape[-1] == 4:
        image = image[:,:,:3]
    
    image = (resize(image, (img_height,img_width)) * 255).astype(int)
    
    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))
    
    return image

def save_image(path, image):
    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)