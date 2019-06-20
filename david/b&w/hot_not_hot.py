#!/usr/bin/env python3
import pickle
import argparse
import numpy as np
from cv2 import imread, resize
def process_image(image):
    '''
    loads image
    '''
    img = 1-imread(image, 0)            #loading in the image in black and white
    img = resize(img,(360, 360))        #reshaping to be 360 by 360 pixles
    img = img.reshape(-1)               #reshaping to 1 dimension
    return img
parser = argparse.ArgumentParser()
#args
parser.add_argument("image", help="image to test")
parser.add_argument("model", help="model to use")
args=parser.parse_args()
if not args.image.endswith('.jpg'):
    print("must be .jpg image")
    quit()
# load the model from disk
loaded_model = pickle.load(open(args.model, 'rb'))
processed_image = process_image(args.image)
print(loaded_model.predict(processed_image.reshape(1,-1)))
