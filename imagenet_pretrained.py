from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np 
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, 
    help = "path to the image")
ap.add_argument('-m', '--model', required = True , default = "vgg16",
    help = "name of pretrained model")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

if args['model'] not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
        "be a key in the `MODELS` dictionary")

inputShape = (224 , 224)
preprocess = imagenet_utils.preprocess_input

if args['model'] in ('xception', 'inception'):
    inputShape = (299, 299)
    preprocess = preprocess_input

print('[INFO] loading {}...'.format(args['model']))
Network = MODELS[args['model']]
model = Network(weights = 'imagenet')

print('[INFO] loading image and preprocessing')
image = load_img(args['image'], target_size = inputShape)
image = img_to_array(image)

image = np.expand_dims(image , axis = 0)
image = preprocess(image)

print('[INFO] Classifying Image')

preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

for (i , (imageID , label , prob)) in enumerate(P[0]):
    print('[INFO] {}. {} --- {:.2f}%'.format(i + 1, label , prob*100))

image = cv2.imread(args['image'])
(id , label , prob) = P[0][0]
cv2.putText(image , "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    0.8, (0, 255, 0), 2)
cv2.imshow('Preds', image)
cv2.waitKey(0)