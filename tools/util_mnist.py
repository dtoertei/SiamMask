import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import colorsys
import imghdr
import random

# preprocess the segmented blob (in it's bbox) to MNIST CNN
def preprocess_mnist(input_img, input_shape):

	dim = (input_shape[0], input_shape[1])

	# first, resize while keeping aspect ratio
	h1 = dim[1] * (input_img.shape[0]/input_img.shape[1])
	w2 = dim[0] * (input_img.shape[1]/input_img.shape[0])
	output = []
	if( h1 <= dim[0]):
		output = cv2.resize(input_img, (dim[1],int(h1)), interpolation = cv2.INTER_AREA)
	else:
		output = cv2.resize(input_img, (int(w2),dim[1]), interpolation = cv2.INTER_AREA)

	top = (dim[0] -output.shape[0]) / 2;
	down = (dim[0] -output.shape[0]+1) / 2;
	left = (dim[1] - output.shape[1]) / 2;
	right = (dim[1] - output.shape[1]+1) / 2;

	img = cv2.copyMakeBorder(output, int(top), int(down), int(left), int(right), cv2.BORDER_CONSTANT, value=(0,0,0))

	# second, make it single channel
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# third, reshape into tensor and scale
	# tensor: num_batches, h, w, c
	gray = np.reshape(gray, (1, input_shape[0], input_shape[1], 1))
	gray = gray/255


	return gray


# return a binary image array where "1"s denote skin non-presence
def extractSkinMask(image):

	img = image.copy()
	# Converting from BGR Colours Space to HSV
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Defining HSV Threadholds
	lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
	upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

	# Single Channel mask,denoting presence of colours in the about threshold
	skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

	# Cleaning up mask using Gaussian Filter
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skinMask[skinMask > 0] = 2
	skinMask[skinMask == 0] = 1
	skinMask[skinMask == 2] = 0


	return skinMask


# YOLO style, works with PIL
# box = [ymin, xmin, ymax, xmax]
# out_class = float
def draw_boxes(image, score, box, out_class, class_names, colors, font_style):
    
    #image = Image.open(img_path)
    font = ImageFont.truetype(font=font_style, size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    predicted_class = class_names[out_class]

    label = '{} {:.2f}'.format(predicted_class, score)

    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    #print(label, (left, top), (right, bottom))

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[out_class])
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[out_class])
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw


    return image

# specifically for drawing YOLO style bboxes
def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.


    return colors



