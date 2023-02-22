import tensorflow as tf
import numpy as np
import time
import functools
import os

style_predict_path = os.path.join('stylemypicture', 'static', 'model','style_predict.tflite')
style_transform_path = os.path.join('stylemypicture', 'static', 'model','style_transform.tflite')

from PIL import Image
# Function to load an image from a file, and add a batch dimension.


def load_img(path_to_img):  # load PIL image insted of loading from other respurse
    # Added by lotfi to the next #
    img = Image.open(path_to_img).convert('RGB')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    #img = tf.io.read_file(path_to_img)
    #img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img
def load_img_from_PIL(PIL_img):  # load PIL image insted of loading from other respurse
    # Added by lotfi to the next #
    img = PIL_img
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    #img = tf.io.read_file(path_to_img)
    #img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img

# Function to pre-process by resizing an central cropping it.
# Function by lotfi


def preprocess_style_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image
# Function by lotfi


def preprocess_target_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    long_dim = max(shape)
    scale = target_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image, new_shape

# function by lotfi

# crop the image to avoid the pad


def postprocess_target_image(image, new_shape):
    image = tf.image.resize_with_crop_or_pad(image, new_shape[0], new_shape[1])
    #image = tf.image.resize_with_crop_or_pad(image, 200,200)
    return image


# style predection
# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return style_bottleneck


# style transform
# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(
        input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return stylized_image


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


# Final function: process PIL images and return PIL image
#content_path = 'belfry.jpg'
#style_path = 'style23.jpg'


def process_my_picture(style_path, content_path):
    content_path = content_path
    style_path = style_path

    content_image = load_img_from_PIL(content_path)
    style_image = load_img(style_path)

    # Preprocess the input images.
    preprocessed_content_image, new_shape = preprocess_target_image(
        content_image, 384)
    preprocessed_style_image = preprocess_style_image(style_image, 256)
    
    

    # Calculate style bottleneck for the preprocessed style image.
    style_bottleneck = run_style_predict(preprocessed_style_image)

    # Stylize the content image using the style bottleneck.
    stylized_image = run_style_transform(
      style_bottleneck, preprocessed_content_image)
    stylized_image =postprocess_target_image(
        stylized_image, new_shape=new_shape)
    return tensor_to_image(stylized_image)

    """  
    # Style blending

    # Calculate style bottleneck of the content image.
    style_bottleneck_content = run_style_predict(
        preprocess_style_image(content_image, 256)
    )

    # Define content blending ratio between [0..1].
    # 0.0: 0% style extracts from content image.
    # 1.0: 100% style extracted from content image.
    content_blending_ratio = 0.5

    # Blend the style bottleneck of style image and content image
    style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
        + (1 - content_blending_ratio) * style_bottleneck

    # Stylize the content image using the style bottleneck.
    stylized_image_blended = run_style_transform(style_bottleneck_blended,
                                                 preprocessed_content_image)

    stylized_image_blended = postprocess_target_image(
        stylized_image_blended, new_shape=new_shape)
    return tensor_to_image(stylized_image_blended)

    """  
    
#pil_img = process_my_picture(style_path, content_path)

#pil_img.save('out.jpg', "JPEG")
