import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from importlib import resources


_image_size = 32
_char_labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!:;-()&'"
_model_file = (resources.files(__package__) / "model.h5")
_model = tf.keras.models.load_model(_model_file)


def _flatten(xs):
    for x in xs:
        if isinstance(x, list):
            yield from _flatten(x)
        else:
            yield x


def _make_square(image):
    original_height, original_width = image.shape
    max_dim = max(original_height, original_width)
    expanded_image = np.zeros((max_dim, max_dim), dtype=image.dtype)

    padding_top = (max_dim - original_height) // 2
    padding_left = (max_dim - original_width) // 2
    expanded_image[padding_top:padding_top + original_height, padding_left:padding_left + original_width] = image

    return expanded_image


def _resize(image, size):
    image = Image.fromarray(image, mode='L')
    return np.array(image.resize((size, size), resample=Image.BICUBIC))


def _preprocess(image):
    image = _make_square(image)
    image = _resize(image, _image_size)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image = np.expand_dims(image, axis=-1)
    image = image.astype(np.float32) / 255
    return image


def char_ocr(images_list):
    images_list = np.array(list(map(_preprocess, images_list)), np.uint8)
    prediction = _model.predict(images_list, verbose=0)
    return [_char_labels[int(np.argmax(x))] for x in prediction]


def pages_ocr(pages):
    chars = list(_flatten(pages))
    chars = [x for x in chars if not isinstance(x, bool)]
    chars = char_ocr(chars)

    index = 0
    for page_i, page_v in enumerate(pages):
        for area_i, area_v in enumerate(page_v):
            for line_i, line_v in enumerate(area_v):
                line = ""
                if line_v[0]:
                    line = "\t"
                for word_i, word_v in enumerate(line_v[1:], start=1):
                    word_len = len(word_v)
                    line += "".join(chars[index:index + word_len]) + " "
                    index += word_len
                line = line[:-1]
                area_v[line_i] = line
