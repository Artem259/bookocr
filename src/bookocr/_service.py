import copy
import cv2
import numpy as np
from scipy.ndimage import rotate


def gray2color(binary_image):
    return cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)


def x1_f(coord):
    return coord[0] + coord[2] - 1


def y1_f(coord):
    return coord[1] + coord[3] - 1


def connected_components_extraction(binary_image, box_areas=False, connectivity=8):
    data = cv2.connectedComponentsWithStats(binary_image, connectivity=connectivity, ltype=cv2.CV_32S)
    data = list(data)
    if box_areas:
        data.append(np.zeros(len(data[2]), dtype=np.int32))
        for coord_i, coord_v in enumerate(data[2]):
            data[4][coord_i] = coord_v[2] * coord_v[3]
    return data


def collapse_connected_components(binary_image):
    height, width = binary_image.shape
    num_labels, labels, coords, centroids = connected_components_extraction(binary_image)

    points_image = np.zeros((height, width), np.uint8)
    for coord in coords:
        x0 = coord[0]
        y0 = coord[1]
        x1 = x0 + coord[2] - 1
        y1 = y0 + coord[3] - 1
        x = (x0 + x1) // 2
        y = y1
        points_image[y][x] = 255
    return points_image


def cut_values(image, v):
    image[image == v] = 0


def move_values(image_from, image_to, v):
    image_to[image_from == v] = v
    image_from[image_from == v] = 0


def copy_update_values(image_from, image_to, v_from, v_to=None):
    if v_to is None:
        v_to = v_from
    image_to[image_from == v_from] = v_to


def move_update_values(image_from, image_to, v_from, v_to=None):
    if v_to is None:
        v_to = v_from
    image_to[image_from == v_from] = v_to
    image_from[image_from == v_from] = 0


def separate_connected_components(labels_image, coords, sep, axis, components_to_sep, sep_to_end=False):
    if axis == 1:
        labels_image = np.transpose(labels_image)
        coords = copy.deepcopy(coords)
        coords[:, [0, 1]] = coords[:, [1, 0]]
        coords[:, [2, 3]] = coords[:, [3, 2]]

    components_to_sep = set(components_to_sep)
    width = labels_image.shape[1]
    components = set(labels_image[sep]) - {0}
    if sep_to_end:
        bottom_components = components_to_sep & components
        top_components = components - bottom_components
    else:
        top_components = components_to_sep & components
        bottom_components = components - top_components

    top_components_coords = [coords[i] for i in top_components]
    bottom_components_coords = [coords[i] for i in bottom_components]

    top_addition = max(map(y1_f, top_components_coords), default=sep) - sep + 1
    bottom_addition = sep - min(map(lambda x: x[1], bottom_components_coords), default=sep)

    top_image = np.vstack((labels_image[:sep, :], np.zeros((top_addition, width), np.int32)))
    bottom_image = np.vstack((np.zeros((bottom_addition, width), np.int32), labels_image[sep:, :]))

    bottom_sep = bottom_addition

    for v in top_components:
        coord = coords[v]
        move_values(bottom_image[bottom_sep:bottom_sep + y1_f(coord) - sep + 1, coord[0]:x1_f(coord) + 1],
                    top_image[sep:y1_f(coord) + 1, coord[0]:x1_f(coord) + 1], v)

    for v in bottom_components:
        coord = coords[v]
        move_values(top_image[coord[1]:sep, coord[0]:x1_f(coord) + 1],
                    bottom_image[bottom_sep - (sep - coord[1]):bottom_sep, coord[0]:x1_f(coord) + 1], v)

    if axis == 1:
        top_image, bottom_image = np.transpose(top_image), np.transpose(bottom_image)
    return top_image, bottom_image


def rotate_image(image, angle, is_binary=False):
    image = rotate(image, angle, reshape=True, order=1)
    if is_binary:
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return image


def crop_image(image, x=None, y=None, w=None, h=None, axis=None):
    if x is None:
        points = cv2.findNonZero(image)
        x, y, w, h = cv2.boundingRect(points)
        if axis == 0:
            return image[y:y + h, :], x, y, w, h
        if axis == 1:
            return image[:, x:x + w], x, y, w, h
        return image[y:y + h, x:x + w], x, y, w, h
    else:
        return image[y:y + h, x:x + w], x, y, w, h


def vertical_histogram(bin_image):
    return np.sum(bin_image, axis=0) // 255


def horizontal_histogram(bin_image):
    return np.sum(bin_image, axis=1) // 255
