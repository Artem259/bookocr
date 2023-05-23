import copy
import cv2
import numpy as np
from pathlib import Path

from bookocr.config import OcrConfig
from bookocr.stats_config import OcrStatsConfig
import bookocr.ocr as _ocr


def transparent_image(shape, foreground_color, background_color, opacity):
    foreground = np.full(shape, fill_value=list(foreground_color), dtype=np.uint8)
    background = np.full(shape, fill_value=list(background_color), dtype=np.uint8)
    image = cv2.addWeighted(foreground, opacity, background, 1 - opacity, 0)
    return image


class Stats:
    def __init__(self, config: OcrConfig, stats_config: OcrStatsConfig):
        self._cg = config
        self._scg = stats_config

    def save_text(self, label, text):
        with open(Path(self._scg.folder_path) / Path(label + ".txt"), "w") as f:
            print(text, file=f)

    def save_image(self, label, image):
        if self._scg.is_enabled:
            path = Path(self._scg.folder_path) / Path(label + ".png")
            cv2.imwrite(str(path), image)

    def line_size(self, thickness=None):
        if thickness is None:
            thickness = self._scg.lines_thickness
        return (thickness - 1) * 2 + 1

    def vertical_histogram(self, histogram_list, color=None):
        if color is None:
            color = self._scg.histograms_color
        height = max(histogram_list) + self._scg.padding
        width = len(histogram_list)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i, v in enumerate(histogram_list):
            cv2.line(image, (i, height), (i, height-int(v)), color, 1)
        return image

    def horizontal_histogram(self, histogram_list, color=None):
        if color is None:
            color = self._scg.histograms_color
        height = len(histogram_list)
        width = max(histogram_list) + self._scg.padding
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i, v in enumerate(histogram_list):
            cv2.line(image, (width, i), (width-int(v), i), color, 1)
        return image

    def vertical_barriers(self, image, separators, barrier_color=None):
        if barrier_color is None:
            barrier_color = self._scg.first_color
        for sep in separators:
            cv2.line(image, (sep, 0), (sep, image.shape[0]), barrier_color, self._scg.lines_thickness)
        return image

    def horizontal_barriers(self, image, separators, barrier_color=None):
        if barrier_color is None:
            barrier_color = self._scg.first_color
        for sep in separators:
            cv2.line(image, (0, sep), (image.shape[1], sep), barrier_color, self._scg.lines_thickness)
        return image

    def vertical_histogram_barriers(self, histogram_list, separators, hist_color=None, barrier_color=None):
        if hist_color is None:
            hist_color = self._scg.histograms_color
        if barrier_color is None:
            barrier_color = self._scg.first_color
        image = self.vertical_histogram(histogram_list, hist_color)
        for sep in separators:
            cv2.line(image, (sep, 0), (sep, image.shape[0]-1-histogram_list[sep]), barrier_color,
                     self._scg.lines_thickness)
        return image

    def horizontal_histogram_barriers(self, histogram_list, separators, hist_color=None, barrier_color=None):
        if hist_color is None:
            hist_color = self._scg.histograms_color
        if barrier_color is None:
            barrier_color = self._scg.first_color
        image = self.horizontal_histogram(histogram_list, hist_color)
        for sep in separators:
            cv2.line(image, (0, sep), (image.shape[1]-1-histogram_list[sep], sep), barrier_color,
                     self._scg.lines_thickness)
        return image

    def vertical_concatenation(self, images_list, padding=None, barrier_color=None, last_barrier=False):
        if padding is None:
            padding = self._scg.padding
        if barrier_color is None:
            barrier_color = self._scg.first_color
        images_n = len(images_list)
        width = images_list[0].shape[1]
        barrier_size = self.line_size()

        height = padding*(images_n*2) + barrier_size*(images_n-1)
        for image in images_list:
            height += image.shape[0]
        if last_barrier:
            height += barrier_size

        pos = 0
        image = np.zeros((height, width, 3), np.uint8)
        for image_i, image_v in enumerate(images_list):
            pos += padding
            image[pos:pos+image_v.shape[0], ] = image_v
            pos += image_v.shape[0]
            if image_i != len(images_list) - 1 or last_barrier:
                pos += padding
                barrier_y = pos + self._scg.lines_thickness - 1
                cv2.line(image, (0, barrier_y), (width, barrier_y), barrier_color, self._scg.lines_thickness)
                pos += barrier_size
        return image

    def horizontal_concatenation(self, images_list, padding=None, barrier_color=None, last_barrier=False):
        if padding is None:
            padding = self._scg.padding
        if barrier_color is None:
            barrier_color = self._scg.first_color
        images_n = len(images_list)
        height = images_list[0].shape[0]
        barrier_size = self.line_size()

        width = padding*(images_n*2) + barrier_size*(images_n-1)
        for image in images_list:
            width += image.shape[1]
        if last_barrier:
            width += barrier_size

        pos = 0
        image = np.zeros((height, width, 3), np.uint8)
        for image_i, image_v in enumerate(images_list):
            pos += padding
            image[:, pos:pos+image_v.shape[1]] = image_v
            pos += image_v.shape[1]
            if image_i != len(images_list) - 1 or last_barrier:
                pos += padding
                barrier_x = pos + self._scg.lines_thickness - 1
                cv2.line(image, (barrier_x, 0), (barrier_x, height), barrier_color, self._scg.lines_thickness)
                pos += barrier_size
        return image

    def add_padding(self, image, axis, padding=None):
        if padding is None:
            padding = self._scg.padding
        if axis == 0:
            image = np.vstack((np.zeros((padding, image.shape[1], 3), np.uint8), image))
            image = np.vstack((image, np.zeros((padding, image.shape[1], 3), np.uint8)))
        else:
            image = np.hstack((np.zeros((image.shape[0], padding, 3), np.uint8), image))
            image = np.hstack((image, np.zeros((image.shape[0], padding, 3), np.uint8)))
        return image

    def draw_rectangle(self, image, point1, point2, rectangle_color=None):
        if rectangle_color is None:
            rectangle_color = self._scg.second_color
        cv2.rectangle(image, point1, point2, rectangle_color, self._scg.lines_thickness)

    def hough_lines_draw(self, image, r_theta_list, line_color=None):
        if line_color is None:
            line_color = self._scg.first_color
        height, width, _ = image.shape
        for r, theta in r_theta_list:
            d = height + width
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + d * (-b))
            y1 = int(y0 + d * a)
            x2 = int(x0 - d * (-b))
            y2 = int(y0 - d * a)
            cv2.line(image, (x1, y1), (x2, y2), line_color, self._scg.lines_thickness)

    def texts_extraction_cells_draw(self, image, cells, cell_size, cell_color=None):
        if cell_color is None:
            cell_color = self._scg.first_color
        for row_i, row_v in enumerate(cells):
            for col_i, v in enumerate(row_v):
                if v == 1:
                    y0 = row_i * cell_size
                    x0 = col_i * cell_size
                    y1 = (row_i + 1) * cell_size
                    x1 = (col_i + 1) * cell_size

                    sub_image = image[y0:y1, x0:x1]
                    color_rect = np.full(sub_image.shape, fill_value=list(cell_color), dtype=np.uint8)
                    transparent_rect = cv2.addWeighted(sub_image, 1 - self._scg.overlay_opacity,
                                                       color_rect, self._scg.overlay_opacity, 0)
                    image[y0:y1, x0:x1] = transparent_rect

    def texts_denoising_image(self, image, labels, coords, labels_to_clean, color=None):
        if color is None:
            color = self._scg.first_color
        indicators_height = image.shape[0]
        indicators_width = self._scg.padding + self.line_size() + self._scg.text_denoise_indicators_width
        cleaned_image = np.hstack((image, np.zeros((indicators_height, indicators_width, 3), np.uint8)))
        line_x = image.shape[1] + self._scg.padding + self._scg.lines_thickness - 1
        cv2.line(cleaned_image, (line_x, 0), (line_x, indicators_height), color, self._scg.lines_thickness)
        for coord_i in labels_to_clean:
            coord_v = coords[coord_i]
            x0, x1 = coord_v[0], _ocr.x1_f(coord_v)
            y0, y1 = coord_v[1], _ocr.y1_f(coord_v)
            _ocr.copy_update_values(labels[y0:y1+1, x0:x1+1], cleaned_image[y0:y1+1, x0:x1+1], coord_i, list(color))
            line_y = (coord_v[1] + _ocr.y1_f(coord_v)) // 2
            cv2.line(cleaned_image, (line_x, line_y), (cleaned_image.shape[1], line_y),
                     color, self._scg.lines_thickness)
        return cleaned_image

    def area_words_extraction_image(self, area_words, from_chars=False):
        paragraph_width_multiplier = self._cg.space_threshold * self._cg.paragraph_spaces
        line_images = []
        for line_words_i, line_words_v in enumerate(area_words):
            is_paragraph = line_words_v[0]
            line_words_v = line_words_v[1:].copy()
            if not from_chars:
                for word_i, word_v in enumerate(line_words_v):
                    word_v = _ocr.gray2color(word_v)
                    line_words_v[word_i] = self.add_padding(word_v, 0)
                line_image = self.horizontal_concatenation(line_words_v, last_barrier=True)
            else:
                line_image = self.horizontal_concatenation(line_words_v, padding=0, last_barrier=True)
            if is_paragraph:
                line_height = line_words_v[0].shape[0]
                paragraph_width = int(line_height * paragraph_width_multiplier)
                paragraph_image = transparent_image((line_height, paragraph_width, 3),
                                                    self._scg.first_color, (0, 0, 0), self._scg.overlay_opacity)
                line_image = self.horizontal_concatenation([paragraph_image, line_image], padding=0)
            line_images.append(line_image)
        max_width = max(map(lambda x: x.shape[1], line_images), default=0)
        for line_image_i, line_image_v in enumerate(line_images):
            add_width = max_width - line_image_v.shape[1]
            if add_width == 0:
                continue
            add_image = transparent_image((line_image_v.shape[0], add_width, 3),
                                          self._scg.first_color, (0, 0, 0), self._scg.overlay_opacity)
            line_images[line_image_i] = np.hstack((line_image_v, add_image))
        return self.vertical_concatenation(line_images, padding=0)

    def area_chars_extraction_image(self, area_chars):
        area_chars = copy.deepcopy(area_chars)
        space_width_multiplier = self._cg.space_threshold * self._cg.paragraph_spaces
        area_words = []
        for line_chars_i, line_chars_v in enumerate(area_chars):
            line_words = [line_chars_v[0]]
            for word_chars_i, word_chars_v in enumerate(line_chars_v[1:], start=1):
                for char_i, char_v in enumerate(word_chars_v):
                    char_v = _ocr.gray2color(char_v)
                    char_v = self.add_padding(char_v, 0)
                    word_chars_v[char_i] = char_v
                word_image = self.horizontal_concatenation(word_chars_v,
                                                           barrier_color=(0, 0, 255 * self._scg.overlay_opacity))
                if word_chars_i != len(line_chars_v) - 1:
                    line_height = word_image.shape[0]
                    space_width = int(line_height * space_width_multiplier)
                    space_image = transparent_image((line_height, space_width, 3),
                                                    self._scg.first_color, (0, 0, 0), self._scg.overlay_opacity)
                    word_image = self.horizontal_concatenation([word_image, space_image], padding=0)
                line_words.append(word_image)
            area_words.append(line_words)
        return self.area_words_extraction_image(area_words, True)
