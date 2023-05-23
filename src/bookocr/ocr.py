import math
import pandas as pd
from statistics import mode, median
from scipy.signal import find_peaks

from bookocr.config import OcrConfig
from bookocr.stats_config import OcrStatsConfig
from bookocr._stats import Stats
from bookocr._service import *
import bookocr._char_ocr as _c


class Ocr:
    def __init__(self, config: OcrConfig = OcrConfig(), stats_config: OcrStatsConfig = OcrStatsConfig()):
        self._cg = config
        self._scg = stats_config
        self._s = Stats(config, stats_config)
        self._data = None

    def get_data_copy(self):
        return copy.deepcopy(self._data)

    def get_data_as_text(self):
        text = ""
        for page_i, page_v in enumerate(self._data):
            for area_i, area_v in enumerate(page_v):
                text += "\n".join(area_v) + "\n"
            text += "\n"
        return text

    def _image_preprocessing(self, color_image):
        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        if self._cg.blur_kernel > 0:
            if self._cg.blur_kernel % 2 == 0:
                self._cg.blur_kernel += 1
            grayscale_image = cv2.GaussianBlur(grayscale_image, (self._cg.blur_kernel, self._cg.blur_kernel), 0)
        t1, t2 = self._cg.otsu_threshold1, self._cg.otsu_threshold2
        if self._cg.invert_colors:
            _, binary_image = cv2.threshold(grayscale_image, t1, t2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary_image = cv2.threshold(grayscale_image, t1, t2, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return binary_image, grayscale_image

    def _alignment_angle_calculation(self, binary_image):
        points_image = collapse_connected_components(binary_image)
        self._s.save_image("points", points_image)

        theta_step = self._cg.hough_angle_step * np.pi / 180
        min_theta = np.pi * (0.5 - self._cg.hough_angle_range / 2)
        max_theta = np.pi * (0.5 + self._cg.hough_angle_range / 2)
        min_threshold = 1
        max_threshold = self._cg.hough_max_threshold
        lines = None
        curr_threshold = -1
        while min_threshold <= max_threshold:
            prev_threshold = curr_threshold
            curr_threshold = math.ceil((min_threshold + max_threshold) / 2)
            if prev_threshold == curr_threshold:
                break
            lines = cv2.HoughLines(points_image, 1, theta_step, curr_threshold,
                                   min_theta=min_theta, max_theta=max_theta)
            if lines is None:
                lines_n = 0
            else:
                lines_n = len(lines)

            if lines_n < self._cg.hough_min_lines:
                max_threshold = curr_threshold - 1
            elif lines_n > self._cg.hough_max_lines:
                min_threshold = curr_threshold + 1
            else:
                break

        if lines is None:
            return 0
        lines = lines[:, 0, ...]

        angles = []
        for _, theta in lines:
            angles.append(theta)

        if self._scg.is_enabled:
            image_for_stats = gray2color(binary_image)
            self._s.hough_lines_draw(image_for_stats, lines)
            self._s.save_image("lines", image_for_stats)

        angle = math.degrees(mode(angles))
        if angle < 0:
            angle = angle + 90
        else:
            angle = angle - 90
        return angle

    def _texts_extraction(self, binary_image, grayscale_image):
        height, width = binary_image.shape
        connected_components_areas = connected_components_extraction(binary_image, True)[4]
        cell_size = int(math.sqrt(median(set(connected_components_areas[1:]))))
        cell_size = int(cell_size * self._cg.cell_size_multiplier)
        edge_map = cv2.Canny(image=grayscale_image,
                             threshold1=self._cg.canny_edges_threshold1, threshold2=self._cg.canny_edges_threshold2)
        self._s.save_image("edge_map", edge_map)

        if height % cell_size != 0:
            add_height = cell_size - height % cell_size
            edge_map = np.vstack((edge_map, np.zeros((add_height, width), np.uint8)))
            height += add_height
        if width % cell_size != 0:
            add_width = cell_size - width % cell_size
            edge_map = np.hstack((edge_map, np.zeros((height, add_width), np.uint8)))
            width += add_width

        threshold1 = self._cg.text_assumption_threshold1 * cell_size
        threshold2 = self._cg.text_assumption_threshold2 * cell_size
        cells = np.zeros((height // cell_size, width // cell_size), np.uint8)
        for row_i, row_v in enumerate(cells):
            for col_i, _ in enumerate(row_v):
                y0 = row_i * cell_size
                x0 = col_i * cell_size
                y1 = (row_i + 1) * cell_size
                x1 = (col_i + 1) * cell_size
                hor_hist = horizontal_histogram(edge_map[y0:y1, x0:x1])
                count_hor = sum(map(lambda x: threshold1 <= x <= threshold2, hor_hist))
                if count_hor >= self._cg.text_assumption_min_occurrences:
                    cells[row_i][col_i] = 1

        image_for_stats = None
        if self._scg.is_enabled:
            image_for_stats = gray2color(binary_image)
            self._s.texts_extraction_cells_draw(image_for_stats, cells, cell_size)

        _, _, coords, _ = connected_components_extraction(cells)
        coords_with_indices = list(coords)
        coords_with_indices = [[i, v] for i, v in enumerate(coords_with_indices)]
        coords_with_indices = sorted(coords_with_indices[1:], key=lambda x: x[1][4])

        deviation = self._cg.text_areas_deviation
        if len(coords_with_indices) == 0:
            return [[]]
        elif len(coords_with_indices) == 1:
            indices = [[1]]
        else:
            indices = [[coords_with_indices[-1][0]]]
            def hor_overlap(coord1, coord2): return (coord1[0] <= x1_f(coord2)) and (x1_f(coord1) >= coord2[0])

            for index, coord in coords_with_indices[:-1]:
                to_skip = False
                for page in indices:
                    if abs(coord[2] - coords[page[0]][2]) > 2 * deviation \
                            or hor_overlap(coord, coords[page[0]]) \
                            or hor_overlap(coords[page[0]], coord):
                        to_skip = True
                        break
                if not to_skip:
                    indices.append([index])

            indices.sort(key=lambda x: coords[x[0]][0])

            for index, coord in coords_with_indices[:-1]:
                for page in indices:
                    if index in page:
                        break
                    exp_coords = coords[page[0]]
                    if abs(coord[0] - exp_coords[0]) <= deviation and abs(x1_f(coord) - x1_f(exp_coords)) <= deviation:
                        page.append(index)
                        break

        areas_coords = []
        for page in indices:
            areas_coords.append([])
            for i, v in enumerate(page):
                areas_coords[-1].append([coords[v][0], coords[v][1], x1_f(coords[v])+1, y1_f(coords[v])+1])

        for page in areas_coords:
            page.sort(key=lambda x: x[1])
            i = 0
            while i < len(page) - 1:
                if page[i+1][1] - page[i][3] <= deviation:
                    page[i] = [min(page[i][0], page[i+1][0]), page[i][1], max(page[i][2], page[i+1][2]), page[i+1][3]]
                    del page[i+1]
                    continue
                i += 1

        text_images = []
        for page in areas_coords:
            text_images.append([])
            for area in page:
                area[0] = area[0] * cell_size
                area[1] = area[1] * cell_size
                area[2] = min(area[2] * cell_size, grayscale_image.shape[1])
                area[3] = min(area[3] * cell_size, grayscale_image.shape[0])

                padding = int(cell_size * self._cg.text_area_padding)
                if not np.all(binary_image[area[1]:area[3], area[0]] == 0):
                    area[0] = max(area[0] - padding, 0)
                if not np.all(binary_image[area[1], area[0]:area[2]] == 0):
                    area[1] = max(area[1] - padding, 0)
                if not np.all(binary_image[area[1]:area[3], area[2]-1] == 0):
                    area[2] = min(area[2] + padding, grayscale_image.shape[1])
                if not np.all(binary_image[area[3]-1, area[0]:area[2]] == 0):
                    area[3] = min(area[3] + padding, grayscale_image.shape[0])

                area_image = binary_image[area[1]:area[3], area[0]:area[2]]
                area_image, x, y, w, h = crop_image(area_image)
                text_images[-1].append(area_image)

                if self._scg.is_enabled:
                    self._s.draw_rectangle(image_for_stats, (area[0] + x, area[1] + y),
                                           (area[0] + x + w - 1, area[1] + y + h - 1))

        self._s.save_image("text_extraction", image_for_stats)
        return text_images

    def _texts_denoising(self):
        for page_i, page_v in enumerate(self._data):
            for area_i, area_v in enumerate(page_v):
                _, labels, coords, _ = connected_components_extraction(area_v)
                median_height = int(median(set(np.transpose(coords)[3][1:])))
                threshold = int(((median_height * self._cg.text_denoising_threshold) + 1) ** 2)

                labels_copy = None
                if self._scg.is_enabled:
                    labels_copy = copy.deepcopy(labels)

                labels_to_clean = []
                for coord_i, coord_v in enumerate(coords[1:], start=1):
                    if coord_v[4] < threshold:
                        x0, x1 = coord_v[0], x1_f(coord_v)
                        y0, y1 = coord_v[1], y1_f(coord_v)
                        move_update_values(labels[y0:y1 + 1, x0:x1 + 1], area_v[y0:y1 + 1, x0:x1 + 1], coord_i, 0)
                        labels_to_clean.append(coord_i)

                if self._scg.is_enabled:
                    cleaned_image = self._s.texts_denoising_image(gray2color(area_v), labels_copy, coords,
                                                                  labels_to_clean)
                    self._s.save_image("td_" + str(page_i) + "_" + str(area_i), cleaned_image)

                area_v = crop_image(area_v)[0]

                page_v[area_i] = area_v

    def _lines_extraction(self):
        for page_i, page_v in enumerate(self._data):
            for area_i, area_v in enumerate(page_v):
                area_lines = []

                _, labels, coords, _ = connected_components_extraction(area_v)
                median_height = int(median(set(np.transpose(coords)[3][1:])))

                window_size = int(median_height * self._cg.lines_hist_window)
                window_size = window_size + 1 if window_size % 2 == 0 else window_size
                window_half_size = (window_size - 1) // 2

                hist = horizontal_histogram(area_v)
                max_v = max(hist)
                reversed_hist = max_v - hist
                numbers_series = pd.Series(reversed_hist)
                windows = numbers_series.rolling(window_size)
                moving_averages = windows.mean()
                moving_averages_list = moving_averages.tolist()
                reversed_hist = [0]*window_half_size + moving_averages_list[window_size-1:] + [0]*window_half_size
                reversed_hist = list(map(int, reversed_hist))

                seps, _ = find_peaks(reversed_hist, distance=median_height * self._cg.lines_hist_frequency)
                seps = list(seps)
                if len(seps) > 0 and seps[0] < median_height:
                    seps = seps[1:]
                if len(seps) > 0 and seps[-1] > len(reversed_hist) - median_height:
                    seps = seps[:-1]

                to_separate = labels
                for sep in reversed(seps):
                    if hist[sep] == 0:
                        to_separate, line = to_separate[:sep, :], to_separate[sep:, :]
                    else:
                        visited = set()
                        top_components = set()
                        for v in to_separate[sep]:
                            if v == 0 or v in visited:
                                continue
                            coord = coords[v]
                            if coord[3] > median_height * 2:
                                cut_values(to_separate[coord[1]:y1_f(coord)+1, coord[0]:x1_f(coord)+1], v)
                                continue
                            visited.add(v)
                            comp_y = (coord[1] + y1_f(coord)) / 2
                            if comp_y < sep:
                                top_components.add(v)
                        to_separate, line = separate_connected_components(to_separate, coords, sep, 0, top_components)
                    area_lines.append(line)
                area_lines.append(to_separate)
                area_lines.reverse()

                for line_i, line_v in enumerate(area_lines):
                    line_v = crop_image(line_v, axis=0)[0]
                    line_v[line_v != 0] = 255
                    line_v = line_v.astype(np.uint8)
                    area_lines[line_i] = line_v
                area_lines = list(filter(lambda x: x.shape[0] != 0, area_lines))

                if self._scg.is_enabled:
                    hist_image = self._s.horizontal_histogram_barriers(hist, seps)
                    self._s.save_image("h_" + str(page_i) + "_" + str(area_i), hist_image)
                    hist_image = self._s.horizontal_histogram_barriers(max_v - reversed_hist, seps)
                    self._s.save_image("hr_" + str(page_i) + "_" + str(area_i), hist_image)
                    text_image = self._s.horizontal_barriers(gray2color(area_v), seps)
                    self._s.save_image("ht_" + str(page_i) + "_" + str(area_i), text_image)
                    result_image = self._s.vertical_concatenation(list(map(gray2color, area_lines)))
                    self._s.save_image("l_" + str(page_i) + "_" + str(area_i), result_image)

                page_v[area_i] = area_lines

    def _words_extraction(self):
        for page_i, page_v in enumerate(self._data):
            for area_i, area_v in enumerate(page_v):
                for line_i, line_v in enumerate(area_v):
                    line_v, x, _, width, text_height = crop_image(line_v)
                    space_threshold = text_height * self._cg.space_threshold

                    is_paragraph = (x >= space_threshold * self._cg.paragraph_spaces)
                    line_words = [is_paragraph]

                    histogram = vertical_histogram(line_v)
                    curr_gap = 0
                    prev_coord = 0
                    for i, v in enumerate(histogram):
                        if v == 0:
                            curr_gap += 1
                        else:
                            if curr_gap != 0:
                                if curr_gap > space_threshold:
                                    curr_coord = i - curr_gap
                                    line_words.append(line_v[:, prev_coord: curr_coord])
                                    prev_coord = i
                                curr_gap = 0
                    line_words.append(line_v[:, prev_coord:])

                    area_v[line_i] = line_words

                if self._scg.is_enabled:
                    area_image = self._s.area_words_extraction_image(area_v)
                    self._s.save_image("w_" + str(page_i) + "_" + str(area_i), area_image)

    def _characters_extraction(self):
        for page_i, page_v in enumerate(self._data):
            for area_i, area_v in enumerate(page_v):
                for line_i, line_v in enumerate(area_v):
                    for word_i, word_v in enumerate(line_v[1:], start=1):
                        word_chars = []

                        _, labels, coords, _ = connected_components_extraction(word_v)
                        coords_with_indices = list(coords)
                        coords_with_indices = [[i, v] for i, v in enumerate(coords_with_indices)][1:]
                        coords_with_indices = sorted(coords_with_indices, key=lambda x: x1_f(x[1]), reverse=True)

                        to_separate = labels
                        while coords_with_indices:
                            index, coord = coords_with_indices.pop(0)
                            sep = coord[0]
                            right_components = {index}
                            flag = True
                            while flag and coords_with_indices:
                                flag = False
                                for i, [index_t, coord_t] in enumerate(coords_with_indices):
                                    x0_t = coord_t[0]
                                    x1_t = x1_f(coord_t)
                                    if x1_t < sep:
                                        flag = False
                                        break
                                    hor_overlap = x1_t - max(sep, x0_t) + 1
                                    min_width = min(to_separate.shape[1] - sep, coord_t[2])
                                    if hor_overlap > min_width * 0.5:
                                        sep = min(sep, x0_t)
                                        right_components.add(index_t)
                                        del coords_with_indices[i]
                                        flag = True
                                        break
                            if sep >= 0:
                                to_separate, char = separate_connected_components(to_separate, coords, sep, 1,
                                                                                  right_components, sep_to_end=True)
                                to_separate = crop_image(to_separate, axis=1)[0]
                                word_chars.append(char)
                        word_chars.append(to_separate)
                        word_chars.reverse()

                        for char_i, char_v in enumerate(word_chars):
                            char_v = crop_image(char_v, axis=1)[0]
                            char_v[char_v != 0] = 255
                            char_v = char_v.astype(np.uint8)
                            word_chars[char_i] = char_v
                        word_chars = list(filter(lambda x: x.shape[1] != 0, word_chars))

                        line_v[word_i] = word_chars

                if self._scg.is_enabled:
                    area_image = self._s.area_chars_extraction_image(area_v)
                    self._s.save_image("c_" + str(page_i) + "_" + str(area_i), area_image)

    def _characters_recognition(self):
        _c.pages_ocr(self._data)

    def image_ocr(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Not an image.")
        self._s.save_image("input_image", image)

        binary_image, grayscale_image = self._image_preprocessing(image)
        self._s.save_image("binary_image", binary_image)
        self._s.save_image("grayscale_image", grayscale_image)

        if self._cg.fix_rotation:
            angle = self._alignment_angle_calculation(binary_image)
            binary_image = rotate_image(binary_image, angle, True)
            grayscale_image = rotate_image(grayscale_image, angle)
        binary_image, x, y, w, h = crop_image(binary_image)
        grayscale_image = crop_image(grayscale_image, x, y, w, h)[0]
        if binary_image.size == 0:
            self._data = []
            return self.get_data_copy()
        self._s.save_image("aligned_binary_image", binary_image)
        self._s.save_image("aligned_grayscale_image", grayscale_image)

        self._data = self._texts_extraction(binary_image, grayscale_image)
        self._texts_denoising()
        self._lines_extraction()
        self._words_extraction()
        self._characters_extraction()
        self._characters_recognition()

        if self._scg.is_enabled:
            self._s.save_text("output", self.get_data_as_text())

        return self.get_data_copy()
