from bookocr._json import JSONPublicSerializable


class OcrConfig(JSONPublicSerializable):
    def __init__(self):
        # for image preprocessing
        self.blur_kernel = 0
        self.invert_colors = False
        self.otsu_threshold1 = 0
        self.otsu_threshold2 = 255

        # for image aligning
        self.fix_rotation = True
        self.hough_max_threshold = 500
        self.hough_min_lines = 10
        self.hough_max_lines = 30
        self.hough_angle_range = 0.5
        self.hough_angle_step = 1

        # for texts detection and extraction
        self.cell_size_multiplier = 2
        self.canny_edges_threshold1 = 255/3
        self.canny_edges_threshold2 = 255
        self.text_assumption_threshold1 = 0.2
        self.text_assumption_threshold2 = 0.6
        self.text_assumption_min_occurrences = 2
        self.text_areas_deviation = 1
        self.text_area_padding = 0.5

        # for texts denoising
        self.text_denoising_threshold = 0.08

        # for lines extraction
        self.lines_hist_window = 0.25
        self.lines_hist_frequency = 1.6

        # for words extraction
        self.space_threshold = 0.15
        self.paragraph_spaces = 4
