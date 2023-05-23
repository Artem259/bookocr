import shutil
from pathlib import Path

from bookocr._json import JSONPublicSerializable


class OcrStatsConfig(JSONPublicSerializable):
    def __init__(self):
        self._is_enabled = False
        self._folder_path = None

        self.first_color = (0, 0, 255)
        self.second_color = (0, 255, 0)
        self.histograms_color = (100, 100, 100)
        self.overlay_opacity = 0.25

        self.padding = 0
        self.lines_thickness = 2

        self.text_denoise_indicators_width = 20

    @property
    def is_enabled(self):
        return self._is_enabled

    def set_enabled_true(self, folder_path):
        self._is_enabled = True
        self._folder_path = Path(folder_path)
        try:
            if self._folder_path.exists():
                shutil.rmtree(self._folder_path)
            self._folder_path.mkdir(parents=True)
        except (Exception,) as e:
            print(e)

    def set_enabled_false(self):
        self._is_enabled = False

    @property
    def folder_path(self):
        return self._folder_path
