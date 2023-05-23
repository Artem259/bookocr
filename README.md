# BookOcr

### Optical character recognition (OCR) tool for printed book pages.

### Package installation:
via ``pip``
```
pip install bookocr
```
or using release files. 

### Usage examples:

```
from bookocr.ocr import Ocr

ocr = Ocr()
image_path = "my_image.png"
nested_list_structure = ocr.image_ocr(image_path)  # pages > text areas > lines > words
text = ocr.get_data_as_text()  # the same result, but joined

print(text)
```

```
from bookocr.config import OcrConfig
from bookocr.stats_config import OcrStatsConfig
from bookocr.ocr import Ocr

# optional
config = OcrConfig()
# ... set config values here

# optional as well
# provides intermediate results of image processing 
stats_config = OcrStatsConfig()
stats_config.set_enabled_true("stats_folder")
# ... set stats_config values here

ocr = Ocr(config, stats_config)
image_path = "my_image.png"
nested_list_structure = ocr.image_ocr(image_path)  # pages > text areas > lines > words
text = ocr.get_data_as_text()  # the same result, but joined

print(text)
```
