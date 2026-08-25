from PIL import Image
import numpy as np

img_loaded = Image.open('overlap_ablation.tiff')

img_loaded.save('image.png', 'PNG')

print("TIFF 文件已成功保存为 PNG 文件。")