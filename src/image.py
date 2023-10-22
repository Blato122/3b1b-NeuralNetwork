import climage
from PIL import Image
import numpy as np

def create_image(pixels, x, y, name):
    path = f"../imgs/{name}.png"
    pixels_255 = (pixels * 255).astype(np.uint8)    
    a = np.reshape(pixels_255, (x, y))
    image = Image.fromarray(a, mode="L")
    image.save(path)

def display_image_terminal(name, width):
    path = f"../imgs/{name}.png"
    output = climage.convert(path, width=width) 
    print(output)

def image_to_grayscale(name):
    path = f"../imgs/{name}.png"
    img = Image.open(path).convert(mode="L")
    pixels = np.array(img)
    pixels = np.reshape(pixels, (784, 1))
    return pixels