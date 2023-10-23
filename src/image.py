import climage
from PIL import Image
import numpy as np
import network as n

def create_image(pixels, x, y, name):
    """Saves a new image created from an array of pixels (grayscale, values between 0 and 1)."""
    path = f"../imgs/{name}.png"
    pixels_255 = (pixels * 255).astype(np.uint8)    
    pixels_255 = np.reshape(pixels_255, (x, y))
    image = Image.fromarray(pixels_255, mode="L") # grayscale mode
    image.save(path)

def display_image_terminal(name, width):
    """Displays an image in a terminal."""
    path = f"../imgs/{name}.png"
    output = climage.convert(path, width=width) 
    print(output)

def image_to_grayscale(name):
    """Converts an image to an array consisting of grayscale values of the pixels (between 0 and 1)."""
    path = f"../imgs/{name}.png"
    img = Image.open(path).convert(mode="L") # grayscale mode
    img = img.resize((n.WIDTH, n.HEIGHT))
    pixels = np.array(img) / 255
    pixels = np.reshape(pixels, (n.INPUT_N, 1))
    return pixels