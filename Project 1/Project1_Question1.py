# Ryan Miller
# Digital Image Processing
# u1067596
# Project 1 Question 1

# Libraries
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import numpy as np 

# 1. Preliminaries -------------------------------------------------
# a) Read Images from File
img = io.imread("philbobjerry.jpg")
print("Shape of color image array:")
print(img.shape)

# b) Convert to greyscale using numpy dot command
# Function
def color2gray(img):
    weights = [.3, .6, .1]
    return np.dot(img[...,:3], weights)

# Convert using function
gray_img = color2gray(img)
print("Shape of gray image array:")
print(gray_img.shape)

# display images
plt.imshow(gray_img, cmap="gray")
plt.title("With Grayscale")
plt.show()

plt.imshow(img)
plt.title("Original Image")
# Save Image 
io.imsave("gray_xray.jpeg", gray_img)

# Alternativelty, convert to grayscale using skimage
gray_img = color.rgb2gray(img)

