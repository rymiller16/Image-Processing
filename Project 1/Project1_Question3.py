# Ryan Miller
# Digital Image Processing
# u1067596
# Project 1 Question 3

# Libraries
from skimage import io
from skimage import color
from skimage import measure
import matplotlib.pyplot as plt 
from skimage.segmentation import flood_fill
import numpy as np
from skimage.measure import label, regionprops
from skimage import morphology

def threshold_array(gray_img):

    size = gray_img.shape
    
    if gray_img[1,1]%1 != 0:
        gray_img*=255
    gray_img = gray_img.astype(np.uint8)
        
    height = size[0]
    length = size[1]

    threshold1 = 200
    threshold2 = 125

    for j in range(height):
        for k in range (length):
            if gray_img[j, k] > threshold1:
                gray_img[j,k] = 0
            elif gray_img[j,k] < threshold2:
               gray_img[j,k] = 0
            else:
                   gray_img[j,k] = 255
            
    return(gray_img)

#first load in the image
img = io.imread("brent.jpg")
gray_img = color.rgb2gray(img)
gray_img1 = threshold_array(gray_img)

#original image
plot1=plt.figure(1)
plt.imshow(img)
plt.show()

#image after being passed through function
plot2=plt.figure(2)
plt.imshow(gray_img1,cmap="gray")
plt.show()

#histogram of new image 
plot3=plt.figure(3)
ax = plt.hist(gray_img1.ravel(), bins = 256)
plt.show()

#Flood fill
plot4=plt.figure(4)
image_FloodFill = flood_fill(gray_img1, (550, 475), 150, tolerance=200)
plt.imshow(image_FloodFill,cmap="gray")
plt.show



#connected components
plot5=plt.figure(5)
all_labels = measure.label(image_FloodFill)
removed_img = morphology.remove_small_objects(all_labels,1000)

plt.imshow(removed_img, cmap="nipy_spectral")
plt.show