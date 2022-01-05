# Ryan Miller
# Digital Image Processing
# u1067596
# Project 1 Question 4

# Libraries
from skimage import io
from skimage import color
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np 
from skimage.morphology import disk
from skimage.exposure import equalize_adapthist

img = io.imread("xray.png")
gray_img = color.rgb2gray(img)

if gray_img[1,1]%1 != 0:
    gray_img*=255
gray_img = gray_img.astype(np.uint8)

#figures in order, original pic, original historgram, original cdf
#after equilization: new pic, new histogram, new cdf

#original picture
plot1=plt.figure(1)
plt.imshow(gray_img, cmap="gray")

#original histogram
plot2=plt.figure(2)
ax = plt.hist(gray_img.ravel(), bins = 256)

#original cdf
plot3=plt.figure(3)
count, bins_count = np.histogram(gray_img, bins = 256)
pdf = count/ sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], cdf)
plt.title("CDF of Original Histogram")

#Histogram Equilization 
img_eq = exposure.equalize_hist(gray_img)
if img_eq[1,1]%1 != 0:
    img_eq*=255
img_eq = img_eq.astype(np.uint8)

#new picture
plot4=plt.figure(4)
plt.imshow(img_eq, cmap="gray")

#new histogram
plot5=plt.figure(5)
ax = plt.hist(img_eq.ravel(), bins = 256)

#new cdf
plot6=plt.figure(6)
count1, bins_count1 = np.histogram(img_eq, bins = 256)
pdf1 = count1/ sum(count1)
cdf1 = np.cumsum(pdf1)
plt.plot(bins_count1[1:], cdf1)
plt.title("New CDF")

#local equilization 
plot7=plt.figure(7)
local_img_eq = equalize_adapthist(gray_img,kernel_size=None, clip_limit=1, nbins=256) #could also try ball
plt.imshow(local_img_eq, cmap="gray")

#local equilization cdf
plot8=plt.figure(8)
count2, bins_count2 = np.histogram(local_img_eq, bins = 256)
pdf2 = count2/ sum(count2)
cdf2 = np.cumsum(pdf2)
plt.plot(bins_count2[1:], cdf2)
plt.title("CDF of Local Equilization Histogram")

plot9=plt.figure(9)
ax = plt.hist(local_img_eq.ravel(), bins = 256)

plt.show()


