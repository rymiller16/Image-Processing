#Ryan Miller
#u1067596
#Image Processing 
#Project 3 - Task 1

#Task 1: Computer fourier transform and show power spectrum with frequnecies such that zero in middle
#Use low pass filter in fourier domain and do iFFT to show results

from skimage import io
import math
from skimage import color
import matplotlib.pyplot as plt
import numpy as np 

print("You will be asked to pick a picture to be analyzed.\n")
print("Picture 1 - Golden Gate Bridge\nPicture 2 - Bay Bridge\nPicture 3 - Corona Bridge\n")
pic_input = input("Pick a picture to be analyzed: (1-3): ")
pic_input = int(pic_input)

#Read in the image to be analyzed
if pic_input == 1:
    img = io.imread("Images - Part1/Golden Gate.png")
    image = color.rgb2gray(img)
elif pic_input == 2:
    img = io.imread("Images - Part1/Bay Bridge.png")
    image = color.rgb2gray(img)
elif pic_input == 3:
    img = io.imread("Images - Part1/Corona Bridge.jpg")
    image = color.rgb2gray(img)

#fourier, shift, and power
H,W = np.shape(image)
fourier = np.fft.fft2(image)
fourier_shifted = np.fft.fftshift(np.fft.fft2(image))
power = np.abs(fourier_shifted)**2

fig = plt.figure(figsize = (12,12))
rows = 2
columns = 3

fig.add_subplot(rows,columns,1)
plt.imshow(image,cmap='gray')
plt.title('Original Image')

fig.add_subplot(rows,columns,2)
plt.imshow(np.log(abs(fourier)), cmap='gray')
plt.title('Fourier Transform')

fig.add_subplot(rows,columns,3)
plt.imshow(np.log(abs(fourier_shifted)), cmap='gray')
plt.title('Fourier Shifted')

fig.add_subplot(rows,columns,4)
plt.imshow(np.log(power), cmap='gray')
plt.title('Power Spectrum')

#low pass filter (index = H/2, W/2)
filter_matrix1 = np.zeros((H,W))
filter_matrix1 = filter_matrix1+0.01
radius = 150

for x in range(int(W/2)-radius,int(W/2)+radius):
    for y in range(int(H/2)-radius,int(H/2)+radius):
        x_ = x - int(W/2)
        y_ = y - int(H/2)
        square = x_**2+y_**2
        dist = math.sqrt(square)
        if dist <= radius:
            filter_matrix1[y,x] = 1

#Kaiser Filter 
distance = H
window1d_kaiser = np.kaiser(distance,1)
window2d_kaiser = np.sqrt(np.outer(window1d_kaiser,window1d_kaiser))
filter_matrix3 = window2d_kaiser

#apply filter 
filtered_img1 = fourier_shifted*filter_matrix3
filtered_img1 = np.fft.ifft2(filtered_img1)
filtered_img1 = np.abs(filtered_img1)
fig.add_subplot(rows,columns,5)
plt.imshow(filtered_img1, cmap='gray')
plt.title('Kaiser Filter')

#apply filter 
filtered_img = fourier_shifted*filter_matrix1
filtered_img = np.fft.ifft2(filtered_img)
filtered_img = np.abs(filtered_img)
fig.add_subplot(rows,columns,6)
plt.imshow(filtered_img, cmap='gray')
plt.title('My Low Pass Filter')
            




