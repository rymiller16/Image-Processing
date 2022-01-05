# Ryan Miller
# Digital Image Processing
# u1067596
# Project 1 Question 2

# Libraries
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import numpy as np 


def hist_array(gray_img):

    size = gray_img.shape
    height = size[0]
    length = size[1]
    
    array = gray_img[:,:]
    
    #checking for int values 0-255
    if gray_img[1,1]%1 != 0:
        array*=255
        
    array = array.astype(np.uint8)
    gray_img = array.astype(np.uint8)

    array_max = array.max()
    array_min = array.min()

    spread = array_max-array_min
    bins = np.linspace(array_min,array_max,num=spread+1)
    count = []
    count = [0 for i in range(spread+1)]
    
    return_array = [[0 for x in range(spread+1)]for y in range(spread+1)]

    for i in range(array_min,array_max+1):
        for j in range(height):
            for k in range (length):
                if gray_img[j, k] == i:
                    count[i]+= 1
    
    count_array = np.array(count)
    
    for a in range(0,spread+1):
        return_array[a] = (count_array[a],bins[a])
        
    return_array2 = np.asarray(return_array)
        
    return(return_array2)

# Read in image, convert to greyscale, and show it
img = io.imread("floyd_q3.jpg")
plot1=plt.figure(1)
g_img = color.rgb2gray(img)
plt.imshow(g_img, cmap="gray")
                
data_array = hist_array(g_img)
count = data_array[:,0]
bins = data_array[:,1]
y_val = count.max()

plot2=plt.figure(2)
print('Done Running')
plt.bar(bins, count,width=1.0)
plt.xlabel("Bin Number")
plt.ylabel("Number of Occurances")
plt.title('Histogram of Image')
plt.ylim((0,y_val))
plt.xlim((0,255))


    
                
            
        

