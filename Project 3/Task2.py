#Ryan Miller
#u1067596
#Image Processing 
#Project 3 - Task 2

#Task 2: Phase Correlation between two images of same size 

from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import numpy as np 
from skimage.metrics import mean_squared_error

print("You will be asked to pick a picture to be analyzed.\n")
print("Picture 1 - Golden Gate Bridge\nPicture 2 - Bay Bridge\nPicture 3 - Corona Bridge\n")
pic_input = input("Pick a picture to be analyzed: (1-3): ")
pic_input = int(pic_input)

#Read in the image to be analyzed
if pic_input == 1:
    img1 = io.imread("Images - Part2/Golden Gate - 1.png")
    image1 = color.rgb2gray(img1)
    img2 = io.imread("Images - Part2/Golden Gate - 2.png")
    image2 = color.rgb2gray(img2)
elif pic_input == 2:
    img1 = io.imread("Images - Part2/Bay Bridge - 1.png")
    image1 = color.rgb2gray(img1)
    img2 = io.imread("Images - Part2/Bay Bridge - 2.png")
    image2 = color.rgb2gray(img2)
elif pic_input == 3:
    img1 = io.imread("Images - Part2/Corona Bridge - 1.jpg")
    image1 = color.rgb2gray(img1)
    img2 = io.imread("Images - Part2/Corona Bridge - 2.jpg")
    image2 = color.rgb2gray(img2)
    
img1 = io.imread("Images - Part2/0001.002.png")
image1 = color.rgb2gray(img1)
img2 = io.imread("Images - Part2/0001.001.png")
image2 = color.rgb2gray(img2)



def phase_correlation(a, b):
    F = np.fft.fft2(a)
    G = np.fft.fft2(b)
    conj_F = np.conjugate(F)
    R = conj_F*G
    R /= np.absolute(R)
    
    #lowpass filter
    H,W = np.shape(a)
    distance = H
    window1d_kaiser = np.kaiser(distance,100)
    window2d_kaiser = np.sqrt(np.outer(window1d_kaiser,window1d_kaiser))

    R = R*window2d_kaiser
    r = np.fft.ifft2(R).real

    canvas = np.zeros((2*H,2*W))
    for x in range(0,W):
        for y in range(0,H):
            canvas[x,y] = a[x,y]
    for x in range(W,2*W):
        for y in range(H,2*H):
            canvas[x,y] = a[x-W,y-H]
    for x in range(W,2*W):
        for y in range(0,H):
            canvas[x,y] = a[x-W,y]
    for x in range(0,W):
        for y in range(H,2*H):
            canvas[x,y] = a[x,y-H]
    
    x, y = (np.where(r==np.amax(r)))        
    x = int(x)
    y = int(y)
    trans_x = W - x
    trans_y = H - y
    
    canvas2 = np.ones((2*H,2*W))
    for x in range(0+trans_x,W+trans_x):
        for y in range(0+trans_y,H+trans_y):
            canvas2[x,y] = b[x-trans_x,y-trans_y]
    
    x, y = (np.where(r==np.amax(r)))       
    x = int(x)
    y = int(y)

    list_1 = list(range(0,trans_x)) + list(range(trans_x+W,2*W))
    list_2 = list(range(0,trans_y)) + list(range(trans_y+H,2*H))
    canvas = np.delete(canvas,(list_1), axis = 0)
    canvas = np.delete(canvas,(list_2), axis = 1)
    canvas2 = np.delete(canvas2,(list_1), axis = 0)
    canvas2 = np.delete(canvas2,(list_2), axis = 1)
    
    MSE_domains = np.zeros(4)
    counter = 0 
    
    #first subdomain (Top Left)
    list_s1 = list(range(x,W))
    list_s2 = list(range(y,H))
    canvas_s1 = np.delete(canvas,(list_s1),axis=0)
    canvas2_s1 = np.delete(canvas2,(list_s2),axis=1)
    canvas_s1 = np.delete(canvas_s1,(list_s2),axis=1)
    canvas2_s1 = np.delete(canvas2_s1,(list_s1),axis=0)
    MSE_domains[counter] = mean_squared_error(canvas_s1, canvas2_s1)
    counter = counter+1
    
    #second subdomain (Top Right)
    list_s1 = list(range(0,x))
    list_s2 = list(range(y,H))
    canvas_s1 = np.delete(canvas,(list_s1),axis=0)
    canvas2_s1 = np.delete(canvas2,(list_s2),axis=1)
    canvas_s1 = np.delete(canvas_s1,(list_s2),axis=1)
    canvas2_s1 = np.delete(canvas2_s1,(list_s1),axis=0)
    MSE_domains[counter] = mean_squared_error(canvas_s1, canvas2_s1)
    counter = counter+1
    
    #third subdomain (Bottom Left)
    list_s1 = list(range(x,W))
    list_s2 = list(range(0,y))
    canvas_s1 = np.delete(canvas,(list_s1),axis=0)
    canvas2_s1 = np.delete(canvas2,(list_s2),axis=1)
    canvas_s1 = np.delete(canvas_s1,(list_s2),axis=1)
    canvas2_s1 = np.delete(canvas2_s1,(list_s1),axis=0)
    MSE_domains[counter] = mean_squared_error(canvas_s1, canvas2_s1)
    counter = counter+1
    
    #fourth subdomain (Bottom Right)
    list_s1 = list(range(0,x))
    list_s2 = list(range(0,y))
    canvas_s1 = np.delete(canvas,(list_s1),axis=0)
    canvas2_s1 = np.delete(canvas2,(list_s2),axis=1)
    canvas_s1 = np.delete(canvas_s1,(list_s2),axis=1)
    canvas2_s1 = np.delete(canvas2_s1,(list_s1),axis=0)
    MSE_domains[counter] = mean_squared_error(canvas_s1, canvas2_s1)
    counter = counter+1
    
    max_MSE = np.argmin(MSE_domains, axis=0)
    if max_MSE == 0:
        loc = "TopL"
    elif max_MSE == 1:
        loc = "TopR"
    elif max_MSE == 2:
        loc = "BotL"
    elif max_MSE == 3:
        loc = "BotR"
        
    x, y = (np.where(r==np.amax(r)))        #x,y of phase correlation peak
    x = int(x)
    y = int(y)
            
    return r, loc, x, y


result, loc, x, y = phase_correlation(image1, image2)

print(loc)
print('\n')
print(x)
print('\n')
print(y)

fig = plt.figure(figsize = (12,12))
rows = 1
columns = 3

fig.add_subplot(rows,columns,1)
plt.imshow(image1,cmap ="gray")
plt.title('Original Image')

fig.add_subplot(rows,columns,2)
plt.imshow(image2,cmap ="gray")
plt.title('Translated Image')

fig.add_subplot(rows,columns,3)
plt.imshow(result,cmap = 'gray')
plt.title('X: %1.2f, Y: %1.2f, Loc: %s' %(y, x, loc))












