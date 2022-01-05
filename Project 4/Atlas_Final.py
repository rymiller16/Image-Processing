#Ryan Miller
#u1067596
#Image Processing 
#Dr. Whitaker 
#Project 4 - Atlas

import numpy as np 
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from Atlas_Formatting import Atlas_Formatting
from scipy.interpolate import interp2d
from scipy.linalg import svd
from skimage.segmentation import flood_fill
import numpy as np
from skimage import measure
from skimage.measure import label, regionprops
from skimage import morphology

file_name = "atlas_params.json"
num_images, img_names, img_list, corrs, corrs_array, files, output_name, num_corrs = Atlas_Formatting(file_name)

#Setting Correspondances for each image and for the canvas 
N = num_corrs
avg_corrs = [[]]*N

H = len(img_list[0])
W = len(img_list[0][0])

corrs_array2 = np.zeros((num_images,num_corrs,2))
for i in range(num_images):
    for j in range(num_corrs):
        corrs_array2[i,j] = corrs_array[j,i]

image_corrs =  np.ndarray.tolist(corrs_array2)
x = 0.0
y = 0.0
for j in range(N):
    for i in range(num_images):
        x = x + image_corrs[i][j][0]
        y = y + image_corrs[i][j][1]
    x = x/num_images
    y = y/num_images
    avg_corrs[j] = [x,y]
    x=0
    y=0
    
#Methodology: same as before (transform from I1 to It to get canvas size)
B = np.zeros((num_images,N+3,N+3))
B_rev = np.zeros((num_images,N+3,N+3))
A = np.zeros((num_images,(N+3)*2,(N+3)*2))
A_rev = np.zeros((num_images,(N+3)*2,(N+3)*2))

def phi_fn(i,j,corrs):
    eps = 0.0001
    x_i = corrs[i]
    x_i = np.asmatrix(x_i)
    x_j = corrs[j]
    x_j = np.asmatrix(x_j)
    xi_xj = x_i - x_j
    norm = np.linalg.norm(xi_xj)
    norm_log = norm+eps
    return norm**2 * np.log(norm_log)

def set_B(corrs):
    B = np.zeros((N+3,N+3))
    for i in range(N):
        B[0,i] = corrs[i][0]
        B[1,i] = corrs[i][1]
        B[2,i] = 1
        B[i+3,N] = corrs[i][1]
        B[i+3,N+1] = corrs[i][0]
        B[i+3,N+2] = 1
    for i in range(N,N+3):
        B[0,i] = 0
        B[1,i] = 0
        B[2,i] = 0
    for i in range(0,N):
        for j in range(0,N):
            B[i+3][j] = phi_fn(i,j,corrs)
    return B

for i in range(num_images):
    B[i,:,:] = set_B(image_corrs[i])
    B_rev[i] = set_B(avg_corrs)
    
def set_A(B):
    size_A = (N+3)*2
    mid = int(size_A / 2)
    A = np.zeros((size_A,size_A))
    for i in range(N+3):
        for j in range(N+3):
            A[i][j] = B[i][j]
            A[mid+i][mid+j] = B[i][j]
    return A

for i in range(num_images):
    A[i,:,:] = set_A(B[i])
    A_rev[i] = set_A(B_rev[i])

b = np.zeros((num_images,2*(N+3),1))
b_rev = np.zeros((num_images,2*(N+3),1))
z = np.zeros((num_images,2*(N+3),1))
z_rev = np.zeros((num_images,2*(N+3),1))

def set_b(corrs):
    b = np.zeros((2*(N+3),1))
    for i in range(3,N+3):
        b[i] = corrs[i-3][0]
        b[i+N+3] = corrs[i-3][1]
    return b 

for i in range(num_images):
    b[i,:] = set_b(avg_corrs)
    b_rev[i,:] = set_b(image_corrs[i])

def solve_svd(A,b):
    U,s,Vh = svd(A)
    c = np.dot(U.T,b)
    w = np.dot(np.diag(1/s),c)
    z = np.dot(Vh.conj().T,w)
    return z

for i in range(num_images):
    z[i,:] = solve_svd(A[i,:,:],b[i,:])
    z_rev[i] = solve_svd(A_rev[i],b_rev[i])

def phi_i(x_bar,i,corrs): #x is (x,y) and x_i is (x_i,y_i)
    eps = 0.0001
    x_i = corrs[i]
    x_i = np.asmatrix(x_i)
    x = np.asmatrix(x_bar)
    x_x_i = x - x_i
    norm = np.linalg.norm(x_x_i)
    norm_log = norm+eps
    return norm**2 * np.log(norm_log)

def set_T(x_,y_,z,img_corrs):

    p_2_x = float(z[num_corrs])
    p_1_x = float(z[num_corrs+1])
    p_0_x = float(z[num_corrs+2])
    p_2_y = float(z[num_corrs*2+3])
    p_1_y = float(z[num_corrs*2+4])
    p_0_y = float(z[num_corrs*2+5])
    T_x = 0
    T_y = 0
    
    for j in range(N):  #this is our summation block
        k_i_x = float(z[j])
        k_i_y = float(z[num_corrs+3+j])
            
        x_bar = [x_,y_]
        phi = phi_i(x_bar,j,img_corrs)
            
        T_x = T_x + k_i_x*phi   
        T_y = T_y + k_i_y*phi
    
    T_x = T_x + p_0_x + p_1_x*x_ + p_2_x*y_
    T_y = T_y + p_0_y + p_1_y*x_ + p_2_y*y_
    
    x_return = T_x
    y_return = T_y
    
    return y_return, x_return

T = np.array([[]]*W*H*num_images + [[1]],dtype=object)[:-1]
T = T.reshape((num_images,H,W))
T_trans = np.array([[]]*W*H*num_images + [[1]],dtype=object)[:-1]
T_trans = T_trans.reshape((num_images,H,W))

for k in range(num_images):
    for i in range(W):
        for j in range(H):
            T[k,j,i] = [j,i]
            T[k,j,i] = [j,i]
            T_trans[k,j,i] = set_T(i,j,z[k],image_corrs[k])              #i is x, j is y

#To determine canvas size
max_x = T_trans[0][0,0][1]
max_y = T_trans[0][0,0][0]
min_x = T_trans[0][0,0][1]
min_y = T_trans[0][0,0][0]

for k in range(num_images):
    for i in range(W):
        for j in range(H):
            if T_trans[k][j,i][0] > max_y:
                max_y = T_trans[k][j,i][0]
            if T_trans[k][j,i][0] < min_y:
                min_y = T_trans[k][j,i][0]
            
            if T_trans[k][j,i][1] > max_x:
                max_x = T_trans[k][j,i][1]
            if T_trans[k][j,i][1] < min_x:
                min_x = T_trans[k][j,i][1]

max_x_canvas = max_x - min_x
max_y_canvas = max_y - min_y

max_x_tmp = int(np.ceil(max_x_canvas))
max_y_tmp = int(np.ceil(max_y_canvas))

Img_T = np.array([[]]*max_x_tmp*max_y_tmp + [[1]],dtype=object)[:-1]
Img_T = Img_T.reshape((max_y_tmp,max_x_tmp))

T_rev = np.array([[]]*max_x_tmp*max_y_tmp*num_images + [[1]],dtype=object)[:-1]
T_rev = T_rev.reshape((num_images,max_y_tmp,max_x_tmp))
for k in range(num_images):
    for i in range(max_x_tmp):
        for j in range(max_y_tmp):
            Img_T[j,i] = [j,i]
            Img_T[j,i] = [j,i]
            T_rev[k][j,i] = set_T(i,j,z_rev[k],avg_corrs)       #i is x, j is y
            T_rev[k][j,i] = set_T(i,j,z_rev[k],avg_corrs)
            
stuffx = np.asarray(list(range(W)))
stuffy = np.asarray(list(range(H)))
x_inter = np.zeros((num_images,W,1))
y_inter = np.zeros((num_images,H,1))
z_new = np.zeros((num_images,max_y_tmp,max_x_tmp))
for k in range(num_images):
    for i in range(W):
        x_inter[k][i][0] = i
       
for k in range(num_images):
    for j in range(H):
        y_inter[k][j][0] = j

z_images = np.zeros((num_images,H,W))
for i in range(num_images):
    z_images[i] = img_list[i]

x_new = np.zeros((num_images,max_x_tmp*max_y_tmp,1))
y_new = np.zeros((num_images,max_x_tmp*max_y_tmp,1))

f = [None]*num_images
for k in range(num_images):
    f[k] = interp2d(x_inter[k],y_inter[k],z_images[k],kind = 'cubic')

idx = 0
for k in range(num_images):
    for i in range(max_x_tmp):
        for j in range(max_y_tmp):
            x_new[k][idx] = T_rev[k][j,i][1]
            y_new[k][idx] = T_rev[k][j,i][0]
            z_new[k][j,i] = f[k](x_new[k][idx],y_new[k][idx])
            idx=idx+1
    idx = 0

z_= 0
canvas = np.zeros((max_y_tmp,max_x_tmp))
std = np.zeros((max_y_tmp,max_x_tmp))
for i in range(max_x_tmp):
    for j in range(max_y_tmp):
        for k in range(num_images):
            z_ = z_ + z_new[k][j,i]
        canvas[j,i] = (z_)/(num_images)
        
        z_ = 0
    
stuff_=0
for i in range(max_x_tmp):
    for j in range(max_y_tmp):
        for k in range(num_images):
            stuff_ = stuff_ + (z_new[k][j,i] - canvas[j,i])**2
        stuff_ = stuff_/num_images
        stuff_ = np.sqrt(stuff_)
        std[j,i] = stuff_
        
        stuff_ = 0
        
stuffx = list(range(155,max_x_tmp))
stuffy = list(range(160, max_y_tmp))
canvas = np.delete(canvas, stuffx, 1)
canvas = np.delete(canvas,stuffy,0)
std = np.delete(std, stuffx, 1)
std = np.delete(std,stuffy,0)

plt.figure(1)
plt.imshow(canvas,cmap='gray')
for i in range(num_corrs):
    plt.scatter(avg_corrs[i][1],avg_corrs[i][0],c='#ff7f0e')
print("The output image is {data['Output file']}")
plt.savefig(output_name)

plt.figure(2)
plt.imshow(std,cmap='gray')
plt.savefig("Atlas - STD")

plt.figure(3)
image_FloodFill = flood_fill(canvas, (50, 50), 15, tolerance=0.01)
# all_labels = measure.label(image_FloodFill)
# removed_img = morphology.remove_small_objects(all_labels,1000)
plt.imshow(image_FloodFill, cmap="nipy_spectral")
plt.savefig("Atlas - Connected")


