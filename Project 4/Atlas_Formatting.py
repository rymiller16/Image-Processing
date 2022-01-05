#Ryan Miller
#u1067596
#Image Processing 
#Dr. Whitaker

import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color

def Atlas_Formatting(atlas_input):
    with open(atlas_input) as f:
        data = json.load(f)

    files = data['Input files']
    #print(f"The images are {data['Input files']}")
    corrs = data["Correspondences"]
    corrs_array = np.asarray(corrs)
    #print(f"There are {corrs_array.shape[0]} sets of correspondences across {corrs_array.shape[1]} images in a  {corrs_array.shape[2]}-dimensional domain")
    #print(f"The output file is {data['Output file']}")

    for i in range(len(files)):
        these_corrs = corrs_array[:,i,:]
        imageName = files[i]
        imageNameShort = (imageName.split('.'))[0]
        image = io.imread(imageName)
        #fig, ax = plt.subplots(1,1)
        #fig.set_size_inches(10.5,10.5)
        #ax.imshow(image)
        #ax.scatter(these_corrs[:,1],these_corrs[:,0])
        #plt.savefig(imageNameShort + "_correspondences"+'.png')
    
    num_images = len(data['Input files'])
    img_names = []
    img_list = []
    for i in range(num_images):
        img_names.append(data['Input files'][i])
        img_list.append(color.rgb2gray(io.imread(img_names[i])))
    
    output_name = data['Output file']
    num_corrs = len(corrs)
    
    return num_images, img_names, img_list, corrs, corrs_array, files, output_name, num_corrs