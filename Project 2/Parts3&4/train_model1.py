#Ryan Miller
#Image Processsing Project 2
#3/4
#Model Number 1

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
from skimage import io
from skimage import color
import skimage
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import sys 
sys.path.append("../../utils")
from utils import NoiseDatsetLoader

dtype = torch.float32
#you can change this to "cuda" to run your code on GPU
cpu = torch.device('cuda')


def checkTestingAccuracy(dataloader,model):
    ## This Function Checks Accuracy on Testing Dataset for the model passed
    ## This function should return the loss the Testing Dataset

    ## Before running on Testing Dataset the model should be in Evaluation Mode    
    model.eval()
    totalLoss = []
    loss_mse = nn.MSELoss()
    for t, temp in enumerate(dataloader):
        NoisyImage = temp['NoisyImage'].to(device=cpu,dtype=dtype)
        referenceImage = temp['image'].to(device=cpu,dtype=dtype)

        ## For Each Test Image Calculate the MSE loss with respect to Reference Image 
        ## Return the mean the total loss on the whole Testing Dataset
        
        ## ************* Start of your Code *********************** ##
        
        output = model(NoisyImage)
        loss = loss_mse(output,referenceImage)
        
        #append mean square error to total loss
        totalLoss.append(loss.cpu().detach().numpy())
      
    #return average of total loss array
    return np.mean(totalLoss)

        ## ************ End of your code ********************   ##


def trainingLoop(dataloader,model,optimizer,nepochs):
    ## This function Trains your model for 'nepochs'
    ## Using optimizer and model your passed as arguments
    ## On the data present in the DataLoader class passed
    ##
    ## This function return the final loss values

    model = model.to(device=cpu)    
    
    ## Our Loss function for this exercise is fixed to MSELoss
    loss_function = nn.MSELoss()
    loss_array =[]
    loss_per_epoch_array=[]
    for e in range(nepochs):
            print("Epoch", e)
            for t, temp in enumerate(dataloader):
                ## Before Training Starts, put model in Training Mode
                model.train()  
                #send the inputs to GPU -> NoisyImage is the image and Ref is target
                NoisyImage = temp['NoisyImage'].to(device=cpu,dtype=dtype)
                referenceImage = temp['image'].to(device=cpu,dtype=dtype)

                ## ************* Start of your Code *********************** ##
               
                #Forward pass through the model
                output = model(NoisyImage)

                #loss value using ground truth and output image
                loss = loss_function(output,referenceImage)   
                
                #set gradients to zero 
                optimizer.zero_grad()
                
                #backward pass 
                loss.backward()
                
                #optimizer step 
                optimizer.step()
                
                ## ************ End of your code ********************   ##
                loss_array.append(loss.cpu().detach().numpy())
            print("Training loss: ",loss) 
            loss_per_epoch_array.append(loss.cpu().detach().numpy()) #works w/o bracket
    return loss, loss_array, loss_per_epoch_array



def main():
    TrainingSet = NoiseDatsetLoader(csv_file='TrainingDataSet.csv', root_dir_noisy='TrainingDataset')
    TestingSet  = NoiseDatsetLoader(csv_file='TestingDataSet.csv' , root_dir_noisy='TestingDataset')

    ## Batch Size is a hyper parameter, You may need to play with this paramter to get a more better network
    batch_size=32

    ## DataLoader is a pytorch Class for iterating over a dataset
    dataloader_train  = torch.utils.data.DataLoader(TrainingSet,batch_size=batch_size,num_workers=4)
    dataloader_test   = torch.utils.data.DataLoader(TestingSet,batch_size=1)

    ## ************* Start of your Code *********************** ##

    #model1
    
    #just conv2d and relu work
    
    model =  torch.nn.Sequential(torch.nn.Conv2d(1,1,kernel_size = (5,5),padding = 2))      #padding has to be double +1 

    ## ************ End of your code ********************   ##
   
    ## Optimizer
    ## ************* Start of your Code *********************** ##

    # for 10 epochs, LR = 1e-2, 
    learning_rate = 1e-3    
    weight_decay  = 1e-2
    epochs        = 200

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    ## ************ End of your code ********************   ##
    
    ## Train Your Model. Complete the implementation of trainingLoop function above 
    start_time = time.time()
    valMSE, loss_array, loss_array_per_epoch = trainingLoop(dataloader_train,model,optimizer,epochs)
    print("Total Runtime", time.time()-start_time)

    ## Test Your Model. Complete the implementation of checkTestingAccuracy function above 
    testMSE = checkTestingAccuracy(dataloader_test,model)
    print("Mean Square Error for the testing Set for the trained model is ", testMSE)
    
    model.eval() # Put you model in Eval Mode

    ## ************* Start of your Code *********************** ##

    fig = plt.figure(figsize = (12,12))
    rows = 1
    columns = 1
    
    epoch_array = list(range(1,epochs+1))
    
    #plotting loss array
    fig.add_subplot(rows,columns,1)
    plt.plot(epoch_array,loss_array_per_epoch)
    plt.title('Loss During Training', fontsize = 18)
    plt.xlabel('Number of Epochs', fontsize = 18)
    plt.ylabel('Loss (MSE)', fontsize = 18)
    plt.savefig("Epoch Graph - Model 1.png")


    ## ************ End of your code ********************   ##
    
    ## Plot some of the Testing Dataset images by passing them through the trained model
    
    for i in range(5): 
        fig = plt.figure(figsize = (12,12))
        rows = 1
        columns = 2
        
        idx = np.random.randint(0,len(TestingSet))
        testing_pic = TestingSet[idx]
        
        #NoisyImage
        testing_NoisyImage = torch.from_numpy(testing_pic['NoisyImage'])
        testing_NoisyImage = testing_NoisyImage.to(device=cpu,dtype=dtype)
        testing_NoisyImage = testing_NoisyImage.unsqueeze(0)
        
        #FilteredImage
        filtered_NoisyImage = model(testing_NoisyImage)
        filtered_NoisyImage = filtered_NoisyImage.squeeze()
        
        fig.add_subplot(rows,columns,1)
        plt.imshow(torch.squeeze(testing_NoisyImage).cpu().numpy(),cmap="gray")
        plt.title('Noisy Image')
        
        fig.add_subplot(rows,columns,2)
        plt.imshow(filtered_NoisyImage.cpu().detach().numpy(),cmap="gray")
        plt.title('Filtered Image')
        
        plt.savefig("Filtered Images Model1 - #%d.png" %i)
        
    #plot my own images from part 1
    img = io.imread("Jerry.jpg")
    ref_image = color.rgb2gray(img)
    
    mean = 0.1
    var = 0.1
    
    #Gauss
    fig = plt.figure(figsize = (12,12))
    rows = 1
    columns = 1
    noisy_image = skimage.util.random_noise(ref_image,mode='gaussian', mean = mean, var = var)
    
    testing_noisy_image = torch.from_numpy(noisy_image)
    testing_noisy_image = testing_noisy_image.to(device=cpu,dtype=dtype)
    
    testing_noisy_image = testing_noisy_image.unsqueeze(0)
    testing_noisy_image = testing_noisy_image.unsqueeze(0)
    test_ref_noise = model(testing_noisy_image)
    test_ref_noise = test_ref_noise.squeeze()
    
    fig.add_subplot(rows,columns,1)
    plt.imshow(test_ref_noise.cpu().detach().numpy(),cmap="gray")
    plt.title('Running Noisy Image through Model', fontsize = 18)
    plt.savefig("Filtered Image through Model1.png")
    
    
#add x label for learning rate, epochs, weight decay, batch size

    ## ************ End of your code ********************   ##

if __name__ == "__main__":

    main()

