from models import StegaStampDecoder
import torch

import torchvision.transforms as transforms
import torchvision
import numpy as np

import torch
from torchvision.utils import save_image
import os
import argparse
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import PIL

parser = argparse.ArgumentParser()

transform = transforms.Compose([
    transforms.ToTensor()  # Convert the image to a tensor
])


#Program to evaluate the accuracy on a set of images already generated.
#Experimentally the accuracy that we have in this way is slightly different (+-0.005) on the total
#if we compare it with the accuracy obtained from a set of images generated and directly analized without saving them before.
#Something is parametrized from console and something not. The reason is that the code is freely accessible 
#and something was better to be modified in the code that parametrized. So, the suggestion is to read the code
#and to modify manually the highlighted paths and variables depending on your necessities.



class CustomImageFolder():
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir #path with the dataset for the training specified via CLI
        self.filenames = glob.glob(os.path.join(data_dir, "*.png")) #to get all the png image's paths 
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg"))) #to add all the jpeg images' path 
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg"))) #to add all the jpg images' path 
        self.filenames = sorted(self.filenames) #order the file name in ascendent order
        self.transform = transform

    #return the image at the specified index
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)

#Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Example of fingerprints that we except to be embedded and obtained from the generated images.
#The fingerprint here defined is used to calculate the accuracy.
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0]).to(device) #embedded fingerprint with seed 42
                    

"""
fingerprint = torch.tensor([0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,0,
                            1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,0,1,
                            0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1]).to(device) #embedded fingerprint with seed 42_3
                            """
args = parser.parse_args()

IMAGE_RESOLUTION = args.image_resolution
IMAGE_CHANNELS = 1


FINGERPRINT_SIZE = len(fingerprint)

#insert the path with the generated images to decode
image_directory=''



#here we don't test and compares three decoders, but only one. Anyway, it is possible to replicate
#the comparison from the others branches 

dec1 = ''

RevealNet_dec1 = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet_dec1.load_state_dict(torch.load(dec1))
RevealNet_dec1 = RevealNet_dec1.to(device)
RevealNet_dec1.eval()


bitwise_accuracy = 0;


dataset = CustomImageFolder(image_directory, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)



j=0
for images, _ in tqdm(dataloader):
    #print(images.shape)
    y_channel_list = []

    for image in images:
        image = image.permute(1, 2, 0).cpu().numpy()
        #print("Single image")
        #print(image.shape)
        #image = (image * 255).astype(np.uint8)

        #from RGB to YUV
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        #print(image_yuv)
        y_channel, u_channel, v_channel = cv2.split(image_yuv)
        image = y_channel
        image = torch.from_numpy(image).unsqueeze(0)
        #print("image shape")
        #print(image.shape)
        y_channel_list.append(image)
    
    images_y_batch = torch.stack(y_channel_list).to(device)


    print("shape di batch")
    print(images_y_batch.shape)


    detected_fingerprints = RevealNet_dec1(images_y_batch)
    detected_fingerprints = (detected_fingerprints > 0).long()

    print("detected fingerprint")
    print(detected_fingerprints.shape)
    print(detected_fingerprints[0])
            
    
    for i in enumerate(detected_fingerprints):
        j = j + 1
        #to calculate the accuracy in retrieving the fingerprint (eventually perturbated)
        bitwise_accuracy += (detected_fingerprints[i].detach() == fingerprint).float().mean().sum().item()
        #print(bitwise_accuracy)

   
    
print(j)
print(bitwise_accuracy/j)