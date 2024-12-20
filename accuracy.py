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


parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)

#Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Example of fingerprints that we except to be embedded and obtained from the generated images.
#The fingerprint here defined is used to calculate the accuracy.

"""
fingerprint = torch.tensor([0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,0,
                            1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,0,1,
                            0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1]).to(device) #embedded fingerprint that you are expecting to find (from seed=42_3 in this case)
"""
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,0,0,0,
                            0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,
                            0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0]).to(device) #embedded fingerprint that you are expecting to find (from seed=42 in this case)





args = parser.parse_args()

IMAGE_RESOLUTION = args.image_resolution
IMAGE_CHANNELS = 3


FINGERPRINT_SIZE = len(fingerprint)

#insert the path with the generated images to decode

image_directory=''

#the idea is to test three decoders at the same time. If you want to test only one decoder, comment the other two
#or use the same path on all the three

dec1 = ''
dec2 = ''
dec3 = ''

RevealNet_1 = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet_1.load_state_dict(torch.load(dec1))
RevealNet_1 = RevealNet_1.to(device)
RevealNet_1.eval()


RevealNet_2 = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet_2.load_state_dict(torch.load(dec2))
RevealNet_2 = RevealNet_2.to(device)
RevealNet_2.eval()


RevealNet_3 = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet_3.load_state_dict(torch.load(dec3))
RevealNet_3 = RevealNet_3.to(device)
RevealNet_3.eval()


bitwise_accuracy_dec1 = 0
bitwise_accuracy_dec2 = 0
bitwise_accuracy_dec3 = 0


j=0
for filename in os.listdir(image_directory):
    
    j = j+1

    #uncomment it to have a preview on the accuracy calculating it on a subset of images from the folder
    #if j==10: break;

    #print(j)

    img_path = os.path.join(image_directory, filename)
    image = cv2.imread(img_path, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #to convert in rgb
    image_rgb_array = np.array(image_rgb) #to convert in array
    image_rgb_tensor = torch.from_numpy(image_rgb_array).permute(2, 0, 1).float().to(device) #to convert in tensor
    
    #print(image)

    detected_fingerprints_dec1 = RevealNet_1(image_rgb_tensor.unsqueeze(0))
    detected_fingerprints_dec2 = RevealNet_2(image_rgb_tensor.unsqueeze(0))
    detected_fingerprints_dec3 = RevealNet_3(image_rgb_tensor.unsqueeze(0))

    #"True" if the element is > 0 and "False" otherwise
    detected_fingerprints_dec1 = (detected_fingerprints_dec1 > 0).long()
    detected_fingerprints_dec2 = (detected_fingerprints_dec2 > 0).long()
    detected_fingerprints_dec3 = (detected_fingerprints_dec3 > 0).long()
    
    
    fingerprint = (fingerprint > 0).long()

    detected_fingerprints_dec1.to(device)
    detected_fingerprints_dec2.to(device)
    detected_fingerprints_dec3.to(device)
    fingerprint.to(device)

    #print(fingerprint)
    
    #print(detected_fingerprints_pre)
    
    bitwise_accuracy_dec1 += (detected_fingerprints_dec1 == fingerprint).float().mean(dim=1).sum().item()
    bitwise_accuracy_dec2 += (detected_fingerprints_dec2 == fingerprint).float().mean(dim=1).sum().item()
    bitwise_accuracy_dec3 += (detected_fingerprints_dec3 == fingerprint).float().mean(dim=1).sum().item()

    

bitwise_accuracy_dec1 = bitwise_accuracy_dec1 / (j) #compute the general accuracy
bitwise_accuracy_dec2 = bitwise_accuracy_dec2 / (j) #compute the general accuracy
bitwise_accuracy_dec3 = bitwise_accuracy_dec3 / (j) #compute the general accuracy

print(f"Bitwise accuracy on fingerprinted images with dec1: {bitwise_accuracy_dec1}")
print(f"Bitwise accuracy on fingerprinted images with dec2: {bitwise_accuracy_dec2}")
print(f"Bitwise accuracy on fingerprinted images with dec3: {bitwise_accuracy_dec3}")
    
print("Successfully terminated")