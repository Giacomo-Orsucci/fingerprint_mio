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


#program to evaluate the accuracy on a set of images already generated
#experimentally the accuracy that we have in this way is slightly different (+-0.005) on the total
#if we compare it with the accuracy obtained from a set of images generated and directly analized without saving them before

parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)

parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)

#Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0]).to(device) #embedded fingerprint



args = parser.parse_args()

IMAGE_RESOLUTION = args.image_resolution
IMAGE_CHANNELS = 3


FINGERPRINT_SIZE = len(fingerprint)


image_directory = '/media/giacomo/hdd_ubuntu/progan_gen_50k'

RevealNet = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet.load_state_dict(torch.load(args.decoder_path))
RevealNet = RevealNet.to(device)
RevealNet.eval()
bitwise_accuracy = 0


j=0
for filename in os.listdir(image_directory):
    
    j = j+1

   #if j==1000:
        #break;


    print(j)


    img_path = os.path.join(image_directory, filename)
           

    img_path = os.path.join(image_directory, filename)
    image = cv2.imread(img_path, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #to convert in rgb
    image_rgb_array = np.array(image_rgb) #to convert in array
    image_rgb_tensor = torch.from_numpy(image_rgb_array).permute(2, 0, 1).float().to(device) #to convert in tensor
    
    print(image)

    detected_fingerprints = RevealNet(image_rgb_tensor.unsqueeze(0))

    #"True" if the element is > 0 and "False" otherwise
    detected_fingerprints = (detected_fingerprints > 0).long()
    fingerprint = (fingerprint > 0).long()

    detected_fingerprints.to(device)
    fingerprint.to(device)

    print(fingerprint)
    
    print(detected_fingerprints)
    
    bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()

    

bitwise_accuracy = bitwise_accuracy / (j) #compute the general accuracy

print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")
    
print("Successfully terminated")