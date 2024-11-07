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

"""
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
"""

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


#image_directory = '/media/giacomo/hdd_ubuntu/old/celeba_fin_old_200k'
#image_directory = '/media/giacomo/hdd_ubuntu/new/stylegan2_gen_50k_config-e_10'
#image_directory = '/media/giacomo/hdd_ubuntu/no_rand/celeba_I_meta_enc_1_norand/fingerprinted_images'
#image_directory = '/home/giacomo/Desktop/fin1/fingerprinted_images'
#image_directory = '/media/giacomo/volume/no_rand/stylegan2_gen_norand1_50k_config-e_25'
image_directory = '/media/giacomo/volume/test_100_gen'
#dec_path_pre = '/home/giacomo/Desktop/enc_dec_pretrained_celeba/dec.pth'
#dec_path_old = '/media/giacomo/hdd_ubuntu/old/trained_byme/dec.pth'
#dec_path_new = '/media/giacomo/hdd_ubuntu/new/dec.pth'

#trio used to test the fingerprinted celeba with enc_norand_1
#dec_old
dec_path_pre = '/media/giacomo/volume/old/trained_byme/dec.pth'
#dec_norand_1
dec_path_old = '/media/giacomo/volume/no_rand/enc-dec_1_20/checkpoints/dec.pth'
#dec_norand_2
dec_path_new = '/media/giacomo/volume/no_rand/enc-dec_2_20/checkpoints/dec.pth'

RevealNet_pre = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet_pre.load_state_dict(torch.load(dec_path_pre))
RevealNet_pre = RevealNet_pre.to(device)
RevealNet_pre.eval()


RevealNet_old = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet_old.load_state_dict(torch.load(dec_path_old))
RevealNet_old = RevealNet_old.to(device)
RevealNet_old.eval()


RevealNet_new = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet_new.load_state_dict(torch.load(dec_path_new))
RevealNet_new = RevealNet_new.to(device)
RevealNet_new.eval()


bitwise_accuracy_pre = 0
bitwise_accuracy_old = 0
bitwise_accuracy_new = 0


j=0
for filename in os.listdir(image_directory):
    
    j = j+1

    #if j==10:
        #break;

    print(j)

    img_path = os.path.join(image_directory, filename)
    image = cv2.imread(img_path, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #to convert in rgb
    image_rgb_array = np.array(image_rgb) #to convert in array
    image_rgb_tensor = torch.from_numpy(image_rgb_array).permute(2, 0, 1).float().to(device) #to convert in tensor
    
    print(image)

    detected_fingerprints_pre = RevealNet_pre(image_rgb_tensor.unsqueeze(0))
    detected_fingerprints_old = RevealNet_old(image_rgb_tensor.unsqueeze(0))
    detected_fingerprints_new = RevealNet_new(image_rgb_tensor.unsqueeze(0))

    #"True" if the element is > 0 and "False" otherwise
    detected_fingerprints_pre = (detected_fingerprints_pre > 0).long()
    detected_fingerprints_old = (detected_fingerprints_old > 0).long()
    detected_fingerprints_new = (detected_fingerprints_new > 0).long()
    
    
    fingerprint = (fingerprint > 0).long()

    detected_fingerprints_pre.to(device)
    detected_fingerprints_new.to(device)
    detected_fingerprints_old.to(device)
    fingerprint.to(device)

    #print(fingerprint)
    
    #print(detected_fingerprints_pre)
    
    bitwise_accuracy_pre += (detected_fingerprints_pre == fingerprint).float().mean(dim=1).sum().item()
    bitwise_accuracy_old += (detected_fingerprints_old == fingerprint).float().mean(dim=1).sum().item()
    bitwise_accuracy_new += (detected_fingerprints_new == fingerprint).float().mean(dim=1).sum().item()

    

bitwise_accuracy_pre = bitwise_accuracy_pre / (j) #compute the general accuracy
bitwise_accuracy_old = bitwise_accuracy_old / (j) #compute the general accuracy
bitwise_accuracy_new = bitwise_accuracy_new / (j) #compute the general accuracy

print(f"Bitwise accuracy on fingerprinted images with dec_pre: {bitwise_accuracy_pre}")
print(f"Bitwise accuracy on fingerprinted images with dec_old: {bitwise_accuracy_old}")
print(f"Bitwise accuracy on fingerprinted images with dec_new: {bitwise_accuracy_new}")
    
print("Successfully terminated")