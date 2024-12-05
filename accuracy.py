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


#program to evaluate the accuracy on a set of images already generated
#experimentally the accuracy that we have in this way is slightly different (+-0.005) on the total
#if we compare it with the accuracy obtained from a set of images generated and directly analized without saving them before



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
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0]).to(device) #embedded fingerprint with seed 42
                    

"""

fingerprint = torch.tensor([0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,1,1,0,0,
                            0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0,1,1,1,0,1,
                            0,1,0,0,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,1,0]).to(device) #embedded fingerprint with seed 49
"""
                       

"""
fingerprint = torch.tensor([0,0,1,1,0,0,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,
                            1,0,0,0,1,0,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,1,0,0,1,0,1,1,0,0,1,0,
                            0,1,1,0,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,0]).to(device) #embedded fingerprint with seed 75
"""


"""
fingerprint = torch.tensor([0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,0,
                            1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,0,1,
                            0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1]).to(device) #embedded fingerprint with seed 42_3
                            """
args = parser.parse_args()

IMAGE_RESOLUTION = args.image_resolution
IMAGE_CHANNELS = 1


FINGERPRINT_SIZE = len(fingerprint)


#image_directory = '/media/giacomo/hdd_ubuntu/test_yuv/celeba'
#image_directory='/media/giacomo/hdd_ubuntu/test_yuv/test_celeab/fingerprinted_images'
#image_directory='/media/giacomo/volume/test_yuv/test'
#image_directory='/media/giacomo/volume/test_yuv/robustness/gau_noise_std_0-100_style2_25_50k/0'
#image_directory='/media/giacomo/volume/yuv_base/test'
#image_directory='/media/giacomo/volume/yuv_base/stylegan2_gen_50k_config-e_25_seed49'
#image_directory='/media/giacomo/volume/yuv_base/prova_42_dataset'
#image_directory='/media/giacomo/volume/yuv_base/test_fin_42'
#image_directory='/media/giacomo/volume/test_yuv/test_a'
#image_directory='/media/giacomo/volume/yuv_base/test_fin_75'
#image_directory='/media/giacomo/volume/yuv_base/stylegan2_gen_50k_config-e_75_seed42'
image_directory='/media/giacomo/volume/yuv_norand/stylegan2_gen_50k_config-e_25'



#the program is thought to make comparison beetwen different decoder and
#fingerprinted datasets, but for the moment is not necessary this comparison

#dec_path_pre = '/media/giacomo/volume/test_yuv/primo/checkpoints/dec.pth'
#dec_path_pre = '/media/giacomo/volume/yuv_base/enc-dec/checkpoints/dec.pth'
dec_path_pre = '/media/giacomo/volume/yuv_norand/enc-dec/checkpoints/dec.pth' #dec_norand1
#dec_path_old = '/media/giacomo/volume/test_yuv/primo/checkpoints/dec.pth'
#dec_path_new = '/media/giacomo/volume/test_yuv/primo/checkpoints/dec.pth'

RevealNet_pre = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )
RevealNet_pre.load_state_dict(torch.load(dec_path_pre))
RevealNet_pre = RevealNet_pre.to(device)
RevealNet_pre.eval()

"""
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
"""

bitwise_accuracy = 0;


dataset = CustomImageFolder(image_directory, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)



j=0
for images, _ in tqdm(dataloader):
    #print(images.shape)
    y_channel_list = []

    for image in images:
        image = image.permute(1, 2, 0).cpu().numpy()
        print("Singola immagine")
        print(image.shape)
        #image = (image * 255).astype(np.uint8)

        # Converti l'immagine RGB in YUV usando OpenCV
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        #print(image_yuv)
        y_channel, u_channel, v_channel = cv2.split(image_yuv)
        image = y_channel
        image = torch.from_numpy(image).unsqueeze(0)
        print("shape di image")
        print(image.shape)

        
        y_channel_list.append(image)
    
    images_y_batch = torch.stack(y_channel_list).to(device)


    print("shape di batch")
    print(images_y_batch.shape)


    detected_fingerprints = RevealNet_pre(images_y_batch)
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