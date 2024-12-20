import argparse
import os
import glob
import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt

#to get parameters inserted via CLI

parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument(
    "--encoder_path", type=str, help="Path to trained StegaStamp encoder."
)
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)
parser.add_argument(
    "--identical_fingerprints", action="store_true", help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints."
)
parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")
parser.add_argument("--cuda", type=int, default=0)


args = parser.parse_args()

#if the folder doesn't exist, creates it
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size

#devices list it's ordered following the PCI order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image



def generate_random_fingerprints(fingerprint_size, batch_size=4):

    #2 excluded, it creates a tensor of 0 and 1 with batch_size x fingerprint_size size
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z

#generate a uniform distribution
uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)


if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")

torch.cuda.empty_cache()

    
print("Device used: ")
print(device)

class CustomImageFolder(Dataset):
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

#to load the dataset
#applies all the preprocessing needed by celebA, if we are using it as dataset
def load_data():
    global dataset, dataloader

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

#to load the pretrained models
def load_models():
    global HideNet, RevealNet
    global FINGERPRINT_SIZE
    
    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 1

    from models import StegaStampEncoder, StegaStampDecoder

    state_dict = torch.load(args.encoder_path)
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder( #encoder and parameters passing
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False,
    )
    RevealNet = StegaStampDecoder( #decoder and parameters passing
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )

    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:#to load state_dictionary and model to the specified device
        RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs)
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs))

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)

#to embed the fingerprint on the dataset provided
def embed_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    #generate identical fingerprints
    fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, 1)
    
    fingerprints = fingerprints.view(1, FINGERPRINT_SIZE).expand(BATCH_SIZE, FINGERPRINT_SIZE) 
    fingerprints = fingerprints.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed) 

    bitwise_accuracy = 0

    for images, _ in tqdm(dataloader):

        #generate arbitrary fingerprints
        if not args.identical_fingerprints:
            fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, BATCH_SIZE)
            fingerprints = fingerprints.view(BATCH_SIZE, FINGERPRINT_SIZE)
            fingerprints = fingerprints.to(device)

        images = images.to(device)
        

        
        #preprocessing for rgb images to fingerprinting only the luminance
        new_images = []
        u = []
        v = []
        for image in images:
                
            # transpose from (3, 128, 128) to (128, 128, 3) to visualize the image properly.
            image_rgb = np.transpose(image.cpu().numpy(), (1, 2, 0))
            image_rgb = np.array(image_rgb)
            image_yuv =  cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)

            # to split the image in it's Y, U and V channels
            y_channel, u_channel, v_channel = cv2.split(image_yuv)

            u.append(u_channel)
            v.append(v_channel)
            y_channel_only = np.expand_dims(y_channel, axis=2) #from (128,128) to (128, 128, 1)
            image = torch.from_numpy(y_channel_only)
            image = np.transpose(image, (2, 0, 1)) #from (128,128,1) a (1,128,128)
            new_images.append(image)
            #print(image.shape)

        new_images = torch.stack(new_images)
        images = new_images
        images = images.to(device)




        
        
        fingerprinted_images = HideNet(fingerprints[: images.size(0)], images) #fingerprinted on luminance (y)

        #print("shape")
        #print(fingerprinted_images.shape)
        

        if args.check:
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            
            #to calculate the accuracy in retrieving the fingerprint (eventually perturbated)
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()


        #print("Dim di fingerprinted_images")
        #print(fingerprinted_images.shape)
       
        
        
        new_fingerprinted_images = torch.empty((0, 3, 128, 128)).to(device)
        for i, fin_image in enumerate(fingerprinted_images):
          
            u_app = np.expand_dims(u[i], axis=2)
            v_app = np.expand_dims(v[i], axis=2)

            fin_image = np.transpose(fin_image.cpu().detach().numpy(), (1, 2, 0))
            
            final_image = np.concatenate((fin_image, u_app, v_app),2)
            final_image = cv2.cvtColor(final_image, cv2.COLOR_YUV2RGB)

            final_image = np.transpose(final_image, (2, 0, 1)) #from (128,128,1) a (1,128,128)
            final_image = torch.from_numpy(final_image).to(device)

            

             
            new_fingerprinted_images = torch.cat((new_fingerprinted_images, final_image.unsqueeze(0)), dim=0)
            
        fingerprinted_images = new_fingerprinted_images


        all_fingerprinted_images.append(fingerprinted_images.detach().cpu()) 
        all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())




    #if the folder for the output doesn't exist, creates it
    dirname = args.output_dir
    if not os.path.exists(os.path.join(dirname, "fingerprinted_images")):
        os.makedirs(os.path.join(dirname, "fingerprinted_images"))

    #saves the fingerprinted images and the fingerprints associated
    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "embedded_fingerprints.txt"), "w")
    for idx in range(len(all_fingerprinted_images)):
        image = all_fingerprinted_images[idx]
        fingerprint = all_fingerprints[idx]
        #splits and discards the first part of the path
        _, filename = os.path.split(dataset.filenames[idx]) 
        filename = filename.split('.')[0] + ".png"
        save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{filename}"), padding=0)
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        f.write(f"{filename} {fingerprint_str}\n")



    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_fingerprints) #calcola l'accuratezza generale
        print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")
        #save the first 49 images organized in rows of 7 elements
        save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        #save the first 49 fingerprinted images organized in rows of 7 elements
        save_image(fingerprinted_images[:49], os.path.join(args.output_dir, "test_samples_fingerprinted.png"), nrow=7)
        #save the first 49 "residual" images organized in rows of 7 elements
        save_image(torch.abs(images - fingerprinted_images)[:49], os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=7)


def main():

    load_data()
    load_models()

    embed_fingerprints()
    print("embedding finished")


if __name__ == "__main__":
    main()
