import argparse
import glob
import PIL
import cv2

#to get parameters inserted via CLI

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution",
    type=int,
    required=True,
    help="Height and width of square images.",
)
parser.add_argument(
    "--decoder_path",
    type=str,
    required=True,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)


args = parser.parse_args()

import os

#devices list it's ordered following the PCI order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np

if args.cuda != -1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


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

#to load the pretrained models
def load_decoder():
    global RevealNet
    global FINGERPRINT_SIZE

    from models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path)
    FINGERPRINT_SIZE = state_dict["dense.2.weight"].shape[0]

    RevealNet = StegaStampDecoder(args.image_resolution, 1, FINGERPRINT_SIZE)
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path, **kwargs))
    RevealNet = RevealNet.to(device)


def load_data():
    global dataset, dataloader

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

#to retrieve fingerprints and save them 
def extract_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    BATCH_SIZE = args.batch_size
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for images, _ in tqdm(dataloader):
        images = images.to(device)
        print("shape di images")
        print(images.shape)

        app = []
        for image in images:
            image = image.to(device)
            image = image.permute(1, 2, 0).cpu().numpy() 
            

            yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

            # Estrai il canale Y
            y_channel, _, _ = cv2.split(yuv_image)
            y_tensor = torch.from_numpy(y_channel).unsqueeze(0).float()
            app.append(y_tensor)

       
        images_y_batch = torch.stack(app).to(device)
        print("Size di batch")
        print(images_y_batch.shape)
        images = images_y_batch
        
        fingerprints = RevealNet(images)
        fingerprints = (fingerprints > 0).long()

        all_fingerprinted_images.append(images.detach().cpu())
        all_fingerprints.append(fingerprints.detach().cpu())

    dirname = args.output_dir
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "detected_fingerprints.txt"), "w")
    for idx in range(len(all_fingerprints)):
        fingerprint = all_fingerprints[idx]
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".png"
        f.write(f"{filename} {fingerprint_str}\n")
    f.close()


if __name__ == "__main__":
    load_decoder()
    load_data()
    extract_fingerprints()
