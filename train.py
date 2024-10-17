import argparse


#to get parameters inserted via CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, required=True, help="Directory with image dataset."
)
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.",
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save results to."
)
parser.add_argument(
    "--fingerprint_length",
    type=int,
    default=100,
    required=True,
    help="Number of bits in the fingerprint.",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default=128,
    required=True,
    help="Height and width of square images.",
)
parser.add_argument(
    "--num_epochs", type=int, default=20, help="Number of training epochs."
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--cuda", type=str, default=0)

parser.add_argument(
    "--l2_loss_await",
    help="Train without L2 loss for the first x iterations",
    type=int,
    default=1000,
)
parser.add_argument(
    "--l2_loss_weight",
    type=float,
    default=10,
    help="L2 loss weight for image fidelity.",
)
parser.add_argument(
    "--l2_loss_ramp",
    type=int,
    default=3000,
    help="Linearly increase L2 loss weight over x iterations.",
)

parser.add_argument(
    "--BCE_loss_weight",
    type=float,
    default=1,
    help="BCE loss weight for fingerprint reconstruction.",
)

args = parser.parse_args()


import glob
import os
from os.path import join
from time import time

#devices list it's ordered following the PCI order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
from datetime import datetime

from tqdm import tqdm
import PIL

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.optim import Adam

import models
from torch.utils.data import SequentialSampler
import matplotlib.pyplot as plt
import numpy as np
import cv2

#paths where to save log, checkpoints, images 
LOGS_PATH = os.path.join(args.output_dir, "logs")
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "./saved_images")

writer = SummaryWriter(LOGS_PATH)

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
    os.makedirs(SAVED_IMAGES)

all_rand_fin = []
def generate_random_fingerprints(fingerprint_length, batch_size=4, size=(400, 400)):
    #2 excluded, it creates a tensor of 0 and 1 with batch_size x fingerprint_size size
    #use the following as default (original code)
    #z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)

    #use the following three lines of code to minimize the randomness
    #i use a seed to make the pseudo-random sequence generation everytime the same 
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
    all_rand_fin.append(z)

    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir #path with the dataset for the training specified via CLI
        self.filenames = glob.glob(os.path.join(data_dir, "*.png")) #to get all the png image's paths 
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg"))) #to add all the jpeg images' path 
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg"))) #to add all the jpg images' path
        self.filenames = sorted(self.filenames)  #order the file name in ascendent order
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
#applies all the preprocessing needed by celebA, if we are using it has dataset
def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 1 #version to embed fingerprint only on luminance Y

    SECRET_SIZE = args.fingerprint_length

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
                transforms.Resize(IMAGE_RESOLUTION),
                transforms.CenterCrop(IMAGE_RESOLUTION),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

# Funzione per stampare i pesi e bias di ogni layer di un modello
def print_model_weights(model):
    print(f"Stampa dei pesi del modello: {model.__class__.__name__}\n")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}")
            print(f"Pesi:\n{param.data}")  # Mostra i pesi del layer
            print(f"Forma: {param.shape}\n")
            


def main():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M:%S") #to formate the datetime
    EXP_NAME = f"stegastamp_{args.fingerprint_length}_{dt_string}" #to concatenate the info retrieved

    device = torch.device("cuda")

    load_data()
    encoder = models.StegaStampEncoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.fingerprint_length,
        return_residual=False,
    )
    decoder = models.StegaStampDecoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.fingerprint_length,
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)


    # Stampiamo i pesi di tutti i layer dell'encoder
    print_model_weights(encoder)

    #Stampiamo i pesi di tutti i layer del decoder
    print_model_weights(decoder)

    #we have the combination of encoder and decoder to update simultaneously decoder and encoder
    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )

    global_step = 0
    steps_since_l2_loss_activated = -1








    #we trained encoder and decoder for the specified number of epochs
    for i_epoch in range(args.num_epochs):

        #use the following ad default (original code)
        """
        dataloader = DataLoader( #to perform the batch fetch
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
        )
        """
        #use the following instead the above to minimize the randomization in image loading
        #the sequentialsampler is useful to reduce the randomness in the batches construction
        dataloader = DataLoader( #to perform the batch fetch
            dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset), num_workers=16
        )

        
        
        for images, _ in tqdm(dataloader): #generates a casual fingerprint for every batch 
            global_step += 1
            #print(images.shape)
            
            #to convert every image of celeba that is in rgb space color in yuv to embed the fingerprint only on the luminance y
            
            new_images = []
            for image in images:
                
                # transpose from (3, 128, 128) to (128, 128, 3) to visualize the image properly.
                image_rgb = np.transpose(image, (1, 2, 0))
                #plt.imshow(image_rgb)
                #plt.title("Test")
                #plt.show()

                image_rgb = np.array(image_rgb)
                image_yuv =  cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)

                # to split the image in it's Y, U and V channels
                y_channel, u_channel, v_channel = cv2.split(image_yuv)

                # setting U and V to 0 to create an image with only the Y channel
                #u_zero = np.zeros_like(u_channel)
                #v_zero = np.zeros_like(v_channel)

                #image_y_only = cv2.merge([y_channel, u_zero, v_zero])

                # conversion from YUV to RGB to visualize the image
                #image_gray = cv2.cvtColor(image_y_only, cv2.COLOR_YUV2RGB)

                y_channel_only = np.expand_dims(y_channel, axis=2) #from (128,128) to (128, 128, 1)

                #plt.imshow(y_channel_only)
                #plt.title('Immagine con solo canale Y (Luminanza)')
                #plt.show()

                image = torch.from_numpy(y_channel_only)
                image = np.transpose(image, (2, 0, 1))
                new_images.append(image)
                #print(image.shape)

            new_images = torch.stack(new_images)
            images = new_images

            #print(images.shape)


            batch_size = min(args.batch_size, images.size(0))
            fingerprints = generate_random_fingerprints(
                args.fingerprint_length, batch_size, (args.image_resolution, args.image_resolution)
            )

            l2_loss_weight = min( #loss computation
                max(
                    0,
                    args.l2_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / args.l2_loss_ramp,
                ),
                args.l2_loss_weight,
            )
            BCE_loss_weight = args.BCE_loss_weight

            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)

            #fingerprinting and residual
            fingerprinted_images = encoder(fingerprints, clean_images)
            residual = fingerprinted_images - clean_images

            decoder_output = decoder(fingerprinted_images)

            criterion = nn.MSELoss()
            l2_loss = criterion(fingerprinted_images, clean_images)

            criterion = nn.BCEWithLogitsLoss()
            #Binary Cross Entropy Loss
            BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))
            #the final mean is the averaged sum of the two losses
            loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

            
            #gradient put to zero, loss back-propagation and parameters update via optimizator
            encoder.zero_grad()
            decoder.zero_grad()

            loss.backward()
            decoder_encoder_optim.step()

            #bitiwise accuracy calculation beetwen original fingerprint and retrieved one
            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )
            if steps_since_l2_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    steps_since_l2_loss_activated = 0
            else:
                steps_since_l2_loss_activated += 1

            #logging and stats printing
            if global_step in plot_points:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy, global_step),
                print("Bitwise accuracy {}".format(bitwise_accuracy))
                writer.add_scalar("loss", loss, global_step),
                writer.add_scalar("BCE_loss", BCE_loss, global_step),
                writer.add_scalars(
                    "clean_statistics",
                    {"min": clean_images.min(), "max": clean_images.max()},
                    global_step,
                ),
                writer.add_scalars(
                    "with_fingerprint_statistics",
                    {
                        "min": fingerprinted_images.min(),
                        "max": fingerprinted_images.max(),
                    },
                    global_step,
                ),
                writer.add_scalars(
                    "residual_statistics",
                    {
                        "min": residual.min(),
                        "max": residual.max(),
                        "mean_abs": residual.abs().mean(),
                    },
                    global_step,
                ),
                print(
                    "residual_statistics: {}".format(
                        {
                            "min": residual.min(),
                            "max": residual.max(),
                            "mean_abs": residual.abs().mean(),
                        }
                    )
                )
                writer.add_image(
                    "clean_image", make_grid(clean_images, normalize=True), global_step
                )
                writer.add_image(
                    "residual",
                    make_grid(residual, normalize=True, scale_each=True),
                    global_step,
                )
                writer.add_image(
                    "image_with_fingerprint",
                    make_grid(fingerprinted_images, normalize=True),
                    global_step,
                )
                save_image(
                    fingerprinted_images,
                    SAVED_IMAGES + "/{}.png".format(global_step),
                    normalize=True,
                )

                writer.add_scalar(
                    "loss_weights/l2_loss_weight", l2_loss_weight, global_step
                )
                writer.add_scalar(
                    "loss_weights/BCE_loss_weight",
                    BCE_loss_weight,
                    global_step,
                )

            # checkpointing
            if global_step % 5000 == 0:
                torch.save(
                    decoder_encoder_optim.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"),
                )
                torch.save(
                    encoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                f = open(join(CHECKPOINTS_PATH, EXP_NAME + "_variables.txt"), "w")
                f.write(str(global_step))
                f.close()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print(all_rand_fin)


if __name__ == "__main__":
    main()
