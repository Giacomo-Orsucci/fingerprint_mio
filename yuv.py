import cv2
import glob
import os
import PIL
import matplotlib.pyplot as plt

image_directory = '/media/giacomo/hdd_ubuntu/dataset_celeba/img_celeba'

# Carica tutte le immagini nel dataset (modifica il percorso secondo necessità)
image_paths = glob.glob('path_to_images/*.jpg')


j = 0
# Ciclo per convertire ogni immagine
for filename in os.listdir(image_directory):
    j += 1

    if j == 3:
        break;
    # Carica l'immagine in formato RGB
    img_path = os.path.join(image_directory, filename)
    img_gbr = cv2.imread(img_path,3)

    #img_rgb = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2RGB)
    #plt.imshow(img_rgb)
    #plt.title("Originale")
    #plt.show()

    # Converti l'immagine da RGB a YUV
    image_yuv = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2YUV)



    

    image_bgr = cv2.cvtColor(img_gbr, cv2.COLOR_YUV2BGR)

    image_rgb = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2RGB)
    


    plt.imshow(image_rgb)
    plt.title("prova")
    plt.show()

    # Salva l'immagine convertita (modifica il percorso di output secondo necessità)
    
    img_yuv_path = os.path.join("/media/giacomo/hdd_ubuntu/dataset_celeba/celeba_yuv") 
    os.makedirs(img_yuv_path, exist_ok=True)
    img_yuv_filename = os.path.join(img_yuv_path, filename)
    PIL.Image.fromarray(image_rgb, "RGB").save(img_yuv_filename)
   