import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

from PIL import Image
import io
import cv2

import numpy as np

# --------------------------------------------------------------------------------
# CLASS LABELS

class_labels = ['Alpinia Galanga (Rasna)',
 'Amaranthus Viridis (Arive-Dantu)',
 'Artocarpus Heterophyllus (Jackfruit)',
 'Azadirachta Indica (Neem)',
 'Basella Alba (Basale)',
 'Brassica Juncea (Indian Mustard)',
 'Carissa Carandas (Karanda)',
 'Citrus Limon (Lemon)',
 'Ficus Auriculata (Roxburgh fig)',
 'Ficus Religiosa (Peepal Tree)',
 'Hibiscus Rosa-sinensis',
 'Jasminum (Jasmine)',
 'Mangifera Indica (Mango)',
 'Mentha (Mint)',
 'Moringa Oleifera (Drumstick)',
 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
 'Murraya Koenigii (Curry)',
 'Nerium Oleander (Oleander)',
 'Nyctanthes Arbor-tristis (Parijata)',
 'Ocimum Tenuiflorum (Tulsi)',
 'Piper Betle (Betel)',
 'Plectranthus Amboinicus (Mexican Mint)',
 'Pongamia Pinnata (Indian Beech)',
 'Psidium Guajava (Guava)',
 'Punica Granatum (Pomegranate)',
 'Santalum Album (Sandalwood)',
 'Syzygium Cumini (Jamun)',
 'Syzygium Jambos (Rose Apple)',
 'Tabernaemontana Divaricata (Crape Jasmine)',
 'Trigonella Foenum-graecum (Fenugreek)']

# --------------------------------------------------------------------------------
# LOAD MODEL

# Load the model on CPU
model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
device = torch.device('cpu')
model.to(device)

# Modify the last linear layer
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 30)

# Load the model state dict on CPU
model.load_state_dict(torch.load('app/EfficientNetV2.pth', map_location=device))

# Set model to evaluation mode
model.eval()

# --------------------------------------------------------------------------------
# PREPROCESS IMAGE

def preprocess_image(image_bytes):
    # 1 load image
    image = Image.open(io.BytesIO(image_bytes))
    
    image = np.array(image)

    # 2 convert image to hsv color space
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 3 define green color range
    lower_green = np.array([35, 40, 50])
    upper_green = np.array([85, 255, 255])
    
    # 4 create mask
    mask = cv2.inRange(img, lower_green, upper_green)
    
    # 5 dilation
    mask_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_c = cv2.dilate(mask, mask_connect, iterations=2)
    
    # 6 find contours
    contours, _ = cv2.findContours(img_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:1] # Extract the largest contours
    
    # 7 draw contours
    mask_con = np.zeros_like(image)
    cv2.drawContours(mask_con, contours, -1, (255, 255, 255), -1)
    
    # 8 masking
    masked_image = cv2.bitwise_and(image, mask_con)
    
    # 9 white background
    white_background = np.full_like(image, (255, 255, 255))
    img_white = np.where(mask_con == 255, image, white_background) 
    
    # 10 image PIL
    image = Image.fromarray((img_white).astype(np.uint8))
    
    # 11 transform function
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)


# --------------------------------------------------------------------------------
# PREDICT

def prediction(img_tensor):
    with torch.no_grad():
        pred = model(img_tensor)
    
    # top 5 predictions and their probabilities
    preds = torch.nn.functional.softmax(pred, dim=1).squeeze().tolist()
    preds = list(zip(class_labels, preds))
    
    # top 5 probabilities
    top5 = sorted(preds, key=lambda x: x[1], reverse=True)[:5]
    
    # top 5 labels
    top5_labels = [x[0] for x in top5]
    
    # top 5 probabilities
    top5 = [x[1] for x in top5]
    
    return top5_labels, top5

# --------------------------------------------------------------------------------
# Preprocess demo image

def pre_image_conv (image_bytes):
    # 1 load image
    image = Image.open(io.BytesIO(image_bytes))
    
    image = np.array(image)

    # 2 convert image to hsv color space
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 3 define green color range
    lower_green = np.array([35, 40, 50])
    upper_green = np.array([85, 255, 255])
    
    # 4 create mask
    mask = cv2.inRange(img, lower_green, upper_green)
    
    # 5 dilation
    mask_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_c = cv2.dilate(mask, mask_connect, iterations=2)
    
    # 6 find contours
    contours, _ = cv2.findContours(img_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:1] # Extract the largest contours
    
    # 7 draw contours
    mask_con = np.zeros_like(image)
    cv2.drawContours(mask_con, contours, -1, (255, 255, 255), -1)
    
    # 8 masking
    masked_image = cv2.bitwise_and(image, mask_con)
    
    # 9 white background
    white_background = np.full_like(image, (255, 255, 255))
    img_white = np.where(mask_con == 255, image, white_background) 
    
    # 10 image PIL
    image = Image.fromarray((img_white).astype(np.uint8))
    
    # resize
    image = image.resize((256, 256))
    
    # convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    
    return img_bytes