'''
dep_site = ".../predict"

import requests

resp = requests.post(dep_site, files={"file": open('t1.jpg','rb')})

print(resp.text)'''

# convert base64 string to image, JSON format: {'preprocessed_image': image}
dep_site = ".../pre_image"

import requests

resp = requests.post(dep_site, files={"file": open('t1.jpg','rb')})

import base64
import io
from PIL import Image

img = base64.b64decode(resp.json()['preprocessed_image'])
img = Image.open(io.BytesIO(img))

img.show()