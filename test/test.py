import requests
from PIL import Image
import numpy as np
import io
import base64

resp = requests.post('http://localhost:5000/predict', files={'file': open('t1.jpg', 'rb')})

print(resp.text)