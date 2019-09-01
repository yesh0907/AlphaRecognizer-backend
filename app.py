try:
  import unzip_requirements
except ImportError:
  pass

from flask import Flask, request, jsonify
from flask_cors import CORS

from PIL import Image
from io import BytesIO
import numpy as np
import os
import base64
import tarfile
import requests

import torch
import torch.nn.functional as F
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)


def load_model():
    # classes for classificaiton
    classes = []
    if not os.path.isfile('./model.tar.gz'):
        url = os.environ["MODEL_URL"].replace("\\", "")
        r = requests.get(url)
        bytestream = BytesIO(r.content)
        tar = tarfile.open(fileobj=bytestream, mode="r:gz")
        for member in tar.getmembers():
            if member.name.endswith(".txt"):
                print("Classes file is :", member.name)
                f=tar.extractfile(member)
                classes = f.read().splitlines()
            if member.name.endswith(".pth"):
                print("Model file is :", member.name)
                f=tar.extractfile(member)
                print("Loading PyTorch model")
                model = torch.jit.load(BytesIO(f.read()), map_location=torch.device('cpu')).eval()
    else:
        print("Loading PyTorch model")
        model = torch.jit.load('./model_jit.pth', map_location=torch.device('cpu')).eval()
        print("Loading classes")
        with open('./classes.txt', 'r') as f:
            classes = f.read().splitlines()
    return model, classes

model, classes = load_model()

def crop_image(img):
    white_pix = []

    width = img.width
    height = img.height

    x_offset = 25
    y_offset = 25

    for x in range(width):
        for y in range(height):
            curr_pix = img.getpixel((x,y))
            if curr_pix == (255, 255, 255):
                white_pix.append((x,y))

    white_pix = np.array(white_pix)
    
    max_white_pix = np.amax(white_pix, axis=0)
    min_white_pix = np.amin(white_pix, axis=0)

    top_white_x = min_white_pix[0]
    bottom_white_x = max_white_pix[0]
    top_white_y = min_white_pix[1]
    bottom_white_y = max_white_pix[1]

    left = top_white_x - x_offset if top_white_x - x_offset >= 0 else top_white_x
    top = top_white_y - y_offset if top_white_y - y_offset >= 0 else top_white_y
    right = bottom_white_x + x_offset if bottom_white_x + x_offset <= width else bottom_white_x
    bottom = bottom_white_y + y_offset if bottom_white_y + y_offset <= height else bottom_white_y

    return img.crop((left, top, right, bottom))

def convert_to_RGB(rgba_img):
    rgba_img.load() # required for rbga_img.split()
    rgb_img = Image.new("RGB", rgba_img.size, (255, 255, 255))
    rgb_img.paste(rgba_img, mask=rgba_img.split()[3])
    return rgb_img


def preprocess(img):
    img = convert_to_RGB(img)
    img = crop_image(img)
    torch_transforms = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = torch_transforms(img)
    return img_tensor.unsqueeze(0)

def predict(input_object, model):
    predict_values = model(input_object)
    preds = F.softmax(predict_values, dim=1)
    conf_score, indx = torch.max(preds, dim=1)
    predict_class = classes[indx].decode('utf-8')
    print('Predicted class is {}'.format(predict_class))
    print('Softmax confidence score is {}'.format(conf_score.item()))
    response = {}
    response['class'] = str(predict_class)
    response['confidence'] = conf_score.item()
    return response

def b64_to_img(imgb64):
    imgb64 = imgb64.split(';')[1]
    imgb64 = imgb64.split(',')[1]
    img_data = base64.decodebytes(imgb64.encode('utf-8'))
    img = Image.open(BytesIO(img_data))
    return img

@app.route('/')
def index():
    return jsonify({"running": "ok"})

@app.route('/predict', methods=['POST'])
def eval_image():
    print('got img')
    b64img_data = request.get_json()['image']
    img = b64_to_img(b64img_data)
    img_tensor = preprocess(img)
    res = predict(img_tensor, model)
    print(res)

    return res