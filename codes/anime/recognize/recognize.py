import os
from django.utils import timezone
import json
from torchvision import transforms, datasets
import torchvision
import torch
import numpy as np
import torchvision.models as models
import cv2

model_path = "static/models/"
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def get_class_indice():
    json_file = open(model_path + "class_indices.json", 'r')
    class_indice = json.load(json_file)
    return class_indice

class_indice = get_class_indice()

def load_model():
    model = models.resnet18(num_classes=len(class_indice))
    model_weight_path = model_path + "resnet18.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    return model

model = load_model()

def save_upload_file(f):
    date = timezone.now()
    save_path = 'static/upload/%s-%s.jpg' % (f.name ,date.strftime("%Y-%m-%d-%H-%M-%S"))
    with open(save_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    
    res = predict(model, save_path)

    return save_path.split("/")[-1], res


def read_img(img_path):
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])

def img_preprocess(img):
    img_pil = torchvision.transforms.ToPILImage()(img)
    data_transform = transforms.Compose([transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img2 = data_transform(img_pil)
    img_exp = np.expand_dims(img2, axis=0)
    img_tensor = torch.tensor(img_exp, dtype=torch.float32)
    return img_tensor

def predict(model, img_path):
    img_raw = read_img(img_path)
    img_pre = img_preprocess(img_raw)
    predict_prob = model(img_pre)
    predict_clf = torch.argmax(predict_prob).numpy()
    return class_indice[str(predict_clf)]