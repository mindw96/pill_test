import io

import torchvision.transforms as transforms
from PIL import Image

from efficientnet_pytorch import EfficientNet
import torch


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


model_name = 'efficientnet-b7'

image_size = EfficientNet.get_image_size(model_name)
model = EfficientNet.from_pretrained(model_name, num_classes=2)

model.load_state_dict(torch.load('class2.pt'))
model.eval()

with open("경로 파일", 'rb') as f:
    image_bytes = f.read()
    inputs = transform_image(image_bytes=image_bytes)

outputs = model(inputs)
_, predict = torch.max(outputs, 1)

print(predict)
