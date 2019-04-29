from PIL import Image
import torch
import torchvision.transforms as transforms

from globals import device, img_size


loader = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
