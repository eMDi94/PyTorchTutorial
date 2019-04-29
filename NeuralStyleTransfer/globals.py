import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_size = (512, 512) if torch.cuda.is_available() else (128, 128)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
