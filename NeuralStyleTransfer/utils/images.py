import matplotlib.pyplot as plt

from unloaders import pil_unloader


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = pil_unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()
