import os

from torch.utils import data
from PIL import Image


class ImagesFromList(data.Dataset):
    """
    A generic data loader that loads images from a list (Based on ImageFolder from pytorch)
    """

    def __init__(self, image_paths, root='', imsize=None, transform=None):

        image_list = [os.path.join(root, image_paths[i]) for i in range(len(image_paths))]

        if len(image_list) == 0:
            raise RuntimeError("Dataset contains 0 images!")

        self.root = root
        self.imsize = imsize
        self.image_list = image_list
        self.transform = transform

    def __getitem__(self, index):
        path = self.image_list[index]
        img = self.image_loader(path)

        if self.imsize is not None:
            img = self.image_resize(img, self.imsize)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def image_loader(path):
        with open(path, 'rb') as handle:
            img = Image.open(handle)
            return img.convert('RGB')

    @staticmethod
    def image_resize(img, imsize):
        img.thumbnail((imsize, imsize), Image.ANTIALIAS)
        return img
