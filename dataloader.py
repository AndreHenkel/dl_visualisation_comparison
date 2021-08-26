#loads images from a predefined folder and puts it into a torch batch


import torch
import utils

import torchvision as tv
from torchvision import transforms

import urllib.request
import tarfile
import os, random

class DataLoader:
    def __init__(self, path, data_set_url, batch_size=10):
        self.path = path

        self.data_set_url = data_set_url

        self.download_and_unpack_dataset(self.data_set_url,path)

        self.resize = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Using Dataset: ", path+"/imagenette2-320/val")
        self.dataset = tv.datasets.ImageFolder(root=path+"/imagenette2-320/val",transform=self.resize)

        #NOTE!
        #expecting the folder structure to be and ordered as in imagenette2-320/val here: https://github.com/fastai/imagenette
        #ordering expeccting: tench, English Springer, cassette player, chain saw, church, french horn, garbage truck, gas pump, golf ball, parachute
        self.idx_map = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
        #self.idx_map = [217,566,482,491,497,569,571,574,701,0]
        self.data_loader = torch.utils.data.DataLoader(self.dataset,batch_size=batch_size, shuffle=True)
        # print("classes:")
        # print(self.dataset.class_to_idx)

    def _select_random_image(self, path):
        im_path = path+"/"+random.choice(os.listdir(path))
        if os.path.isdir(im_path) == True:
            im_path = self._select_random_image(im_path) #recursive
        return im_path

    def download_and_unpack_dataset(self,url, dest):
        if not os.path.isdir(self.path+"/imagenette2-320"):
            print("No Dataset was detected. Downloading and extracting: " + self.data_set_url)
            print("Into destination: ",dest)
            print("...")
            tar_file = url
            ftpstream = urllib.request.urlopen(tar_file)
            tar_file = tarfile.open(fileobj=ftpstream, mode="r|gz")
            tar_file.extractall(path=dest)
            tar_file.close()
        else:
            print(self.path+"/imagenette2-320"+" was detected. No new download required!")
        print("Dataset is ready for use!")


    def get_image_batch(self):
        """
        also already puts the correct label on it, with the idx_map
        """
        batch,idx = next(iter(self.data_loader))
        label = []
        for i in idx:
            label.append(self.idx_map[i])
        return batch, label

    def get_data_loader(self):
        return self.data_loader

        # """
        # also resizes it already in advance for AlexNet
        # """
        #
        # image_batch = torch.Tensor((batch_size,3,224,224))
        #
        # for i in range(batch_size):
        #     im_path = self._select_random_image(self.path)
        #     image = utils.load_image(im_path) #Tensor (3,224,224)
        #     image_batch[i] = image
        #
        # return image_batch
