import torchvision.transforms as transforms
from .ClevrData import *

def get_clevr_data(config):
    data_path = "./data/CLEVR/"
    imsize = 192
    training_data = ClevrDataset(data_dir=data_path,
                                        imsize=imsize,
                                        split="train",
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
                               )

    validation_data = ClevrDataset(data_dir=data_path,
                                        imsize=imsize,
                                        split="test",
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

                                  )

    return training_data, validation_data
