import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt 
from torchvision.datasets import ImageFolder

'''
The Office dataset:
Contains 3 domains Amazon, webcam amd Dslr. Each Contain images from amazon.com,or
office environment images taken with varying lighting and pose changes using a webcam,
or a dslr camera, respectively. Contains 31 categories in each domain.
'''

class Amazon(Dataset):
    def __init__(self,train): 
        self.path = "./data/office/amazon"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = [transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                            normalize]
        transform_eval = [transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          normalize]

        if train:
            self.transform = transform_train
        else:
            self.transform = transform_eval

        self.dataset = ImageFolder(self.path,transform=transforms.Compose(self.transform))
        #print(self.dataset.class_to_idx) 
    

class Dslr(Dataset):
    def __init__(self,train):
        self.path = "./data/office/dslr"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = [transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                            normalize]
        transform_eval = [transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          normalize]
        if train:
            self.transform = transform_train
        else:
            self.transform = transform_eval
        self.dataset = ImageFolder(self.path,transform=transforms.Compose(self.transform))
        #print(self.dataset.class_to_idx) 

class Webcam(Dataset):
    def __init__(self,train):    
        self.path = "./data/office/webcam"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = [transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                            normalize]
        transform_eval = [transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          normalize]
        if train:
            self.transform = transform_train
        else:
            self.transform = transform_eval

        self.dataset = ImageFolder(self.path,transform=transforms.Compose(self.transform))
        #print(self.dataset.class_to_idx) 

def get_office(name,train,batch_size=32):
    if name == 'amazon':
        dataset = Amazon
    elif name == 'dslr':
        dataset = Dslr
    elif name == 'webcam':
        dataset = Webcam
    dataset = dataset(train)
    # print(dataset.dataset.class_to_idx) 
    # print(dataset.dataset.classes)
    loader = DataLoader(dataset.dataset,batch_size,shuffle=train,num_workers=2)
    return loader 

if __name__ == "__main__":
    import numpy as np
    loader = get_office('amazon',True,64)
    print(type(loader))
    print("batch_num:",len(loader))

    batch = next(iter(loader))
    print(len(batch))
    first = batch[0][0].numpy()
    label = batch[1][0].numpy()
    print("image shape:",first.shape)
    print("label shape and dtype:",label.shape,label.dtype)
    print("max pixel in image:",np.max(first))
    print("min pixel in image:",np.min(first))
    print("dtype of image:",first.dtype)
    
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title("mnist data")
    imgs = vutils.make_grid(batch[0][:64],padding=2,normalize=True).numpy()
    plt.imshow(np.transpose(imgs,(1,2,0)))
    plt.show()



        

