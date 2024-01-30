import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np 
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import h5py

'''
usps dataset:
classes: 10
training set: 7291
testing set: 2007
feature size: 256 -> 16x16 (images)
'''

class Usps(Dataset):
    def __init__(self,train=True,transform=None,use_all=False,path=None,image_size=28):
        self.train=train
        if transform !=None:
            self.transform=transform
        else:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,),(0.5,))])
        self.use_all = use_all 
        if path==None:
            path = "./data/usps/usps.h5"
        self.path = path 
        self.get_data()

    def get_data(self):
        with h5py.File(self.path,'r') as hf:
            if self.train:
                data = hf.get('train')
                self.x = data.get('data')[:] # (7291,256)
                self.y = data.get('target')[:]# (7291,)
            else:
                data = hf.get('test')
                self.x = data.get('data')[:] # (2007,256)
                self.y = data.get('target')[:] # (2007,)
        if self.use_all == False and self.train==True:
            idx = np.random.choice(np.arange(len(self.y)),1800)
            self.x = self.x[idx]
            self.y = self.y[idx] 
        self.x = np.reshape(self.x,(self.x.shape[0],16,16))
        #print(np.max(self.x),np.min(self.x))
        self.x = self.x*255


    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        image = self.x[idx]
        image = Image.fromarray(np.uint8(image),mode='L')
        if self.transform:
            image = self.transform(image)
        label = self.y[idx].astype(np.int64)
        return image,label 

def get_usps(train,batch_size=32,use_all=True,transform=None,path=None,image_size=28):
    dataset = Usps(train,transform,use_all, path, image_size=image_size)
    if train==True:
        shuffle=True
    else:
        shuffle=False
    if batch_size == -1:
        batch_size=len(dataset)
    loader = DataLoader(dataset,batch_size,shuffle,num_workers=4)
    print("USPS training:{} dataset:{} batch_num:{}".format(train,len(dataset),len(loader)))
    return loader 

def get_train_and_test(batch_size,use_all=True,transform=None,path=None,image_size=None):
    train_loader = get_usps(True, batch_size, use_all, transform, path, image_size)
    test_loader = get_usps(False, batch_size, use_all, transform, path, image_size)
    return train_loader,test_loader 
    

def get_images():
    dataset = Usps(True,None,True,path=None,image_size=28)    
    for i in range(10):
        for j in range(len(dataset.x)):
            if dataset.y[j] == i:
                img = dataset.x[j]
            
                #img = np.transpose(dataset.x[j],(1,2,0))
                print(img.shape)

                img = Image.fromarray(np.uint8(img),mode='L')
                img = img.resize((32,32))
                img.save("./data/usps/"+str(i)+".jpg")
                break
    print("done!")

if __name__ == "__main__":
    loader = get_usps(True,32,True)
    batch = next(iter(loader))
    first = batch[0][0].numpy()
    print("image shape:",first.shape)
    print("max pixel in image:",np.max(first))
    print("min pixel in image:",np.min(first))
    print("dtype of image:",first.dtype)
    
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title("usps data")
    imgs = vutils.make_grid(batch[0][:64],padding=2,normalize=True).numpy()
    plt.imshow(np.transpose(imgs,(1,2,0)))
    plt.show()
