import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np 
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt 

'''
mnist-m dataset
classes : 10
training set: 
test set: 9001
'''

class Mnist_m(Dataset):
    def __init__(self,train=True,transform=None,path=None,image_size=32):
        self.train=train
        if transform!=None:
            self.transfrom = transform
        else:
            self.transform=transforms.Compose([transforms.Resize(image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
                                                
        if path == None:
            self.path = './data/mnist_m'
        else:
            self.path = path 
        self.get_data()

    def get_data(self):
        if self.train:
            data_dir = os.path.join(self.path,"mnist_m_train")
            label_dir = os.path.join(self.path,"mnist_m_train_labels.txt")
        else:
            data_dir = os.path.join(self.path,"mnist_m_test")
            label_dir = os.path.join(self.path,"mnist_m_test_labels.txt")
        
        label_f = open(label_dir)
        records = np.array(label_f.readlines())
        label_f.close()
        image_list,label_list=[],[]
        for i in records:
            line = i.strip().split()
            img_path = os.path.join(data_dir,line[0])
            label = line[1]
            image_list.append(img_path)
            label_list.append(label)
        
        # image = []
        # for path in image_list:
        #     image.append(Image.open(path))
        self.image = image_list
        self.label = label_list
        print("images number:",len(self.image))
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        image = Image.open(self.image[idx])
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        label = np.array(label).astype(np.int64)
        return image,label 

def get_mnist_m(train,batch_size=32,transform=None,path=None,image_size=32):
    dataset = Mnist_m(train,transform,path,image_size)
    if batch_size == -1:
        batch_size = len(dataset)
    if train == True:
        shuffle = True
        loader = DataLoader(dataset, batch_size, shuffle, num_workers=8, drop_last= True)
    else:
        shuffle = False
        loader = DataLoader(dataset, batch_size, shuffle, num_workers=8)

    print("MNIST-M training:{} dataset:{} batch_num:{}".format(train,len(dataset),len(loader)))
    return loader 

def get_train_and_test(batch_size,transform=None,path=None,image_size=28):
    train_loader = get_mnist_m(True,batch_size,transform,path,image_size)
    test_loader = get_mnist_m(False,batch_size,transform,path,image_size)
    return train_loader,test_loader 
        
if __name__ == "__main__":
    loader = get_mnist_m(False,64,image_size=28)
    print("batch_num:",len(loader))
    batch = next(iter(loader))
    first = batch[0][0].numpy()
    print(batch[1].numpy())
    print("image shape:",first.shape)
    print("max pixel in image:",np.max(first))
    print("min pixel in image:",np.min(first))
    print("dtype of image:",first.dtype)
    
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title("mnist data")
    imgs = vutils.make_grid(batch[0][:64],padding=2,normalize=True).numpy()
    print(batch[1].numpy())
    plt.imshow(np.transpose(imgs,(1,2,0)))
    plt.show()
