import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np 
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt 

'''
mnist dataset
classes : 10
training set: 60000
testing set: 10000
feature size: 28x28 
'''
class Mnist(Dataset):
    def __init__(self,train=True,transform=None,use_all=True,path=None,image_size=28,gray=True,aug=False):
        self.train = train
        self.use_all = use_all
        self.gray = gray
        self.aug = aug
        if transform!=None:
            self.transform=transform
        else:
            if self.gray == False:
                if self.aug:
                    self.transform = transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.RandomApply([transforms.Normalize((1,1,1),(-1,-1,-1))],p=0.5), 
                                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) 
                else:
                    self.transform = transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            else:
                if self.aug:
                    self.transform = transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.RandomApply([transforms.Normalize((1,),(-1,))],p=0.5), 
                                                    transforms.Normalize((0.5,),(0.5,)),
                                                    ])
                else:
                    self.transform = transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,),(0.5,)),
                                                    ])
                    
        

        if path == None:
            path = './data/mnist/'
        self.path = path
        self.get_data() 

    def get_data(self):
        if self.train:
            self.data = np.load(os.path.join(self.path,'train.npy')).item()
        else:
            self.data = np.load(os.path.join(self.path,'test.npy')).item()
        self.x = self.data['image']
        self.y = self.data['label']

        if (not self.use_all) and self.train:# if experiment use usps,2000 mnist data should be used   
            # random choose methods
            self.x = self.x[:2000]
            self.y = self.y[:2000]
        # print(self.x.shape)
        # print(self.y.shape)

    def __len__(self,):
        return len(self.x)

    def __getitem__(self,idx):
        image = self.x[idx]
        image = Image.fromarray(image)
        if self.gray == False:
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)

        label = self.y[idx].astype(np.int64)
        
        return image,label 

def get_mnist(train,batch_size=32,use_all=True,transform=None,path=None,image_size=28,gray=True,aug=False):
    dataset = Mnist(train,transform,use_all,path=path,image_size=image_size,gray=gray,aug=aug)
    if batch_size==-1:
        batch_size = len(dataset)

    if train==True:
        shuffle=True
        loader = DataLoader(dataset, batch_size, shuffle, num_workers=4, drop_last=True)
    else:
        shuffle=False
        loader = DataLoader(dataset, batch_size, shuffle, num_workers=4)

    print("MNIST training:{} dataset:{} batch_num:{}".format(train,len(dataset),len(loader)))
    return loader 

def get_train_and_test(batch_size,use_all=True,transform=None,path=None,image_size=28,gray=True,aug=False):
    train_loader = get_mnist(True,batch_size,use_all,transform,path,image_size,gray,aug)
    test_loader = get_mnist(False,batch_size,use_all,transform,path,image_size,gray,aug=False)
    return train_loader, test_loader 

def get_images():
    dataset = Mnist(True,None,True,path=None,image_size=28,gray=True)    
    for i in range(10):
        for j in range(len(dataset.x)):
            if dataset.y[j] == i:
                img = dataset.x[j]
            
                #img = np.transpose(dataset.x[j],(1,2,0))
                print(img.shape)

                img = Image.fromarray(img,mode='L')
                img.save("./result/"+str(i)+".jpg")
                break
    print("done!")

if __name__ == "__main__":
    # loader = get_mnist(True,64,True,gray=False,aug=True)
    # print("batch_num:",len(loader))
    # batch = next(iter(loader))
    # first = batch[0][0].numpy()
    # print(first)
    # print("image shape:",first.shape)
    # print("max pixel in image:",np.max(first))
    # print("min pixel in image:",np.min(first))
    # print("dtype of image:",first.dtype)
    
    # plt.figure(figsize=(8,8))
    # plt.axis('off')
    # plt.title("mnist data")
    # imgs = vutils.make_grid(batch[0][:64],padding=2,normalize=True).numpy()
    # plt.imshow(np.transpose(imgs,(1,2,0)))
    # plt.show()
    get_images()



