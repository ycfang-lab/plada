import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np 
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import scipy.io as io 

'''
The Street view house numbers (SVHN) Dataset
classes : 10
training set: 73257
testing set: 26032
additional set: 531131
feature size: 32x32
'''
class SVHN(Dataset):
    def __init__(self,train=True,transform=None,path=None,image_size=32,gray=False):
        self.train = train 
        self.gray = gray 
        if transform !=None:
            self.transform = transform 
        else:
            if self.gray == True:
                self.transform = transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,),(0.5,))])
            else:
                self.transform = transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        if path == None:
            path = './data/svhn'
        self.path = path
        self.get_data()
    
    def get_data(self):
        if self.train:
            self.data = io.loadmat(os.path.join(self.path,'train_32x32.mat'))
        else:
            self.data = io.loadmat(os.path.join(self.path,'test_32x32.mat'))
        self.data['X'] = np.transpose(self.data['X'],[3,0,1,2])
        self.data['y'] = np.reshape(self.data['y'],(self.data['y'].shape[0]))
        print('data:',self.data['X'].shape)
        print('label:',self.data['y'].shape)
    
    def __len__(self):
        return len(self.data['X'])

    def __getitem__(self,idx):
        image = self.data['X'][idx]
        image = Image.fromarray(image,mode='RGB')
        if self.gray == True:
            image = image.convert('L')

        if self.transform:
            image = self.transform(image)
        label = self.data['y'][idx]
        if label == 10:
            label = 0
        label = np.array(label).astype(np.int64)
        return image,label

def get_svhn(train,batch_size=32,transform=None,path=None,image_size=32,gray=False):
    dataset = SVHN(train,transform,path=path,image_size=image_size,gray=gray)
    if train == True:
        shuffle = True
    else:
        shuffle = False
    if batch_size == -1:
        batch_size = len(dataset)
    loader = DataLoader(dataset,batch_size,shuffle,num_workers=4)
    print("SVHN training:{} dataset:{} batch_num:{}".format(train,len(dataset),len(loader)))
    return loader

def get_train_and_test(batch_size,transform=None,path=None,image_size=28,gray=False):
    train_loader = get_svhn(True,batch_size,transform,path,image_size,gray)
    test_loader = get_svhn(False,batch_size,transform,path,image_size,gray)
    return train_loader,test_loader 

def get_images():
    dataset = SVHN(True,None,path=None,image_size=28,gray=False)    
    for i in range(1,11):
        for j in range(len(dataset.data['X'])):
            if dataset.data['y'][j] == i:
                img = dataset.data['X'][j]
            
                #img = np.transpose(dataset.x[j],(1,2,0))
                print(img.shape)

                img = Image.fromarray(img,mode='RGB')
                img = img.resize((32,32))
                img.save("./data/svhn/"+str(i)+".jpg")
                break
    print("done!")

if __name__ == "__main__":
    # loader = get_svhn(True,64,gray=False)
    # print("batch_num:",len(loader))
    # batch = next(iter(loader))
    # first = batch[0][0].numpy()
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

    

