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
synthsign dataset
classes: 43
training set: 100000 - test_num
testing set: test_num
feature size: 40x40
'''
class Synthsign(Dataset):
    def __init__(self,train=True,transform=None,path=None,image_size=40,gray=False,test_num=2000):
        self.train = train
        self.gray = gray
        self.test_num = test_num
        if transform!=None:
            self.transform=transform
        else:
            if self.gray == False:
                self.transform = transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) 
            else:
                self.transform = transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,),(0.5,))])
        if path == None:
            path = './data/synthsign/'
        self.path = path
        self.get_data() 

    def get_data(self):
        image_file = self.path
        label_path = os.path.join(self.path,'train_labelling.txt')

        label_f = open(label_path)
        records = np.array(label_f.readlines())
        label_f.close()
        image_list,label_list = [],[]
        for i in records:
            line = i.strip().split()
            image_path = os.path.join(image_file,line[0])

            label = line[1]

            image_list.append(image_path)
            label_list.append(int(label))
        
        image_num = len(label_list)
        # print(max(label_list))
        # print(min(label_list))
        # #print(label_list)

        if self.train==True:
            self.image = image_list[:image_num-self.test_num]
            self.label = label_list[:image_num-self.test_num]
        else:
            self.image = image_list[image_num-self.test_num:]
            self.label = label_list[image_num-self.test_num:]
        print('images number',len(self.image))
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        image = Image.open(self.image[idx])

        #print(image.size,image.format,image.mode)
        label = self.label[idx]

        if self.gray==True:
            image = image.convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = np.array(label).astype(np.int64)
        return image,label 

def get_synthsign(train,batch_size=32,transform=None,path=None,image_size=40,gray=False):
    dataset = Synthsign(train,transform,path,image_size=image_size,gray=gray)
    if train==True:
        shuffle = False
    else:
        shuffle = False 

    if batch_size == -1:
        batch_size = len(dataset)
    loader = DataLoader(dataset,batch_size,shuffle,num_workers=4)
    print("Synthsign dataset training:{} dataset:{} batch_num:{}".format(train,len(dataset),len(loader)))
    return loader 

def get_train_and_test(batch_size,transform=None,path=None,image_size=40,gray=False):
    train_loader = get_synthsign(True,batch_size,transform,path,image_size,gray)
    test_loader = get_synthsign(False,batch_size,transform,path,image_size,gray)
    return train_loader,test_loader


if __name__ == "__main__":
    loader = get_synthsign(False,64,gray=False)
    print("batch_num:",len(loader))
    plt.figure(figsize=(8,8))
    batch = next(iter(loader))
    first = batch[0][0].numpy()
    print("label:",batch[1])
    print("image shape:",first.shape)
    print("max pixel in image:",np.max(first))
    print("min pixel in image:",np.min(first))
    print("dtype of image:",first.dtype)
    

    plt.axis('off')
    plt.title("mnist data")
    imgs = vutils.make_grid(batch[0][:64],padding=2,normalize=True).numpy()
    plt.imshow(np.transpose(imgs,(1,2,0)))

    plt.show()
    # loader = get_synthsign(False,64,gray=False)
    # a = [0]*43
    # for i,(item,label) in enumerate(loader):
    #     for each in label.numpy():
    #         a[each] +=1
    # print(a)