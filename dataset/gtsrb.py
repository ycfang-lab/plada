import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np 
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import scipy.io as io 
import pandas as pd 

'''
GTSRB dataset
classes: 43
total: 39209
training set: 31367
testing set:7842
feature size: no fiexed images
'''
'''
create images according csv file.
new image file is ./data/gtsrb/images/
'''
def create_new_image_file(path='./data/gtsrb'):
    # new image file
    new_image_file = os.path.join(path,'images')
    # ppm image file
    root_image_file = os.path.join(path,"Final_Training","Images")
    # the list of image file name
    image_file_list = [item for item in os.listdir(root_image_file)]
    

    for files in image_file_list:
        path = os.path.join(new_image_file,files)
        print("create at path:%s"%(path))
        if not os.path.exists(path):
            os.makedirs(path)

        # the file of images with classes information
        data_dir = os.path.join(root_image_file,files)
        #print("data_dir:",data_dir)
        # the list of image path with the class "files"
        file_names = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(".ppm")]

        # get csv file path
        for f in os.listdir(data_dir):
            if f.endswith('.csv'):
                csv_dir = os.path.join(data_dir,f)
        
        #print("csv_dir:",csv_dir)
        # read csv file data
        csv_data = pd.read_csv(csv_dir)
        # change to numpy array
        csv_data_array = np.array(csv_data)

        for i in range(csv_data_array.shape[0]):
            csv_data_list = csv_data_array[i,:].tolist()[0].split(";")
            #print(csv_data_list)

            sample_dir = os.path.join(data_dir,csv_data_list[0])
            #print(sample_dir)

            img = Image.open(sample_dir)
            box = (int(csv_data_list[3]),int(csv_data_list[4]),int(csv_data_list[5]),int(csv_data_list[6]))
            crop_img = img.crop(box)
            #img.show()
            #crop_img.show()

            new_file = os.path.join(path,csv_data_list[0].split('.')[0]+".jpg")
            #print(new_file)
            crop_img.save(new_file,"JPEG")
            #assert False
        

def save_image_file(path='./data/gtsrb'):
        root_image_file = os.path.join(path,"images")
        image_file_list = [os.path.join(root_image_file,item) for item in os.listdir(root_image_file)]
        label = [int(item) for item in os.listdir(root_image_file)]
        

        image_list,label_list = [],[]

        for j in image_file_list:
            for img in os.listdir(j):
                if img.endswith('.jpg'):
                    image_list.append(os.path.join(j,img))
                    label_list.append(int(os.path.basename(j)))


        print(label_list)

        # random list
        idx = np.arange(len(image_list))
        seed = np.random.seed(100)
        np.random.shuffle(idx)
        image_list = np.array(image_list)[idx]
        label_list = np.array(label_list)[idx]
        print("total image num:",len(label_list))
        
        train_image = image_list[:31367]
        train_label = label_list[:31367]
      
        test_image = image_list[31367:]
        test_label = label_list[31367:]


        file1_dict,file2_dict = dict(),dict() 
        file1_dict['X'],file1_dict['y'] = train_image,train_label
        file2_dict['X'],file2_dict['y'] = test_image,test_label

        np.save(os.path.join(path,"train.npy"),file1_dict)
        np.save(os.path.join(path,"test.npy"),file2_dict)


class GTSRB(Dataset):
    def __init__(self,train=True,transform=None,path=None,image_size=40,gray=False):
        self.train = train 
        self.gray = gray 
        if transform !=None:
            self.transform = transform 
        else:
            if self.gray == True:
                self.transform = transforms.Compose([transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,),(0.5,))])
            else:
                self.transform = transforms.Compose([transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        if path == None:
            path = './data/gtsrb'
        self.path = path
        self.get_data()

    def get_data(self):
        if self.train:
            image_path_file = os.path.join(self.path,'train.npy')
        else:
            image_path_file = os.path.join(self.path,'test.npy')
        file_list = np.load(image_path_file).item()
        image_path_list = file_list['X']
        label_list = file_list['y']

        self.image = image_path_list
        self.label = label_list 

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        image = Image.open(self.image[idx])
        
        label = self.label[idx]

        if self.gray == True:
            image = image.convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = np.array(label).astype(np.int64)

        return image,label 
    

def get_gtsrb(train,batch_size=32,transform=None,path=None,image_size=40,gray=False):
    dataset = GTSRB(train,transform,path,image_size,gray=gray)
    if train == True:
        shuffle = False
    else:
        shuffle = False
    
    if batch_size == -1:
        batch_size = len(dataset)
    loader = DataLoader(dataset,batch_size,shuffle,num_workers=4)
    print("GTSRB datasettraining:{} dataset:{} batch_num:{}".format(train,len(dataset),len(loader)))
    return loader 

def get_train_and_test(batch_size,transform=None,path=None,image_size=40,gray=False):
    train_loader = get_gtsrb(True,batch_size,transform,path,image_size,gray)
    test_loader = get_gtsrb(False,batch_size,transform,path,image_size,gray)
    return train_loader,test_loader

if __name__ == "__main__":

    loader = get_gtsrb(True,64,gray=False)
    print("batch_num:",len(loader))
    
    for batch in loader:
        plt.figure(figsize=(8,8))
        first = batch[0][0].numpy()
        label = batch[1].numpy()
        print(label)
        print("image shape:",first.shape)
        print("max pixel in image:",np.max(first))
        print("min pixel in image:",np.min(first))
        print("dtype of image:",first.dtype)
        
        
        #plt.cla()
        plt.axis('off')
        plt.title("mnist data")
        imgs = vutils.make_grid(batch[0][:64],padding=2,normalize=True).numpy()
        plt.imshow(np.transpose(imgs,(1,2,0)))
        plt.show()
        
    # # plt.show()
    # # loader = get_gtsrb(False,64,gray=False)
    # # a = [0]*43
    # # for i,(item,label) in enumerate(loader):
    # #     for each in label.numpy():
    # #         a[each] +=1
    # # print(a)

        
