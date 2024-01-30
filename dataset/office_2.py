import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt 
import numpy as np 
from dataset import pre_process
#import pre_process


label_map = {'back_pack': 0, 'bike': 1, 'bike_helmet': 2,
             'bookcase': 3, 'bottle': 4, 'calculator': 5, 
             'desktop_computer': 6, 'desk_chair': 7, 'desk_lamp': 8, 
             'file_cabinet': 9, 'headphones': 10, 'keyboard': 11,
             'laptop_computer': 12, 'letter_tray': 13, 'mobile_phone': 14, 
             'monitor': 15, 'mouse': 16, 'mug': 17, 
             'paper_notebook': 18, 'pen': 19, 'phone': 20, 
             'printer': 21, 'projector': 22, 'punchers': 23, 
             'ring_binder': 24, 'ruler': 25, 'scissors': 26,
             'speaker': 27, 'stapler': 28, 'tape_dispenser': 29,
             'trash_can': 30}
# first thing I should do 
def create_list(data_path,name):
    data_img_list = []
    data_labs_list = []
    # get the files under the path
    data_path = os.path.abspath(data_path)
    folder_list = os.listdir(data_path)
    print(folder_list)
    for i,p in enumerate(folder_list,0):
        
        label = label_map[p]
        cate_path = os.path.join(data_path,p)
        image_list = [os.path.join(cate_path,d) for d in os.listdir(cate_path)]
        data_img_list.extend(image_list)
        label_list = [label for ite in range(len(image_list))]
        data_labs_list.extend(label_list)
    data_list = [img+'\t'+str(lab)+"\n" for img,lab in zip(data_img_list,data_labs_list)]
    
    with open(name+'.txt','w') as f:
        f.writelines(data_list)
    

#create_list(r'../data/office/amazon','../data/office/amazon')
#create_list(r'E:/Mycode/CVPR2019/prototyoe_refine/data/office/webcam','../data/office/webcam')
#create_list(r'E:/Mycode/CVPR2019/prototyoe_refine/data/office/dslr','../data/office/dslr')

class BaseData(Dataset):
    def __init__(self, train, types):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

        transform_test = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize])
        #transform_train = pre_process.image_train()
        #transform_test = pre_process.image_test()

        if train:
            self.transform = transform_train
        else:
            self.transform = transform_test

        self._get_images(types)
    
    def _get_images(self, types):
        if types == 'amazon':
            path = './data/office/amazon.txt'
        elif types == 'webcam':
            path = './data/office/webcam.txt'
        else:
            path = './data/office/dslr.txt'

        with open(path, 'r') as f:
            lines = f.readlines()
        
        self.image_list = []
        self.label_list = []
        for line in lines:
            self.image_list.append(line.strip().split('\t')[0])
            self.label_list.append(int(line.strip().split('\t')[1]))
    
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])

        if self.transform:
            img = self.transform(img)

        lab = self.label_list[idx]
        lab = np.array(lab).astype(np.int64)

        return img, lab


class Amazon(BaseData):
    def __init__(self, train):
        super(Amazon, self).__init__(train, 'amazon')
        if train:
            print("load Amazon training data")
        else:
            print("load Amazon testing data")


class Webcam(BaseData):
    def __init__(self, train):
        super(Webcam, self).__init__(train, 'webcam')
        if train:
            print("load Webcam training data")
        else:
            print("load Webcam testing data")


class Dslr(BaseData):
    def __init__(self, train):
        super(Dslr, self).__init__(train, 'dslr')
        if train:
            print("load Dslr training data")
        else:
            print("load Dslr testing data")


def get_office(name, train, batch_size):
    if name == 'amazon':
        dataset = Amazon
    elif name == 'dslr':
        dataset = Dslr
    elif name == 'webcam':
        dataset = Webcam 
    data = dataset(train)
    if train:
        loader = DataLoader(data, batch_size, shuffle=True, num_workers=8, drop_last=True)
    else:
        loader = DataLoader(data, batch_size, shuffle=True, num_workers=8)
    return loader 


if __name__ == "__main__":
    create_list(r'./data/office/amazon/images','./data/office/amazon')
    create_list(r'./data/office/webcam/images','./data/office/webcam')
    create_list(r'./data/office/dslr/images','./data/office/dslr')
    loader = get_office('amazon',True,64)
    print(type(loader))
    print("batch_num:",len(loader))

    batch = next(iter(loader))
    print(batch[0][0].requires_grad)
    assert False
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
    plt.title("OFFICE data")
    imgs = vutils.make_grid(batch[0][:64],padding=2,normalize=True).numpy()
    plt.imshow(np.transpose(imgs,(1,2,0)))
    plt.show()


            

