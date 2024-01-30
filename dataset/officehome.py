from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

label_map = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 
                      'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 
                      'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 
                      'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 
                      'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 
                      'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29,
                      'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 
                      'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 
                      'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44,
                      'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49,
                      'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54,
                     'Soda': 55, 'Speaker': 56, 'Spoon': 57,  'Table': 58, 'Telephone': 59,
                     'ToothBrush': 60, 'Toys': 61, 'Trash_Can': 62, 'TV': 63, 'Webcam': 64}

def image_path_list(path): #..../Art
    class_list = [x for x in os.listdir(path)]
    path_list = []
    label_list = []
    for i in class_list:
        label = label_map[i]
        # print(i,label)
        sub_path = os.path.join(path,i) #..../Art/Alarm_Clock
        img_list = [x for x in os.listdir(sub_path)]
        img_list.sort()
        for j in img_list:
            img_path = os.path.join(sub_path, j)
            path_list.append(img_path)
            label_list.append(label)
    return path_list, label_list

class officehome(Dataset):
    def __init__(self, domain, train, path=None):
        super(officehome, self).__init__()

        if domain == 'a':
            self.path = os.path.join(path, 'Art')
        elif domain == 'c':
            self.path = os.path.join(path, 'Clipart')
        elif domain == 'p':
            self.path = os.path.join(path, 'Product')
        elif domain == 'r':
            self.path = os.path.join(path, 'Real World')
        self.image_list, self.label_list = image_path_list(self.path)
        
        #参数更改？？？
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize])

        self.transform_test = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  normalize])
        self.train=train

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx]).convert("RGB")
        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        lab = np.array(self.label_list[idx]).astype(np.int64)
        return img, lab

def get_officehome(train, domain, path, batch_size):
    dataset = officehome(domain, train, path)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=train)
    return dataloader

if __name__ == "__main__":
    '''
    path = '/media/antec/sda/zw/data/officehome/OfficeHomeDataset'
    domain = 'a'
    data = get_officehome(True,domain,path,32)
    '''
    path_list, label_list = image_path_list('/media/antec/sda/zw/data/officehome/OfficeHomeDataset/Clipart')
    print(len(path_list))
    print(len(label_list))

