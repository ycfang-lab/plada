import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np 
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def split_image(path='./data/visda',train=True):
    if train == True:
        data_path = os.path.join(path, "train")
    else:
        data_path = os.path.join(path, "validation")
    file_path = os.path.join(data_path, "image_list.txt")
    
    image_list, label_list = [], []
    files = open(file_path)
    record = files.readlines()
    files.close()

    for ind,i in enumerate(record, 0):
        line = i.strip().split()
        image_list.append(os.path.join(data_path,line[0]))
        label_list.append(int(line[1]))

    # shuffle
    idx = np.arange(len(image_list))
    seed = np.random.seed(100)
    np.random.shuffle(idx)
    image_list = np.array(image_list)[idx]
    label_list = np.array(label_list)[idx]
    print("total image num:",len(label_list))

    # split
    if train:
        train_image_list = image_list[:106679]
        train_label_list = label_list[:106679]
        test_image_list = image_list[106679:152397]
        test_label_list = label_list[106679:152397] 
    else:
        train_image_list = image_list[:38772]
        train_label_list = label_list[:38772]
        test_image_list = image_list[38772:55388]
        test_label_list = label_list[38772:55388]
    
    train_file = {"X":train_image_list,"y":train_label_list}
    test_file = {"X":test_image_list,"y":test_label_list}

    # file name set
    if train:
        train_file_name = "synthtic_train_list.npy"
        test_file_name = "synthtic_test_list.npy"
    else:
        train_file_name = "real_train_list.npy"
        test_file_name = "real_test_list.npy"
    
    np.save(os.path.join(path,train_file_name),train_file)
    np.save(os.path.join(path,test_file_name),test_file)

    print("done!")


class VisDA(Dataset):
    def __init__(self, train=True, types="train", path=None):
        self.train = train
        self.types = types
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        self.transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize])
        
        self.transform_eval = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  normalize])

        if path is None:
            self.path = './data/visda/'
        else:
            self.path = path

        print(self.path)
        self.X = None
        self.y = None
        self.get_data()

    def get_data(self):
        if self.types == "train":
            if self.train:
                '''synth data train set'''
                file_path = os.path.join(self.path, "synthtic_train_list.npy")
            else:
                '''synth data test set'''
                file_path = os.path.join(self.path, "synthtic_test_list.npy")
        elif self.types == 'eval':
            if self.train:
                '''real data train set'''
                file_path = os.path.join(self.path, "real_train_list.npy")
            else:
                '''real data test set'''
                file_path = os.path.join(self.path, "real_test_list.npy")
        else:
            raise ValueError("don't support this dataset setup")

        dicts = np.load(file_path, allow_pickle=True).item()

        self.X = dicts['X']
        self.y = dicts['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        label = self.y[idx]
        if image.mode != "RGB":
            image = image.convert('RGB')
        # do transform
        # for images
        if self.train:
            image = self.transform_train(image)
        else:
            image = self.transform_eval(image)
        label = label.astype(np.int64)
        return image, label


def get_visda(train=True, types='train', batch_size=64, path=None):
    visda = VisDA(train, types, path)
    dataloader = torch.utils.data.DataLoader(visda, batch_size, shuffle=train, num_workers=16, drop_last=train)
    return dataloader


if __name__ == "__main__":
    path = '/media/rtx/DA18EBFA09C1B27D/zw/data/visda'
    split_image(path=path, train=True)
    split_image(path=path, train=False)

    loader = get_visda(train=False, types="eval", batch_size=64, path='/media/rtx/DA18EBFA09C1B27D/zw/data/visda')
    print("batch_num:", len(loader))
    plt.ion()
    plt.figure(figsize=(8, 8))
    for (x, y) in loader:

        y = y.detach().numpy()
        print(y)
        print(y.dtype)
        print("image shape:", x[0].shape)
        print("dtype of image:", x.dtype)
    
        plt.cla()
        plt.axis('off')
        plt.title("visda data")
        imgs = vutils.make_grid(x[:32, :, :, :], padding=2, normalize=True).numpy()
        plt.imshow(np.transpose(imgs, (1, 2, 0)))
        plt.pause(1)
    plt.show()