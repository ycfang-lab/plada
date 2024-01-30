from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def image_list(path,data_path):
    image_list = []
    label_list = []
    with open(data_path) as f:
        lines = f.readlines()
    for i in lines:
        node = i.strip().split()
        image_list.append(os.path.join(path, node[0]))
        label_list.append(int(node[1]))
    return image_list, label_list

class visda(Dataset):
    def __init__(self, domain, train, path=None):
        super(visda, self).__init__()

        if domain=='source':
            self.data_path = os.path.join(path,'train')
            self.list_path = os.path.join(self.data_path,'image_list.txt')
        elif domain=='target':
            self.data_path = os.path.join(path,'validation')
            self.list_path = os.path.join(self.data_path,'image_list.txt')

        self.image_list, self.label_list = image_list(self.data_path, self.list_path)

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


def get_visda(train, domain, path, batch_size):
    dataset = visda(domain, train, path)
    dataloader = DataLoader(dataset, batch_size, shuffle=train, num_workers=4, drop_last=train)
    return dataloader
''''''

if __name__ == "__main__":
    # loader = get_clef(True, 'i', r"E:\Mycode\Domain Adaptation\data\imageclef\image_CLEF" , 64)
    # print(type(loader))
    # print("batch_num:", len(loader))
    #
    # batch = next(iter(loader))
    # print(len(batch))
    # first = batch[0][0].numpy()
    # label = batch[1].numpy()
    # print(label)
    # print("image shape:", first.shape)
    # print("label shape and dtype:", label.shape, label.dtype)
    # print("max pixel in image:", np.max(first))
    # print("min pixel in image:", np.min(first))
    # print("dtype of image:", first.dtype)
    #
    # plt.figure(figsize=(8, 8))
    # plt.axis('off')
    # plt.title("OFFICE data")
    # imgs = vutils.make_grid(batch[0][:64], padding=2, normalize=True).numpy()
    # plt.imshow(np.transpose(imgs, (1, 2, 0)))
    # plt.show()

    data_path = '/media/disk1/zw/data/visda-c/train'
    list_path = '/media/disk1/zw/data/visda-c/train/image_list.txt'
    image,label = image_list(data_path,list_path)
    print(len(image))
    print(len(label))
