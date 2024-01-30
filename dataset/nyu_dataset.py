import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os 
from PIL import Image
import matplotlib.pyplot as plt 
import torchvision.utils as vutils
import scipy.io as io 
import h5py
from utils import tools


# def path_helper(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#         print("create the path: %s"%path)
#
#
# class NYU(Dataset):
#     def __init__(self, mat_path, orign_rgb_path, orign_depth_path):
#         self.path = os.path.join(mat_path, 'nyu_depth_v2_labeled.mat')
#         self.rgb_path = orign_rgb_path
#         self.dep_path = orign_depth_path
#         path_helper(self.rgb_path)
#         path_helper(self.dep_path)
#
#         print(os.path.exists(self.path))
#
#
#         self.data = h5py.File(self.path, 'r')
#         # print(len(self.data['names']))
#         print(self.data['names'])
#         print(self.data['accelData'])
#         for i in range(894):
#             name = self.data['names'][0][i]
#             obj = self.data[name]
#             strr = "".join(chr(i) for i in obj[:])
#             print(strr, i)
#
#         # print(self.data.keys())
#
#     def __len__(self):
#         return None
#
#     def get_orign_rgb_image(self):
#         images = np.array(self.data['images'])
#         for i in range(len(images)):
#             image = images[i]
#             r = Image.fromarray(image[0]).convert('L')
#             g = Image.fromarray(image[1]).convert('L')
#             b = Image.fromarray(image[2]).convert('L')
#             image = Image.merge("RGB", (r, g, b))
#             image = image.transpose(Image.ROTATE_270)
#             #image.show()
#             save_path = os.path.join(self.rgb_path, str(i)+".jpg")
#             image.save(save_path, optimize=True)
#         print("finished!")
#
#
#     def get_orign_depth_image(self):
#         depths = np.array(self.data['depths'])
#         max = depths.max()
#
#         depths = depths / max * 255
#         depths = depths.transpose((0, 2, 1))
#
#         for i in range(len(depths)):
#             print(depths[i].dtype)
#             depth = Image.fromarray(depths[i])
#             depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
#             depth.show()
#             assert False
#
#
#     def get_label_dict(self):
#         label = self.data['labels']
#         label0 = np.array(label[0])
#         print(label0.max(), label0.min())
#         label0 = label0 / label0.max() * 255
#         img = Image.fromarray(label0)
#         img = img.transpose(Image.ROTATE_270)
#         img.show()
#
#     def __getitem__(self, idx):
#         return None
#
# def splits_info():
#     path = r"../data/NYU/splits.mat"
#     # mats = h5py.File(path, 'r')
#     mats = io.loadmat(path)
#     print(mats.keys())
#     print(mats['__header__'])
#     print(mats['__version__'])
#     print(mats['__globals__'])
#
#     print(mats['trainNdxs'].shape)
#     print(mats['testNdxs'].shape)
#
# def support_labels():
#     path = r"../data/NYU/support_labels.mat"
#     mats = io.loadmat(path)
#     print(mats.keys())
#     print(mats['__header__'])
#     print(mats['__version__'])
#     print(mats['__globals__'])
#
#     print(mats['supportLabels'][0])
#
# def data_analysis(root_dir):
#     category = os.listdir(root_dir)
#     num = []
#     for i in category:
#         sub_folder = os.path.join(root_dir, i)
#         print("sub_folder:", sub_folder)
#         num.append(len(os.listdir(sub_folder)))
#
#     print("total_num: %d" % sum(num))


class NYU(Dataset):
    def __init__(self, train, domain, path):
        super(NYU, self).__init__()
        self.path = path

        if domain == 'images':
            self.list_path = os.path.join(path, 'source.txt')
        else:
            self.list_path = os.path.join(path, 'target.txt')

        if train:
            self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                  # transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
        else:
            self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                  # transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])

        self.path_list, self.label_list = self.read_text()
        print("data_size:", len(self.path_list), 'Train:', train, "domain:", domain)

    def read_text(self):
        with open(self.list_path) as f:
            lines = f.readlines()
        path_list = []
        label_list = []
        for line in lines:
            line = line.strip().split()
            temp_path = self.path + line[0]
            # print(self.path)
            # print(temp_path)
            path_list.append(temp_path)
            label_list.append(int(line[1]))
        return path_list, label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        label = self.label_list[idx]
        img = Image.open(path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        label = np.array(label).astype(np.int64)
        return img, label


def dataset_analysis(root_path):
    files = os.listdir(root_path)
    nums = []
    for i in files:
        sub_path = os.path.join(root_path, i)
        print(sub_path, len(os.listdir(sub_path)))
        nums.append(len(os.listdir(sub_path)))

    print(sum(nums))

def group_path_label_pair(root_path, train, name):
    path_list = []
    label_list = []
    if train:
        cat1 = 'train'
        cat2 = 'images'
    else:
        cat1 = 'test'
        cat2 = 'depth'

    root_path_domain = os.path.join(root_path, cat1, cat2)

    categorys = os.listdir(root_path_domain)
    for i, category in enumerate(categorys):
        sub_path = os.path.join(root_path_domain, category)
        for j in os.listdir(sub_path):
            save_path = "/%s/%s/%s/%s" % (cat1, cat2, category, j)
            path_list.append(save_path)
            label_list.append(i)
    save_path = os.path.join(root_path, "%s" % name)
    with open(save_path, 'w') as f:
        lines = ["%s %d\n"%(i, j) for i, j in zip(path_list, label_list)]
        f.writelines(lines)
    print("finished!")


def get_nyu(train, domain, path, batch_size):
    dataset = NYU(train, domain, path)
    dataloader = DataLoader(dataset, batch_size, shuffle=train,
                            num_workers=8, drop_last=train)
    return dataloader

if __name__ == "__main__":
    # path = r"E:\dataset\NYU"
    # rgb_path = os.path.join(path, "Images", "RGB")
    # depth_path = os.path.join(path, "Images", "Depth")
    # data = NYU(path, rgb_path, depth_path)
    # data.get_label_dict()

    # splits_info()
    # support_labels()
    root_dir = r"E:\Mycode\domain_adaptation_plus\joint-adversarial-progresive\data\nyu\NYUD_multimodal"
    test_root_dir =r"E:\Mycode\domain_adaptation_plus\joint-adversarial-progresive\data\nyu\NYUD_multimodal\test\depth"
    # dataset_analysis(test_root_dir)
    # group_path_label_pair(root_dir, False, 'target.txt')

    # nyu_image_train = NYU(True, 'images', root_dir)
    # nyu_image_test = NYU(False, 'images', root_dir)
    # nyu_depth_train = NYU(True, 'depth', root_dir)
    # nyu_depth_test = NYU(False, 'depth', root_dir)
    # print(len(nyu_image_train), len(nyu_image_test), len(nyu_depth_train), len(nyu_depth_test))

    loader = get_nyu(True, 'depth', root_dir, 64)
    tools.show_batch_data(loader)



