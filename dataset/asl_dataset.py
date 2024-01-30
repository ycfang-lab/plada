from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np


def get_image_list_by_key_and_path(key, path):
    label_list = []
    image_list = []

    categorys = os.listdir(path)

    for ids, cate in enumerate(categorys):
        images_folder = os.path.join(path, cate)
        images_names = [i for i in os.listdir(images_folder) if i.startswith(key)]
        image_list.extend(["{}/{}/{}".format(path[-1], cate, i) for i in images_names])
        label_list.extend([ids for i in range(len(images_names))])
    return image_list, label_list


def write_lists_into_file(path_list, label_list, save_path):
    strs = ""
    with open(save_path, 'w') as f:
        for i in range(len(path_list)):
            strs += "{} {}\n".format(path_list[i], label_list[i])
        f.writelines(strs)
    print("finished!")


def train_test_split_by_user(path, save_path, domain, out_user='A'):
    user_list = ['A', 'B', 'C', 'D', 'E']
    if out_user in user_list:
        test_split = os.path.join(path, out_user)
        user_list.remove(out_user)
        train_split = [os.path.join(path, path_user) for path_user in user_list]
    else:
        raise ValueError("not suport this user out protocol")

    if domain == 'color':
        key_word = 'color'
    else:
        key_word = 'depth'

    # train split:
    tr_image, tr_label = [], []
    for i in train_split:
        print(i)
        images, labels = get_image_list_by_key_and_path(key_word, i)
        tr_image.extend(images)
        tr_label.extend(labels)

    # test split:
    te_image, te_label = [], []
    images, labels = get_image_list_by_key_and_path(key_word, test_split)
    te_image.extend(images)
    te_label.extend(labels)

    tr_save_path = os.path.join(save_path, '{}_out_{}_train.txt'.format(key_word, out_user))
    te_save_path = os.path.join(save_path, "{}_out_{}_test.txt".format(key_word, out_user))

    write_lists_into_file(tr_image, tr_label, tr_save_path)
    write_lists_into_file(te_image, te_label, te_save_path)


class aslfsp(Dataset):
    def __init__(self, train, domain, path, split_user, root_path=None):
        super(aslfsp, self).__init__()

        if root_path is None:
            root_path = path
        if train:
            self.path = os.path.join(path, "da", "{}_out_{}_train.txt".format(domain, split_user))
        else:
            self.path = os.path.join(path, "da", "{}_out_{}_test.txt".format(domain, split_user))

        self.train = train
        self.split_suer = split_user
        self.domain = domain

        self.image_list, self.label_list = self.read_from_txt(self.path, root_path)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform_train = transforms.Compose([transforms.Resize(224),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize])

        self.transform_test = transforms.Compose([transforms.Resize(224),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  normalize])

    def read_from_txt(self, path, root_dir):
        image_list, label_list = [], []
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            image_list.append(os.path.join(root_dir, line[0]))
            label_list.append(line[1])
        return image_list, label_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])

        # img = np.array(img)
        # print("max:", np.max(img), "min:", np.min(img))
        # assert False
        if self.domain == 'depth':
            img = np.array(img)
            img = img / 9235 * 255.0
            img = Image.fromarray(img).convert('RGB')

        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)

        lab = np.array(self.label_list[idx]).astype(np.int64)
        return img, lab


def get_by_domain(train, domain, path, batch_size, root_dir, split_user):
    dataset = aslfsp(False, domain, path, split_user, root_dir)
    dataloader = DataLoader(dataset, batch_size, shuffle=train, num_workers=8, drop_last=train)
    return dataloader