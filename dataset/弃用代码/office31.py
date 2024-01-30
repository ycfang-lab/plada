from .office_loader import pre_process as prep
from .office_loader import data_list as data_list
from torch.utils.data import DataLoader


def get_office(train, root_dir, path, batch_size, test_10_crop=False):
    if train:
        transform = prep.image_train(resize_size=256, crop_size=224, alexnet=False)
    else:
        if test_10_crop:
            transform = prep.image_test_10crop(resize_size=256, crop_size=224, alexnet=False)
        else:
            transform = prep.image_test(resize_size=256, crop_size=224, alexnet=False)
    image_lists = open(path).readlines()

    if train and test_10_crop:
        datasets = [data_list.ImageList(root_dir, image_lists, transform=transform[i]) for i in range(10)]
        data_loader = [DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8) for dataset in datasets]
    else:
        dataset = data_list.ImageList(root_dir, image_lists, transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=train, num_workers=8, drop_last=True)
    return data_loader







