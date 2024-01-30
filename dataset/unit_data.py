import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 
import dataset.mnist as mnist
import dataset.usps as usps
import dataset.svhn as svhn

def unit_data_by_loader(source_loader,target_loader,epochs):
    uni_train_data_loader = iter_2_data(source_loader,target_loader,epochs)
    print("unit_data_loader batch_num:{}".format(len(uni_train_data_loader)))
    return uni_train_data_loader

def unit_data(source,target,batch_size,epochs,image_size=28,use_all=True):

    s_loader_tr = dataset_select(source)(True,batch_size,use_all,image_size=image_size)
    #s_loader_te = dataset_select(source)(False,image_size=image_size)
    t_loader_tr = dataset_select(target)(True,batch_size,use_all,image_size=image_size)
    #t_loader_te = dataset_select(target)(False,image_size=image_size)
    train_data_loader = iter_2_data(s_loader_tr,t_loader_tr,epochs)

    print("union data batch_num:{}".format(len(train_data_loader)))
    return train_data_loader#,s_loader_tr,s_loader_te,t_loader_tr,t_loader_te

def get_data(name,batch_size,image_size=28,use_all=True):
    if name=='mnist' or 'name'=='usps':
        tr_data_loader = dataset_select(name)(True,batch_size,use_all,image_size=image_size)
    else:
        tr_data_loader = dataset_select(name)(True,batch_size,image_size=image_size)
    te_data_loader = dataset_select(name)(False,batch_size,image_size=image_size)
    return tr_data_loader,te_data_loader 


def dataset_select(name):
    if name == "mnist":
        return mnist.get_mnist 
    elif name == "usps":
        return usps.get_usps
    elif name == 'svhn':
        return svhn.get_svhn
    else:
        raise ValueError("Don't support this dataset:{}".format(name))


class iter_2_data():
    def __init__(self,loader1,loader2,max_data_size):
        self.data_loader_A = loader1
        self.data_loader_B = loader2 
        self.max_data_size=max_data_size # max epoch the data can be iteration

    def __len__(self):
        return self.max_data_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False 
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self 
    
    def __next__(self):
        A,A_label = None,None
        B,B_label = None,None 
        try:
            A,A_label = next(self.data_loader_A_iter)

        except StopIteration:
            if A is None or A_label is None:
                self.data_loader_A_iter = iter(self.data_loader_A)
                A,A_label = next(self.data_loader_A_iter)

        try:
            B,B_label = next(self.data_loader_B_iter)

        except StopIteration:
            if B is None or B_label is None:
                self.data_loader_B_iter = iter(self.data_loader_B)
                B,B_label = next(self.data_loader_B_iter)

        if self.iter > self.max_data_size:
            raise StopIteration() 
        else:
            self.iter += 1
        return A,A_label,B,B_label