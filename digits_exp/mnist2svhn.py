import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np  
from dataset import unit_data, svhn, mnist
import os 
import matplotlib.pyplot as plt
import utils
from networks import prototype, mnist_svhn
import adapt_model 

def step1(batch_size=64,out_dim=10,
         path_source=None, path_target=None,
         c_lr=0.001,c_epochs=20,
         checkpoint_dir='./checkpoint/MNIST2SVHN/bn_all-out-',lamb=0.01,gamma=1.0,
         use_all=True,bn=True,train=True, device="cuda:0"):
    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")
    # some defined hyperparameter
    image_size=32
    nc = 3 # gray image
    class_num = 10 # the size of label
    checkpoint_dir = checkpoint_dir+str(out_dim)
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    # mnist dataset
    s_tr,s_te = mnist.get_train_and_test(batch_size,use_all,None, path=path_source,image_size=image_size,gray=False)
    # usps dataset 
    t_te = svhn.get_svhn(False,batch_size,path=path_target, image_size=image_size,gray=False)

    # build network model
    Gs = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim).to(device)

    # prepare optimizer
    c_optim = optim.SGD(list(Gs.parameters())+list(ps.parameters()),lr=c_lr,momentum=0.9,weight_decay=0.0005)

    # train source networks
    model = adapt_model.ProtoDA(device = device)
    model.step1(Gs,ps,s_tr,s_te,t_te,
                c_optim,c_epochs,checkpoint_dir,class_num,lamb,gamma,
                train,bn)


def step2(batch_size=64,out_dim=10, path_source=None, path_target=None,
            g_lr=0.0001,d_lr=0.0001,ad_epochs=2000,gan_style='gan',
            checkpoint_dir='./checkpoint/MNIST2SVHN/bn_all-out-',
            use_all=True,bn=True,G_iter=1,D_iter=1,
            beta=0.01,margin=0.01,theta=1.0, device='cuda:0'):
    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")
    image_size=32
    nc =3 # gray image
    class_num = 10 # the size of label
    checkpoint_dir = checkpoint_dir+str(out_dim)
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    tr_s_dataloader = mnist.get_mnist(True,batch_size,use_all,None, path=path_source, image_size=image_size,gray=False)
    tr_t_dataloader,te_t_dataloader = svhn.get_train_and_test(batch_size,path=path_target, image_size=image_size,gray=False)
    uni_dataloader = unit_data.unit_data_by_loader(tr_s_dataloader,tr_t_dataloader,ad_epochs)

    # build model
    Gs = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim).to(device)
    
    Gt = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    pt = prototype.Simple_Prototype(class_num,out_dim).to(device)
    
    D = mnist_svhn.Discriminator(class_num,bn).to(device)
    
    # prepare optimizer
    g_optim = optim.Adam(list(Gt.parameters()),lr=g_lr,betas=(0.5,0.999))
    d_optim = optim.Adam(D.parameters(),lr=d_lr,betas=(0.5,0.999))

    #build model and train
    model = adapt_model.ProtoDA(device)
    model.step2(Gs,ps,Gt,pt,D,gan_style,
                uni_dataloader,te_t_dataloader, g_optim,d_optim,checkpoint_dir,ad_epochs,
                bn,G_iter,D_iter,beta,margin,theta)



if __name__ == "__main__":
    import random
    # Set random seem for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1,10000)
    print("Random Seed: ",manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    step1()
    step2()