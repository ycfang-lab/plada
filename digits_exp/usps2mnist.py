import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np  
from dataset import unit_data,usps,mnist
import os 
import matplotlib.pyplot as plt
import utils
from networks import prototype,mnist_usps
import adapt_model 

def step1(batch_size=32,out_dim=10,c_lr=0.01,c_epochs=20,
         checkpoint_dir='./checkpoint/USPS2MNIST/all-out-',lamb=0.01,gamma=1.0,
         use_all=True,bn=False,train=True):
    # some defined hyperparameter
    image_size=28
    nc = 1 # gray image
    class_num = 10 # the size of label
    checkpoint_dir = checkpoint_dir+str(out_dim)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    # mnist dataset
    s_tr,s_te = unit_data.get_data('usps',batch_size,image_size,use_all)
    # usps dataset 
    t_tr,t_te = unit_data.get_data('mnist',batch_size,image_size,use_all)

    # build network model
    Gs = mnist_usps.CNN_Extract(nc,out_dim,bn).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim).to(device)

    # prepare optimizer
    c_optim = optim.SGD(list(Gs.parameters())+list(ps.parameters()),lr=c_lr,momentum=0.9,weight_decay=0.0005)

    # train source networks
    model = adapt_model.ProtoDA(device = device)
    model.step1(Gs,ps,s_tr,s_te,t_te,
                c_optim,c_epochs,checkpoint_dir,class_num,lamb,gamma,
                train,bn)

# 98.19%
def step2(batch_size=32,out_dim=10,
          g_lr=0.001,d_lr=0.001,ad_epochs=2000,loss_style='gan',
          checkpoint_dir='./checkpoint/USPS2MNIST/all-out-',
          use_all=True,bn=False,G_iter=1,D_iter=1,
          beta=0.01,margin=0.01,theta=1.0):
    # hyperparameter
    image_size=28
    nc = 1
    class_num=10
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    checkpoint_dir = checkpoint_dir+str(out_dim)

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("checkpoint_dir:",os.path.realpath(checkpoint_dir))

    # prepare data
    ad_data_tr = unit_data.unit_data('usps','mnist',
                                     batch_size,ad_epochs,
                                     image_size=image_size,
                                     use_all=use_all)
    
    s_data_tr,s_data_te = unit_data.get_data('usps',batch_size,image_size,use_all)
    t_data_tr,t_data_te = unit_data.get_data('mnist',batch_size,image_size,use_all)

    # create graph
    Gs = mnist_usps.CNN_Extract(nc,out_dim).to(device)
    Gt = mnist_usps.CNN_Extract(nc,out_dim).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim).to(device)
    pt = prototype.Simple_Prototype(class_num,out_dim).to(device)
    D = mnist_usps.Discriminator(class_num).to(device)
    
    #g_optim = optim.Adam(list(Gt.parameters())+list(pt.parameters()),lr=g_lr,betas=(0.5,0.999))
    g_optim = optim.Adam(list(Gt.parameters()),lr=g_lr,betas=(0.5,0.999))
    d_optim = optim.Adam(D.parameters(),lr=d_lr,betas=(0.5,0.999))

    model = adapt_model.ProtoDA(device=device)
    model.step2(Gs,ps,Gt,pt,D,loss_style,
                ad_data_tr,t_data_te,g_optim,d_optim,checkpoint_dir,ad_epochs,
                bn,G_iter,D_iter,beta,margin,theta)



def proto_analysis(checkpoint_dir='./checkpoint/USPS2MNIST/all-out-10'):
    class_num=10
    out_dim=10
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ps = prototype.Simple_Prototype(class_num,out_dim).to(device)
    pt = prototype.Simple_Prototype(class_num,out_dim).to(device)
    
    model = adapt_model.ProtoDA(device=device)
    model.proto_analysis(ps,pt,checkpoint_dir)

if __name__ == "__main__":
    import random 
    #Set random seem for reproducibility
    manualSeed = 666
    # manualSeed = random.randint(1,10000)
    print("Random Seed: ",manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    #step1()
    step2()

    proto_analysis()