import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np  
import dataset
import os 
import matplotlib.pyplot as plt
import utils
import networks 
import adapt_model 
import adda_model
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
def normal_target_only(train,exp,
                       checkpoint_dir,device,
                       batch_size=64,
                       c_lr=0.001,c_epochs=40,
                       bn=True,use_all=True):   
    print("target only exp:",exp)
    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))
    if exp == "svhn2mnist":
        class_num = 10
        image_size = 32
        nc = 1
        tr_data,te_data = dataset.mnist.get_train_and_test(batch_size,True,image_size=image_size,gray=True)
        Gs = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
    elif exp=='synthsign2gtsrb':
        class_num = 43
        image_size = 40
        nc=3
        tr_data,te_data = dataset.gtsrb.get_train_and_test(batch_size,image_size=image_size,gray=False)
        Gs = networks.synthsign_gtsrb.CNN_Extract(nc,class_num,bn).to(device)
    elif exp=="mnist2usps":
        class_num = 10
        image_size = 28
        nc = 1
        tr_data,te_data = dataset.usps.get_train_and_test(batch_size,use_all,None,None,image_size)
        Gs = networks.mnist_usps.CNN_Extract(nc,class_num,bn).to(device)
    elif exp=="synthdigits2svhn":
        class_num = 10
        image_size = 32
        nc = 3
        tr_data,te_data = dataset.svhn.get_train_and_test(batch_size,None,None,image_size,gray=False)
        Gs = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
    elif exp == "mnist2mnist_m":
        class_num = 10
        image_size = 32
        nc = 3
        tr_data,te_data = dataset.mnist_m.get_train_and_test(batch_size,None,None,image_size)
        Gs = networks.mnist_mnist_m.CNN_Extract(nc,class_num,bn).to(device)
    else:
        raise ValueError("don't support this experiments!")
    c_optim = optim.SGD(Gs.parameters(),lr=c_lr,momentum=0.9,weight_decay=0.0005)
    model = adapt_model.ProtoDA(device=device)
    model.train_conventional_network(tr_data,te_data,None,Gs,c_optim,train,bn,c_epochs,
                                    checkpoint_dir,device)
    
    
def normal_source_only(train,exp,
                       checkpoint_dir,device,
                       batch_size=64,
                       c_lr=0.001,c_epochs=40,
                       bn=True,use_all=True):   
    print("source only exp:",exp)
    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))
    if exp == "svhn2mnist":
        class_num = 10
        image_size = 32
        nc = 1
        s_tr_data,s_te_data = dataset.svhn.get_train_and_test(batch_size,None,None,image_size=image_size,gray=True)
        t_te_data = dataset.mnist.get_mnist(False,batch_size,True,None,None,image_size=image_size,gray=True)
        Gs = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
    elif exp=='synthsign2gtsrb':
        class_num = 43
        image_size = 40
        nc=3
        s_tr_data,s_te_data = dataset.synthsign.get_train_and_test(batch_size,None,None,image_size,gray=False)
        t_te_data = dataset.gtsrb.get_gtsrb(False,batch_size,None,None,image_size,gray=False)
        Gs = networks.synthsign_gtsrb.CNN_Extract(nc,class_num,bn).to(device)
    elif exp=="mnist2usps":
        class_num = 10
        image_size = 28
        nc = 1
        s_tr_data,s_te_data = dataset.mnist.get_train_and_test(batch_size,use_all,None,None,image_size,gray=True)
        t_te_data = dataset.usps.get_usps(False,batch_size,use_all,None,None,image_size)
        Gs = networks.mnist_usps.CNN_Extract(nc,class_num,bn).to(device)
    elif exp=="synthdigits2svhn":
        class_num = 10
        image_size = 32
        nc = 3
        s_tr_data,s_te_data = dataset.syndigits.get_train_and_test(batch_size,None,None,image_size,gray=False)
        t_te_data = dataset.svhn.get_svhn(False,batch_size,None,None,image_size,gray=False)
        Gs = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
    elif exp == "mnist2mnist_m":
        class_num = 10
        image_size = 32
        nc = 3
        s_tr_data,s_te_data = dataset.mnist.get_train_and_test(batch_size,True,None,None,image_size,False,True)
        t_te_data = dataset.mnist_m.get_mnist_m(False,batch_size,None,None,image_size)
        Gs = networks.mnist_mnist_m.CNN_Extract(nc,class_num,bn).to(device)
    else:
        raise ValueError("don't support this experiments!")
    c_optim = optim.SGD(Gs.parameters(),lr=c_lr,momentum=0.9,weight_decay=0.0005)
    model = adapt_model.ProtoDA(device=device)
    model.train_conventional_network(s_tr_data,s_te_data,t_te_data,Gs,c_optim,train,bn,c_epochs,
                                    checkpoint_dir,device)

def adda_method(train,exp,checkpoint_dir,device,batch_size,
                g_lr=0.0001,d_lr=0.0001,ad_epochs=2000,bn=True,
                G_iter=1,D_iter=1,test_step=100,use_all=True):
    print("adda training exp:",exp)
    # check checkpoint dir

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    if exp == "svhn2mnist":
        class_num = 10
        image_size = 32
        nc = 1
        s_tr_data = dataset.svhn.get_svhn(True,batch_size,None,None,image_size,gray=True)
        t_tr_data,t_te_data = dataset.mnist.get_train_and_test(batch_size,True,image_size=image_size,gray=True)
        Gs = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
        Gt = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
        D_class = networks.mnist_svhn.Discriminator
    elif exp=='synthsign2gtsrb':
        class_num = 43
        image_size = 40
        nc=3
        s_tr_data = dataset.synthsign.get_synthsign(True,batch_size,None,None,image_size,gray=False)
        t_tr_data,t_te_data = dataset.gtsrb.get_train_and_test(batch_size,image_size=image_size,gray=False)
        Gs = networks.synthsign_gtsrb.CNN_Extract(nc,class_num,bn).to(device)
        Gt = networks.synthsign_gtsrb.CNN_Extract(nc,class_num,bn).to(device)
        D_class = networks.synthsign_gtsrb.Discriminator
    elif exp=="mnist2usps":
        class_num = 10
        image_size = 28
        nc = 1
        s_tr_data = dataset.mnist.get_mnist(True,batch_size,use_all,None,None,image_size)
        t_tr_data,t_te_data = dataset.usps.get_train_and_test(batch_size,use_all,None,None,image_size)
        Gs = networks.mnist_usps.CNN_Extract(nc,class_num,bn).to(device)
        Gt = networks.mnist_usps.CNN_Extract(nc,class_num,bn).to(device)
        D_class = networks.mnist_usps.Discriminator
    elif exp=="synthdigits2svhn":
        class_num = 10
        image_size = 32
        nc = 3
        s_tr_data = dataset.syndigits.get_syndigits(True,batch_size,None,None,image_size,gray=False)
        t_tr_data,t_te_data = dataset.svhn.get_train_and_test(batch_size,None,None,image_size,gray=False)
        Gs = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
        Gt = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
        D_class = networks.mnist_svhn.Discriminator
    elif exp == "mnist2mnist_m":
        class_num = 10
        image_size = 32
        nc = 3
        s_tr_data = dataset.mnist.get_mnist(True,batch_size,use_all,None,None,image_size,gray=False,aug=True)
        t_tr_data,t_te_data = dataset.mnist_m.get_train_and_test(batch_size,None,None,image_size)
        Gs = networks.mnist_mnist_m.CNN_Extract(nc,class_num,bn).to(device)
        Gt = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
        D_class = networks.mnist_mnist_m
    else:
        raise ValueError("don't support this experiments!")

    D = D_class(class_num,bn).to(device)
    unit_dataloader = dataset.unit_data.unit_data_by_loader(s_tr_data,t_tr_data,ad_epochs)
    g_optim = optim.Adam(Gt.parameters(),lr=g_lr,betas=(0.5,0.999))
    d_optim = optim.Adam(D.parameters(),lr=d_lr,betas=(0.5,0.999))

    model = adda_model.ADDA(device)
    model.step2(Gs,Gt,D,unit_dataloader,t_te_data,
                g_optim,d_optim,checkpoint_dir,ad_epochs,
                bn,G_iter,D_iter,test_step)

def cpl_target_only(train,exp,
                    checkpoint_dir,device,
                    batch_size=64,out_dim=10,
                    c_lr=0.001,c_epochs=40,
                    bn=True,use_all=True,gamma=1.0,lamb=0.01):
    print("cpl target only exp:",exp)
    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))
    if exp == "svhn2mnist":
        class_num = 10
        image_size = 32
        nc = 1
        tr_data,te_data = dataset.mnist.get_train_and_test(batch_size,True,image_size=image_size,gray=True)
        Gs = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    elif exp=='synthsign2gtsrb':
        class_num = 43
        image_size = 40
        nc=3
        tr_data,te_data = dataset.gtsrb.get_train_and_test(batch_size,image_size=image_size,gray=False)
        Gs = networks.synthsign_gtsrb.CNN_Extract(nc,out_dim,bn).to(device)
    elif exp=="mnist2usps":
        class_num = 10
        image_size = 28
        nc = 1
        tr_data,te_data = dataset.usps.get_train_and_test(batch_size,use_all,None,None,image_size)
        Gs = networks.mnist_usps.CNN_Extract(nc,out_dim,bn).to(device)
    elif exp=="synthdigits2svhn":
        class_num = 10
        image_size = 32
        nc = 3
        tr_data,te_data = dataset.svhn.get_train_and_test(batch_size,None,None,image_size,gray=False)
        Gs = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    elif exp == "mnist2mnist_m":
        class_num = 10
        image_size = 32
        nc = 3
        tr_data,te_data = dataset.mnist_m.get_train_and_test(batch_size,None,None,image_size)
        Gs = networks.mnist_mnist_m.CNN_Extract(nc,out_dim,bn).to(device)
    else:
        raise ValueError("don't support this experiments!")
    p = networks.prototype.Simple_Prototype(class_num,out_dim,gamma).to(device)

    c_optim = optim.SGD(list(Gs.parameters())+list(p.parameters()),lr=c_lr,momentum=0.9,weight_decay=0.0005)
    model = adapt_model.ProtoDA(device=device)
    model.step1(Gs,p,tr_data,te_data,None,
                c_optim,c_epochs,checkpoint_dir,class_num,lamb,gamma,
                train,bn)

def cpl_source_only(train,exp,
                    checkpoint_dir,device,
                    batch_size=64,out_dim=10,
                    c_lr=0.001,c_epochs=40,
                    bn=True,use_all=True,gamma=1.0,lamb=0.01):
    print("cpl source only:",exp)
    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))
    if exp == "svhn2mnist":
        class_num = 10
        image_size = 32
        nc = 1
        s_tr_data,s_te_data = dataset.svhn.get_train_and_test(batch_size,None,None,image_size=image_size,gray=True)
        t_te_data = dataset.mnist.get_mnist(False,batch_size,True,None,None,image_size=image_size,gray=True)
        Gs = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    elif exp=='synthsign2gtsrb':
        class_num = 43
        image_size = 40
        nc=3
        s_tr_data,s_te_data = dataset.synthsign.get_train_and_test(batch_size,None,None,image_size,gray=False)
        t_te_data = dataset.gtsrb.get_gtsrb(False,batch_size,None,None,image_size,gray=False)
        Gs = networks.synthsign_gtsrb.CNN_Extract(nc,out_dim,bn).to(device)
    elif exp=="mnist2usps":
        class_num = 10
        image_size = 28
        nc = 1
        s_tr_data,s_te_data = dataset.mnist.get_train_and_test(batch_size,use_all,None,None,image_size,gray=True)
        t_te_data = dataset.usps.get_usps(False,batch_size,use_all,None,None,image_size)
        Gs = networks.mnist_usps.CNN_Extract(nc,out_dim,bn).to(device)
    elif exp=="synthdigits2svhn":
        class_num = 10
        image_size = 32
        nc = 3
        s_tr_data,s_te_data = dataset.syndigits.get_train_and_test(batch_size,None,None,image_size,gray=False)
        t_tr_data = dataset.svhn.get_svhn(False,batch_size,None,None,image_size,gray=False)
        Gs = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    elif exp == "mnist2mnist_m":
        class_num = 10
        image_size = 32
        nc = 3
        s_tr_data,s_te_data = dataset.mnist.get_train_and_test(batch_size,True,None,None,image_size,False,True)
        t_te_data = dataset.mnist_m.get_mnist_m(False,batch_size,None,None,image_size)
        Gs = networks.mnist_mnist_m.CNN_Extract(nc,out_dim,bn).to(device)
    else:
        raise ValueError("don't support this experiments!")
    p = networks.prototype.Simple_Prototype(class_num,out_dim,gamma).to(device)

    c_optim = optim.SGD(list(Gs.parameters())+list(p.parameters()),lr=c_lr,momentum=0.9,weight_decay=0.0005)
    model = adapt_model.ProtoDA(device=device)
    model.step1(Gs,p,s_tr_data,s_data_te,t_te_data,
                c_optim,c_epochs,checkpoint_dir,class_num,lamb,gamma,
                train,bn)
    
def cpl_adversarial_training(train,exp,checkpoint,method,device,
                        out_dim=10,g_lr=0.0001,d_lr=0.0001,ad_epochs=2000,
                        gan_style='gan',bn=True,
                        G_iter=1,D_iter=1,beta=0.01,test_step=100):
    print("cpl adversarial training exp:",exp)
    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    if exp == "svhn2mnist":
        class_num = 10
        image_size = 32
        nc = 1
        s_tr_data = dataset.svhn.get_svhn(True,batch_size,None,None,image_size,gray=True)
        t_tr_data,t_te_data = dataset.mnist.get_train_and_test(batch_size,True,image_size=image_size,gray=True)
        Gs = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
        Gt = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
        D_class = networks.mnist_svhn.Discriminator
    elif exp=='synthsign2gtsrb':
        class_num = 43
        image_size = 40
        nc=3
        s_tr_data = dataset.synthsign.get_synthsign(True,batch_size,None,None,image_size,gray=False)
        t_tr_data,t_te_data = dataset.gtsrb.get_train_and_test(batch_size,image_size=image_size,gray=False)
        Gs = networks.synthsign_gtsrb.CNN_Extract(nc,out_dim,bn).to(device)
        Gt = networks.synthsign_gtsrb.CNN_Extract(nc,out_dim,bn).to(device)
        D_class = networks.synthsign_gtsrb.Discriminator
    elif exp=="mnist2usps":
        class_num = 10
        image_size = 28
        nc = 1
        s_tr_data = dataset.mnist.get_mnist(True,batch_size,use_all,None,None,image_size)
        t_tr_data,t_te_data = dataset.usps.get_train_and_test(batch_size,use_all,None,None,image_size)
        Gs = networks.mnist_usps.CNN_Extract(nc,out_dim,bn).to(device)
        Gt = networks.mnist_usps.CNN_Extract(nc,out_dim,bn).to(device)
        D_class = networks.mnist_usps.Discriminator
    elif exp=="synthdigits2svhn":
        class_num = 10
        image_size = 32
        nc = 3
        s_tr_data = dataset.syndigits.get_syndigits(True,batch_size,None,None,image_size,gray=False)
        t_tr_data,t_te_data = dataset.svhn.get_train_and_test(batch_size,None,None,image_size,gray=False)
        Gs = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
        Gt = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
        D_class = networks.mnist_svhn.Discriminator
    elif exp == "mnist2mnist_m":
        class_num = 10
        image_size = 32
        nc = 3
        s_tr_data = dataset.mnist.get_mnist(True,batch_size,use_all,None,None,image_size,gray=False,aug=True)
        t_tr_data,t_te_data = dataset.mnist_m.get_train_and_test(batch_size,None,None,image_size)
        Gs = networks.mnist_mnist_m.CNN_Extract(nc,out_dim,bn).to(device)
        Gt = networks.mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
        D_class = networks.mnist_mnist_m
    else:
        raise ValueError("don't support this experiments!")
    p = networks.prototype.Simple_Prototype(class_num,out_dim,gamma).to(device)
    if method=='a':
        D = D_class(out_dim,bn).to(device)
    elif method == 'b':
        D = D_class(class_num,bn).to(device)
    
    uni_dataloader = dataset.unit_data.unit_data_by_loader(s_tr_data,t_tr_data,ad_epochs)
    g_optim = optim.Adam(list(Gt.parameters()),lr=g_lr,betas=(0.5,0.999))
    d_optim = optim.Adam(D.parameters(),lr=d_lr,betas=(0.5,0.999))

    model = adapt_model.ProtoDA(device)
    model.step2(Gs,p,Gt,D,gan_style,
                uni_dataloader,t_te_data,
                g_optim,d_optim,checkpoint_dir,ad_epochs,
                bn,G_iter,D_iter,beta,test_step=test_step,method=method,train=train)

def plot_tsne_plot(types='orign',name='figure',show_proto=True):
    image_size = 32
    # for svhn2mnist2
    #tr_data = dataset.svhn.get_svhn(True,-1,None,None,image_size,gray=True)
    #te_data = dataset.mnist.get_mnist(False,-1,True,None,None,image_size=image_size,gray=True)
    
    #for mnist2mnist_m
    tr_data = dataset.mnist.get_mnist(True,-1,True,None,None,32,False,True)
    te_data = dataset.mnist_m.get_mnist_m(False,-1,None,None,32)
    (s_image,s_label) = next(iter(tr_data))
    (t_image,t_label) = next(iter(te_data))
    fig= plt.figure()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu") 
    if types=='orign':
        s_image = s_image.reshape((s_image.size(0),-1)).numpy()
        t_image = t_image.reshape((t_image.size(0),-1)).numpy()
        s_label = s_label.numpy()
        t_label = t_label.numpy()
        utils.plot_tsne_da(fig,False,None,s_image[:2000],s_label[:2000],t_image[:2000],t_label[:2000],
                           samples=500,name="orign data")

    elif types=='normal_net_source_only':
        class_num = 10
        out_dim=10
        gamma=1.0
        checkpoint_dir= './checkpoint/MNIST2MNIST_M/'
        
        #Gs = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
        Gs = networks.mnist_mnist_m.CNN_Extract(3,10,True).to(device)
        ps = networks.prototype.Simple_Prototype(class_num,out_dim,gamma).to(device)
        Gs.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_Gs.pt")))
        ps.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_ps.pt")))
        s_feat = Gs(s_image[:2000].to(device)).to('cpu').detach().numpy()
        t_feat = Gs(t_image[:2000].to(device)).to('cpu').detach().numpy()
        s_label = s_label.numpy()
        t_label = t_label.numpy()
        proto = ps.proto.detach().to('cpu').numpy()
        utils.plot_tsne_da(fig,True,proto,s_feat,s_label[:2000],t_feat,t_label[:2000],
                           samples=400,name="source only",device=device)

    elif types == 'source_only':
        class_num = 10
        out_dim=10
        gamma=1.0
        checkpoint_dir= './checkpoint/MNIST2MNIST_M/bn-all-out-10'
        
        #Gs = networks.mnist_svhn.CNN_Extract(nc,class_num,bn).to(device)
        Gs = networks.mnist_mnist_m.CNN_Extract(3,10,True).to(device)
        ps = networks.prototype.Simple_Prototype(class_num,out_dim,gamma).to(device)
        Gs.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_Gs.pt")))
        ps.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_ps.pt")))
        s_feat = Gs(s_image[:2000].to(device)).to('cpu').detach().numpy()
        t_feat = Gs(t_image[:2000].to(device)).to('cpu').detach().numpy()
        s_label = s_label.numpy()
        t_label = t_label.numpy()
        proto = ps.proto.detach().to('cpu').numpy()
        utils.plot_tsne_da(fig,True,proto,s_feat,s_label[:2000],t_feat,t_label[:2000],
                           samples=400,name="source only",device=device)
    
    elif types == 'adapt':
        class_num = 10
        nc=1
        bn=True 
        out_dim=10
        gamma=1.0
        #checkpoint_dir = './checkpoint/SVHN2MNIST/seed-256-a-nowpl-bn-all-out-10-8738'
        #checkpoint_dir = './checkpoint/SVHN2MNIST/seed-384-b-nowpl-bn-all-out-10-8933'
        checkpoint_dir= './checkpoint/MNIST2MNIST_M/seed-955-b-no-nowpl-all-out-10-9403'

        print("checkpoint_dir:",checkpoint_dir)
        Gs = networks.mnist_mnist_m.CNN_Extract(3,10,bn).to(device)
        ps = networks.prototype.Simple_Prototype(class_num,out_dim,gamma).to(device)
        Gt = networks.mnist_mnist_m.CNN_Extract(3,10,bn).to(device)
        Gs.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_Gs.pt")))
        ps.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_ps.pt")))
        Gt.load_state_dict(torch.load(os.path.join(checkpoint_dir,"b-Gt.pt")))

        s_feat = Gs(s_image[:2000].to(device)).to('cpu').detach().numpy()
        t_feat = Gt(t_image[:2000].to(device)).to('cpu').detach().numpy()
        s_label = s_label.numpy()
        t_label = t_label.numpy()
        
        proto = ps.proto.detach().to('cpu').numpy()
        utils.plot_tsne_da(fig,show_proto,proto,s_feat,s_label[:2000],t_feat,t_label[:2000],
                           samples=500,name=name,device=device)
    plt.show()



def plot_feature(checkpoint_dir="./checkpoint/TOY/out-",method='target',ad_method='b',name='toy-plada-b'):
    out_dim = 2
    class_num = 2
    cm_bright = ListedColormap(['red','green'])
    #device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cpu")
    checkpoint_dir = checkpoint_dir+str(out_dim)
    # form classifier
    source_g = networks.toy_net.network(out_dim).to(device)
    target_g = networks.toy_net.network(out_dim).to(device)
    ps = networks.prototype.Simple_Prototype(class_num,out_dim).to(device)
    source_g.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_Gs.pt")))
    ps.load_state_dict(torch.load(os.path.join(checkpoint_dir,'source_best_ps.pt')))
    
    target_file_name = ad_method+"-Gt"+".pt"
    target_g.load_state_dict(torch.load(os.path.join(checkpoint_dir,target_file_name)))

    # get orign data
    source_data = dataset.toy_data.get_toy(True,300,300,'source')
    target_data = dataset.toy_data.get_toy(True,300,300,'target')

    print(target_data)
    x,y = next(iter(source_data))
    new_x,new_y = next(iter(target_data))

    # output feat 
    
    source_feat = source_g(x).detach().cpu().numpy()
    target_feat_no_out = target_g(new_x)
    target_feat = target_g(new_x).detach().cpu().numpy()
    proto = ps.proto.detach().numpy()
    y,new_y = y.detach().cpu().numpy(),new_y.detach().cpu().numpy()

    print(source_feat.shape)
    print(target_feat.shape)
    print(proto.shape)
    fig= plt.figure()
    ax = fig.add_subplot(111)
    
    # # grid prepare
    h=.005
    x_min,x_max = source_feat[:,0].min()-.2,source_feat[:,0].max() + 0.2
    y_min,y_max = source_feat[:,1].min()-.2,source_feat[:,1].max() + 0.2
    # x_min,x_max = x[:,0].min()-.5,x[:,0].max() + 0.5
    # y_min,y_max = x[:,1].min()-.5,x[:,1].max() + 0.5 
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h)) 
     
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['red','green'])
    # get classification boundary
    inputs = np.c_[xx.ravel(),yy.ravel()]
    inputs = torch.from_numpy(inputs).float()

    source_result = ps.forward_feat(inputs,device).reshape(xx.shape)
    target_result = ps.forward_feat(inputs,device).reshape(xx.shape)
    ax.contourf(xx,yy,source_result.detach().numpy(),cmap=cm_bright,alpha=.2)
    #ax.contourf(xx,yy,target_result.detach().numpy(),cmap=cm_bright,alpha=.1)
    ax.contour(xx,yy,source_result.detach().numpy(),levels=[0],cmap="Greys_r",linewidths=3,linestyles='solid')
    #ax.contour(xx,yy,target_result.detach().numpy(),levels=[0],cmap="Greys_r",linewidths=3,linestyles='dashed')
    
    
    ax.scatter(source_feat[:,0],source_feat[:,1],c=y,label='source',cmap=cm_bright,marker='.',alpha=0.6)
    # X = source_feat
    # for i in range(X.shape[0]):
    #     plt.text(X[i,0],X[i,1],str(y[i]),color=plt.cm.bwr(0/ 1.), fontdict={"weight":"bold","size":9},alpha=0.6)

    ax.scatter(target_feat[:,0],target_feat[:,1],label='target',c='b',marker='.',alpha=0.6)

    X = target_feat
    pred_y = ps.forward_feat(target_feat_no_out,device).detach().cpu().numpy()
    
    for i in range(X.shape[0]):
        if pred_y[i]!=new_y[i]:
            plt.text(X[i,0],X[i,1],str(new_y[i]), fontdict={"weight":"bold","size":9},alpha=0.6,color='cyan')
    ax.scatter(proto[:,0],proto[:,1],c='y',label='prototype',s=100,marker='o',zorder=10,edgecolors='black')

    for i in range(proto.shape[0]):
            #ax.text(ps[i,0],ps[i,1],str(i),color='cyan', fontdict={"weight":"bold","size":9},alpha=1.0)
            #digits_img = plt.imread('./data/svhn/example/'+str(i+1)+".jpg",format='jpg')

            #imagebox = OffsetImage(digits_img,zoom=0.2)
        imagebox = TextArea(str(i),minimumdescent=False,textprops=dict(size=30))
        #imagebox.image.axes=ax
        ab = AnnotationBbox(imagebox,proto[i,:],pad=0.2,boxcoords="offset points",xybox=(-20,20),
                            arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=3"))
        ax.add_artist(ab)


    ax.set_xticks([]),ax.set_yticks([])
    #plt.legend()
    plt.savefig("./result/toy/"+name+".png",bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    #checkpoint_dir='./checkpoint/SYNTHSIGN2GTSRB/bn-normal-target-only'
    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # normal_target_only(exp="synthsign2gtsrb",
    #                     checkpoint_dir=checkpoint_dir,
    #                     device=device,train=False,batch_size=64,c_lr=0.001,c_epochs=40,bn=True)
    #plot_tsne_plot('adapt','pada-a-adapted-svhn2mnist')
    #plot_tsne_plot('adapt','mnist2mnistmnowpl')
    plot_feature()