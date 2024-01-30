import torch
import torch.optim as optim
from dataset import unit_data,svhn,mnist
import os
from networks import prototype,mnist_svhn
import adapt_model_v as adapt_model
from digits_exp import solvers


def adda_adapt_model(train=False, checkpoint_dir='./checkpoint/SVHN2MNIST/bn-normal-source-only',
                    batch_size=64,g_lr=0.000001,d_lr=0.0000001,ad_epochs=2000,
                    bn=True,G_iter=1,D_iter=1,test_step=30):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    solvers.adda_method(train, 'svhn2mnist', checkpoint_dir, device, batch_size,
                        g_lr, d_lr, ad_epochs, bn,
                        G_iter, D_iter, test_step)

def normal_target_only(train=False,
                       checkpoint_dir='./checkpoint/SVHN2MNIST/bn-normal-target-only',
                       batch_size=64,c_lr=0.001,c_epochs=40,bn=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    solvers.normal_target_only(exp="svhn2mnist",
                               checkpoint_dir=checkpoint_dir,
                               device=device, train=train, batch_size=batch_size,
                               c_lr=c_lr, c_epochs=c_epochs, bn=bn)

def normal_source_only(train=False,
                      checkpoint_dir='./checkpoint/SVHN2MNIST/bn-normal-source-only',
                      batch_size=64,c_lr=0.001,c_epochs=40,bn=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    solvers.normal_source_only(train, "svhn2mnist",
                               checkpoint_dir, device,
                               batch_size, c_lr, c_epochs, bn)

def cpl_target_only(train=False,
                    checkpoint_dir='./checkpoint/SVHN2MNIST/bn-cpl-target-only',
                    batch_size=64,c_lr=0.001,c_epochs=20,bn=True,gamma=1.0,lamb=0.01,out_dim=10):
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")
    checkpoint_dir = checkpoint_dir+"-all-out-"+str(out_dim)
    solvers.cpl_target_only(train, "svhn2mnist", checkpoint_dir, device,
                            batch_size, out_dim, c_lr, c_epochs, bn, gamma=1.0, lamb=0.01)
    

def step1(batch_size=64,out_dim=10,
         path_source=None,path_target=None,
         c_lr=0.01,c_epochs=20,
         checkpoint_dir='./checkpoint/SVHN2MNIST/bn-all-out-',lamb=0.01,gamma=1.0,
         use_all=True,bn=True,train=True, device='cuda:0'):
    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")
    # some defined hyperparameter
    image_size=32
    nc = 1 # gray image
    class_num = 10 # the size of label
    checkpoint_dir = checkpoint_dir+str(out_dim)
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    # mnist dataset
    s_tr,s_te = svhn.get_train_and_test(batch_size,path=path_source, image_size=image_size,gray=True)
    # usps dataset 
    t_te = mnist.get_mnist(False,batch_size,use_all,path_target,image_size=image_size,gray=True)

    # build network model
    Gs = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim, device).to(device)

    # prepare optimizer
    c_optim = optim.SGD(list(Gs.parameters())+list(ps.parameters()),lr=c_lr,momentum=0.9,weight_decay=0.0005)

    # train source networks
    model = adapt_model.ProtoDA(device = device)
    model.step1(Gs,ps,s_tr,s_te,t_te,
                c_optim,c_epochs,checkpoint_dir,class_num,lamb,gamma,
                train,bn)
    # model.step1(Gs,ps,s_tr,s_te,None,
    #             c_optim,c_epochs,checkpoint_dir,class_num,lamb,gamma,
    #             train,bn)

# need bacthc normalizatin.
def step2(batch_size=64,out_dim=10, path_source=None, path_target=None,
            g_lr=0.0001,d_lr=0.0001,ad_epochs=2000,gan_style='gan',
            checkpoint_dir='./checkpoint/SVHN2MNIST/bn-all-out-',
            use_all=True,bn=True,G_iter=1,D_iter=1,
            beta=0.1,method='b',train=True,test_step=100, device='cuda:0'):
    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")
    image_size=32
    nc =1 # gray image
    class_num = 10 # the size of label
    checkpoint_dir = checkpoint_dir+str(out_dim)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    tr_s_dataloader = svhn.get_svhn(True,batch_size,path=path_source, image_size=image_size,gray=True)
    tr_t_dataloader,te_t_dataloader = mnist.get_train_and_test(batch_size,use_all,path=path_target, image_size=image_size,gray=True)
    uni_dataloader = unit_data.unit_data_by_loader(tr_s_dataloader,tr_t_dataloader,ad_epochs)

    # build model
    Gs = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim,device).to(device)
    
    Gt = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    pt = prototype.Simple_Prototype(class_num,out_dim, device).to(device)
    
    if method=='b':
        D = mnist_svhn.Discriminator(class_num,bn).to(device)
    elif method == 'a':
        D = mnist_svhn.Discriminator(out_dim,bn).to(device)
    
    # prepare optimizer
    #g_optim = optim.Adam(list(Gt.parameters())+list(pt.parameters()),lr=g_lr,betas=(0.5,0.999))
    g_optim = optim.Adam(list(Gt.parameters()),lr=g_lr,betas=(0.5,0.999))
    d_optim = optim.Adam(D.parameters(),lr=d_lr,betas=(0.5,0.999))

    #build model and train
    model = adapt_model.ProtoDA(device)
    print("beta:",beta,"method:",method)
    model.step2(Gs,ps,Gt,D,gan_style,
                uni_dataloader,te_t_dataloader, 
                g_optim,d_optim,checkpoint_dir,ad_epochs,
                bn,G_iter,D_iter,beta,method=method,train=train)


if __name__ == "__main__":
    import random
    # Set random seem for reproducibility
    manualSeed = 128
    # manualSeed = random.randint(1,10000)
    print("Random Seed: ",manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    out_dim = 10
    step1(batch_size=64,train=True,c_epochs=40,bn=True,out_dim=out_dim)
    beta=0.1
    method='b'
    #step2(g_lr=0.0001,out_dim=out_dim,d_lr=0.0001,ad_epochs=10000,beta=beta,method=method,test_step=30,G_iter=1)
    #normal_target_only(train=False)
    #normal_source_only(train=False)
    #cpl_target_only(True)
    #adda_adapt_model(train=True)