import torch
import torch.optim as optim
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from dataset import unit_data,svhn,syndigits
from networks import prototype, mnist_svhn
import adapt_model_v2 as am


path_synth = r'/media/lenovo/系统/zw/20181207/prototype/data/synthdigits'
path_svhn = r'/media/lenovo/系统/zw/20181207/prototype/data/svhn'
def normal_target_only(train=False,
                       checkpoint_dir='./checkpoint/SYNTHDIGITS2SVHN/bn-normal-target-only',
                       batch_size=64,c_lr=0.001,c_epochs=40,bn=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    solvers.normal_target_only(exp="synthdigits2svhn",
                        checkpoint_dir=checkpoint_dir,
                        device=device,train=train,batch_size=batch_size,
                        c_lr=c_lr,c_epochs=c_epochs,bn=bn)

def normal_source_only(train=False,
                      checkpoint_dir='./checkpoint/SYNTHDIGITS2SVHN/bn-normal-source-only',
                      batch_size=64,c_lr=0.001,c_epochs=40,bn=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    solvers.normal_source_only(train,"synthdigits2svhn",
                               checkpoint_dir,device,
                               batch_size,c_lr,c_epochs,bn)

def cpl_target_only(train=False,
                    checkpoint_dir='./checkpoint/SYNTHDIGITS2SVHN/bn-cpl-target-only',
                    batch_size=64,c_lr=0.001,c_epochs=20,bn=True,gamma=1.0,lamb=0.01,out_dim=10):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    checkpoint_dir = checkpoint_dir+"-all-out-"+str(out_dim)
    solvers.cpl_target_only(train,"synthdigits2svhn",checkpoint_dir,device,
                            batch_size,out_dim,c_lr,c_epochs,bn,gamma=1.0,lamb=0.01)
                            
def step1(batch_size=64,out_dim=10, path_source=None, path_target=None,
         c_lr=0.01,c_epochs=20,
         checkpoint_dir='./checkpoint/SYNTHDIGITS2SVHN/bn-all-out-',lamb=0.01,gamma=1.0,
         use_all=True,bn=True,train=True, device='cuda:0'):

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
    s_tr,s_te = syndigits.get_train_and_test(batch_size,path=path_source,
                                             image_size=image_size,gray=False)
    # usps dataset 
    t_te = svhn.get_svhn(False,batch_size,path=path_target,image_size=image_size,gray=False)

    # build network model
    Gs = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim,device).to(device)
    prototype.proto_init(ps,Gs,s_tr,device)

    # prepare optimizer
    c_optim = optim.SGD(list(Gs.parameters())+list(ps.parameters()),lr=c_lr,momentum=0.9,weight_decay=0.0005)

    # train source networks
    model = am.ProtoDA(device = device,bn=bn,gamma=gamma)
    model.step1(Gs,ps,s_tr,s_te,t_te,
                c_optim,c_epochs,checkpoint_dir,lamb,train)


def step2(batch_size=64,out_dim=10, path_source=None, path_target=None,
          g_lr=0.0001,d_lr=0.0001,ad_epochs=2000,
          checkpoint_dir='./checkpoint/SYNTHDIGITS2SVHN/bn-all-out-',
          bn=True,G_iter=1,D_iter=1,
          beta=0.01,method='b',train=True):
    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")
    image_size=32
    nc =3 # gray image
    class_num = 10 # the size of label
    checkpoint_dir = checkpoint_dir+str(out_dim)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    tr_s_dataloader = syndigits.get_syndigits(True,batch_size,path=path_source,image_size=image_size,gray=False)
    tr_t_dataloader,te_t_dataloader = svhn.get_train_and_test(batch_size,path=path_target,image_size=image_size,gray=False)
    uni_dataloader = unit_data.unit_data_by_loader(tr_s_dataloader,tr_t_dataloader,ad_epochs)

    # build model
    Gs = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim,device).to(device)
    Gt = mnist_svhn.CNN_Extract(nc,out_dim,bn).to(device)
    
    if method == 'a':
        D = mnist_svhn.Discriminator(out_dim,bn).to(device)
    else:
        D = mnist_svhn.Discriminator(class_num,bn).to(device)
    
    # prepare optimizer
    # g_optim = optim.Adam(list(Gt.parameters()),lr=g_lr,betas=(0.5,0.999))
    # d_optim = optim.Adam(D.parameters(),lr=d_lr,betas=(0.5,0.999))
    g_optim = optim.SGD(Gt.parameters(), lr=g_lr, momentum=0.9, weight_decay=5e-4)
    d_optim = optim.SGD(D.parameters(), lr=d_lr, momentum=0.9, weight_decay=5e-4)

    #build model and train
    model = am.ProtoDA(device,bn=bn, show_step=300)
    model.step2(Gs,ps,Gt,D,uni_dataloader,te_t_dataloader,
                g_optim,d_optim,checkpoint_dir,
                ad_epochs, G_iter,D_iter,beta,method=method,train=train)


if __name__ == "__main__":
    bn = True
    checkpoint_dir='./checkpoint/SYNTHDIGITS2SVHN/all-out-'
    method = 'b'
    beta = 1.0
    out_dim = 10
    lamb = 1.0
    step1(out_dim=out_dim, train=True, c_lr=0.001, batch_size=64,lamb=lamb,
          c_epochs=20, bn=bn, checkpoint_dir=checkpoint_dir)
    step2(out_dim=out_dim, ad_epochs=40000, train=True, method='a', g_lr=1e-6,
          d_lr=1e-6, beta=beta, bn=bn, G_iter=1, checkpoint_dir=checkpoint_dir)
    step2(out_dim=out_dim, ad_epochs=40000, train=True, method='b', g_lr=1e-6,
          d_lr=1e-6, beta=beta, bn=bn, G_iter=1, checkpoint_dir=checkpoint_dir)