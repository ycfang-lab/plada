import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
from dataset import unit_data,mnist_m,mnist
import utils
from networks import prototype, mnist_mnist_m, discriminator
#import adapt_model
import adapt_model_v2 as adapt_model
# import adda_model
import solvers


def normal_target_only(train=False,
                       checkpoint_dir='./checkpoint/MNIST2MNIST_M/bn-normal-target-only',
                       batch_size=64,c_lr=0.001,c_epochs=40,bn=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    solvers.normal_target_only(exp="mnist2mnist_m",
                               checkpoint_dir=checkpoint_dir,
                               device=device, train=train, batch_size=batch_size,
                               c_lr=c_lr, c_epochs=c_epochs, bn=bn)


def normal_source_only(train=False,
                      checkpoint_dir='./checkpoint/MNIST2MNIST_M/bn-normal-source-only',
                      batch_size=64,c_lr=0.001,c_epochs=40,bn=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    solvers.normal_source_only(train, "mnist2mnist_m",
                               checkpoint_dir, device,
                               batch_size, c_lr, c_epochs, bn)


def cpl_target_only(train=False,
                    checkpoint_dir='./checkpoint/MNIST2MNIST_M/bn-cpl-target-only',
                    batch_size=64,c_lr=0.001,c_epochs=20,bn=True,gamma=1.0,lamb=0.01,out_dim=10):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    checkpoint_dir = checkpoint_dir+"-all-out-"+str(out_dim)
    solvers.cpl_target_only(train, "mnist2mnist_m", checkpoint_dir, device,
                            batch_size, out_dim, c_lr, c_epochs, bn, gamma=1.0, lamb=0.01)


# def adda_step1(batch_size,c_epoch,checkpoint_dir,c_lr):
#     # some defined hyperparameter
#     image_size=32
#     nc = 3 # gray image
#     class_num = 10 # the size of label
#     use_all = True
#     checkpoint_dir = checkpoint_dir
#     #device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#     device='cpu'
#     bn = True
#     # check checkpoint dir
#     if not os.path.exists(checkpoint_dir):
#             os.makedirs(checkpoint_dir)
#     print(os.path.realpath(checkpoint_dir))

#     # prepare data
#     # mnist dataset
#     s_tr,s_te = mnist.get_train_and_test(batch_size,use_all,
#                                         path = r"E:\Mycode\CVPR2019\prototype\data\mnist",
#                                         image_size=image_size,gray=False,aug=True)
#     # usps dataset 
#     t_te = mnist_m.get_mnist_m(False,batch_size,path=r'E:\Mycode\CVPR2019\prototype\data\mnist_m',image_size=image_size)

#     # build network model
#     Gs = mnist_mnist_m.CNN_Extract(nc,class_num,bn).to(device)

#     # prepare optimizer
#     c_optim = optim.SGD(list(Gs.parameters()),lr=c_lr,momentum=0.9,weight_decay=0.0005)


#     # train source networks
#     model = adda_model.ADDA(device = device)
#     model.step1(s_tr,s_te,t_te,Gs,c_optim,True,bn,c_epoch,checkpoint_dir,show_step=3)


# def adda_step2(batch_size,checkpoint_dir,ad_epoch):
#     # hyperparameter
#     image_size=32
#     nc = 3
#     bn=True
#     class_num=10
#     #device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#     device = 'cpu'
#     checkpoint_dir = checkpoint_dir

#     # check checkpoint dir
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     print("checkpoint_dir:",os.path.realpath(checkpoint_dir))

#     # prepare data
#     tr_s_dataloader = mnist.get_mnist(True,batch_size,use_all,image_size=image_size,gray=False,aug=True)
#     tr_t_dataloader,te_t_dataloader = mnist_m.get_train_and_test(batch_size,image_size=image_size)
#     ad_data_tr = unit_data.unit_data_by_loader(tr_s_dataloader,tr_t_dataloader,ad_epochs)

#     # create graph
#     Gs = mnist_mnist_m.CNN_Extract(nc,class_num,bn).to(device)
#     Gt = mnist_mnist_m.CNN_Extract(nc,class_num,bn).to(device)
    
#     D = mnist_mnist_m.Discriminator(class_num,bn).to(device)
    
#     g_optim = optim.Adam(Gt.parameters(),lr=g_lr,betas=(0.5,0.999))
#     d_optim = optim.Adam(D.parameters(),lr=d_lr,betas=(0.5,0.999))

#     model = adda_model.ADDA(device=device)
#     model.step2(Gs,Gt,D,ad_data_tr,te_t_dataloader,g_optim,d_optim,
#                  checkpoint_dir,ad_epochs,bn,test_step=200)


def plot_tsne_normal():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    s_te = mnist.get_mnist(False,500,True,image_size=32,gray=False,aug=True)
    t_te = mnist_m.get_mnist_m(False,500,image_size=32)

    Gs = mnist_mnist_m.CNN_Extract(3,10,True).to(device)
    Gt = mnist_mnist_m.CNN_Extract(3,10,True).to(device)
    Gs.load_state_dict(torch.load("./checkpoint/MNIST2MNIST_M/adda/normal-/source-best-Gs.pt"))
    Gt.load_state_dict(torch.load("./checkpoint/MNIST2MNIST_M/adda/normal-/target-best-Gt.pt"))

    Gs.eval()
    Gt.eval()

    s_image,s_label = next(iter(s_te))
    t_image,t_label = next(iter(t_te))
    s_image,s_label, t_image, t_label = s_image.to(device),s_label.to(device),t_image.to(device),t_label.to(device)

    source_feat = Gs(s_image)
    target_feat = Gt(t_image)
    fig = plt.figure()
    utils.plot_tsne_da_normal(fig,source_feat,s_label,target_feat,t_label,
                              samples=300,name="adda_normal",device=device)

def tsne(out_dim=10, checkpoint_dir='./checkpoint/MNIST2MNIST_M/bn-all-out-'):

    image_size = 32
    nc = 3 # gray image
    class_num = 10 # the size of label
    checkpoint_dir = checkpoint_dir+str(out_dim)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        raise ValueError("don't exist this folder")
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    # mnist dataset
    
    s_te = mnist.get_mnist(False,batch_size=1000,path=path_mnist,
                           image_size=image_size,gray=False)
    t_te = mnist_m.get_mnist_m(False,batch_size=1000,
                               path=path_mnist_m,image_size=image_size)

    # build network model
    # Gs = office_network_2.ResNetFC(out_dim, 'resnet50', True, False, new_cls=True).to(device)
    #Gs = mnist_mnist_m.CNN_Extract(nc,out_dim,bn).to(device)
    ps = prototype.Simple_Prototype(class_num,out_dim,device).to(device)
    Gs.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_Gs.pt")))
    ps.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_ps.pt")))

    fig = plt.figure()
    s_img,s_label = next(iter(s_te))
    s_img = s_img.to(device)
    t_img,t_label = next(iter(t_te))
    t_img= t_img.to(device)

    source_feat = Gs(s_img).detach().cpu().numpy()
    s_label = s_label.numpy()
    target_feat = Gs(t_img).detach().cpu().numpy()
    t_label = t_label.numpy()

    proto = ps.proto.detach().cpu().numpy()
    utils.plot_tsne_da(fig, True, proto, source_feat, s_label,
                       target_feat, t_label, name="source only")


def step1(batch_size=64, out_dim=10,
          path_source=None, path_target=None, 
          c_lr=0.01,c_epochs=40,
         checkpoint_dir='./checkpoint/MNIST2MNIST_M/bn-all-out-', 
         lamb=0.01,gamma=1.0, use_all=True, bn=True, train=True, device='cuda:0'):
    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")

    # some defined hyperparameter
    image_size = 32
    nc = 3
    class_num = 10
    checkpoint_dir = checkpoint_dir+str(out_dim)
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    # mnist dataset
    s_tr, s_te = mnist.get_train_and_test(batch_size, use_all, path=path_source,
                                          image_size=image_size,
                                          gray=False, aug=True)
    t_te = mnist_m.get_mnist_m(False, batch_size, path=path_target,
                               image_size=image_size)

    # build network model
    # Gs = office_network_2.ResNetFC(out_dim, 'resnet50', True, False, new_cls=True).to(device)

    # Gs = LeNet.CNN_Extract_(nc, out_dim, True, 32).to(device)
    # Gs = resnet.ResNetFc("ResNet50", False, new_cls=True, class_num=out_dim).to(device)
    Gs = mnist_mnist_m.CNN_Extract(nc, out_dim, bn).to(device)
    ps = prototype.Simple_Prototype(class_num, out_dim, device).to(device)
    prototype.proto_init(ps, Gs, s_tr, device)

    # prepare optimizer
    c_optim = optim.SGD(list(Gs.parameters())+list(ps.parameters()),
                        lr=c_lr, momentum=0.9, weight_decay=5e-4)

    # c_optim = optim.Adam(list(Gs.parameters())+list(ps.parameters()),
    #                      lr=c_lr, betas=(0.9, 0.999), weight_decay=5e-4)
    # train source networks
    model = adapt_model.ProtoDA(device=device, bn=bn, gamma=gamma)
    model.step1(Gs, ps, s_tr, s_te, t_te,
                c_optim, c_epochs,
                checkpoint_dir, lamb, train)

def step2(batch_size=32, out_dim=10, path_source=None, path_target=None,
          g_lr=0.0001, d_lr=0.0001, ad_epochs=10000,
          checkpoint_dir='./checkpoint/MNIST2MNIST_M/bn-all-out-',
          use_all=True, bn=True, G_iter=1, D_iter=1,
          beta=0.01, method='b', train=False, show_step=100, device='cuda:0'):

    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")

    # hyperparameter
    image_size = 32
    nc = 3
    class_num = 10
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")
    checkpoint_dir = checkpoint_dir+str(out_dim)

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("checkpoint_dir:", os.path.realpath(checkpoint_dir))

    # prepare data
    s_tr = mnist.get_mnist(True, batch_size, use_all, None, path_source,
                           image_size, False, True)
    t_tr, t_te = mnist_m.get_train_and_test(batch_size, None, path_target,
                                            image_size=image_size)

    ad_data_tr = unit_data.unit_data_by_loader(s_tr, t_tr, ad_epochs)

    # create graph
    Gs = mnist_mnist_m.CNN_Extract(nc,out_dim,bn).to(device)
    Gt = mnist_mnist_m.CNN_Extract(nc,out_dim,bn).to(device)

    ps = prototype.Simple_Prototype(class_num, out_dim, device=device,gamma=1.0).to(device)

    if method == 'a':
        D = mnist_mnist_m.Discriminator(out_dim, bn).to(device)
        # D = discriminator.Discriminator(out_dim, 256, 128, 64, True).to(device)
    else:
        D = mnist_mnist_m.Discriminator(class_num, bn).to(device)
        # D = discriminator.Discriminator(class_num, 256, 128, 64, True).to(device)

    g_optim = optim.Adam(Gt.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_optim = optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

    # g_optim = optim.SGD(Gt.parameters(), lr=g_lr, momentum=0.9)
    # d_optim = optim.SGD(D.parameters(), lr=d_lr, momentum=0.9)

    model = adapt_model.ProtoDA(device=device, bn=bn, show_step=show_step)
    # print("beta:", beta, "method:", method)
    model.step2(Gs, ps, Gt, D, ad_data_tr, t_te,
                g_optim, d_optim, checkpoint_dir,
                ad_epochs, G_iter, D_iter, beta,
                method=method, train=train, need_dce=False)
