import torch
import torch.optim as optim
import os
import sys
from dataset import unit_data, usps, mnist
import matplotlib.pyplot as plt
import utils
from networks import prototype,mnist_usps
from networks import discriminator as disc
import adapt_model_v2 as adapt_model
import solvers


def adda_adapt_model(train=False, checkpoint_dir='./checkpoint/MNIST2USPS/bn-normal-source-only',
                    batch_size=32,g_lr=0.00001,d_lr=0.00001,ad_epochs=2000,
                    bn=True,G_iter=1,D_iter=1,test_step=30,use_all=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    solvers.adda_method(train, 'mnist2usps', checkpoint_dir, device, batch_size,
                        g_lr, d_lr, ad_epochs, bn,
                        G_iter, D_iter, test_step, use_all)


def normal_target_only(train=False,
                      checkpoint_dir='./checkpoint/MNIST2USPS/bn-normal-target-only',
                      batch_size=64,c_lr=0.001,c_epochs=40,bn=True,use_all=True):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        solvers.normal_target_only(exp='mnist2usps',
                                   checkpoint_dir=checkpoint_dir,
                                   device=device, train=train, batch_size=batch_size,
                                   c_lr=c_lr, c_epochs=c_epochs, bn=bn, use_all=use_all)


def normal_source_only(train=False,
                      checkpoint_dir='./checkpoint/MNIST2USPS/bn-normal-source-only',
                      batch_size=64,c_lr=0.001,c_epochs=40,bn=True,use_all=True):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        solvers.normal_source_only(train, 'mnist2usps', checkpoint_dir,
                                   device, batch_size, c_lr, c_epochs, bn, use_all)


def cpl_target_only(train=False,
                    checkpoint_dir='./checkpoint/MNIST2USPS/bn-cpl-target-only',
                    batch_size=64,c_lr=0.001,c_epochs=20,bn=True,
                    gamma=1.0,lamb=0.01,out_dim=10,use_all=True):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        checkpoint_dir = checkpoint_dir+"-all-out-"+str(out_dim)
        solvers.cpl_target_only(train, 'mnist2usps', checkpoint_dir, device,
                                batch_size, out_dim, c_lr, c_epochs,
                                bn, use_all, gamma, lamb)

def proto_analysis(bn,checkpoint_dir='./checkpoint/MNIST2USPS/bn-all-out-10'):
    
    image_size=28
    class_num=10
    out_dim=10
    nc=1
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    Gs = mnist_usps.CNN_Extract(nc,out_dim,bn=bn).to(device)
    Gt = mnist_usps.CNN_Extract(nc,out_dim,bn=bn).to(device)
    Gs.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_Gs.pt")))
    Gt.load_state_dict(torch.load(os.path.join(checkpoint_dir,'Gt.pt')))
    
    ps = prototype.Simple_Prototype(class_num,out_dim).to(device)
    ps.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source_best_ps.pt")))
    
    s_data_te = mnist.get_mnist(False,batch_size=-1,use_all=True,image_size=image_size,gray=True)
    t_data_te = usps.get_usps(False,batch_size=-1,use_all=True,image_size=image_size)

    img_s,label_s = next(iter(s_data_te))
    img_t,label_t = next(iter(t_data_te))
    img_s,img_t=img_s.to(device),img_t.to(device)
    label_s,label_t = label_s.detach().numpy(),label_t.detach().numpy()
    
    feat_gs = Gs(img_s).to('cpu').detach().numpy()
    feat_gt = Gt(img_t).to('cpu').detach().numpy()

    proto_s = ps.proto.detach().to('cpu').numpy()


    #fig_1 = plt.figure("toy")
    #utils.plot_tsne_toy(fig_1,proto_s,feat_gs,label_s)

    #fig0 = plt.figure("source distribution")
    #utils.plot_tsne_one_domain(fig0,True,proto_s,feat_gs,label_s,name="source distribution")
    
    fig2 = plt.figure("convolution feature")
    utils.plot_tsne_da(fig2,True,proto_s,feat_gs,label_s,feat_gt,label_t,1000,name="convolution feature mnist2usps")
    plt.show()


def step1(batch_size=64, out_dim=10, c_lr=0.01, c_epochs=30, path_source=None, path_target=None,
          checkpoint_dir='./checkpoint/MNIST2USPS/bn-all-out-', lamb=0.01, gamma=1.0,
          use_all=True, bn=False, train=True, device="cuda:0"):

    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")
    # some defined hyperparameter
    image_size = 28
    nc = 1  # gray image
    class_num = 10  # the size of label
    checkpoint_dir = checkpoint_dir + str(out_dim)
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # prepare data
    # mnist dataset
    s_tr, s_te = mnist.get_train_and_test(batch_size, use_all, None, path=path_source,
                                          image_size=image_size, gray=True, aug=False)

    # usps dataset
    t_te = usps.get_usps(False, batch_size, use_all, None, path=path_target,
                         image_size=image_size)

    # build network model
    Gs = mnist_usps.CNN_Extract(nc, out_dim, bn).to(device)
    ps = prototype.Simple_Prototype(class_num, out_dim, device).to(device)

    # prepare optimizer
    c_optim = optim.SGD(list(Gs.parameters()) + list(ps.parameters()),
                        lr=c_lr, momentum=0.9, weight_decay=0.0005)

    # train source networks
    model = adapt_model.ProtoDA(device=device, gamma=gamma, bn=bn)
    model.step1(Gs, ps, s_tr, s_te, t_te,
                c_optim, c_epochs, checkpoint_dir, lamb, train)


def step2(batch_size=64, out_dim=10, path_source=None, path_target=None,
          g_lr=0.0001, d_lr=0.0001, ad_epochs=2000,
          checkpoint_dir='./checkpoint/MNIST2USPS/bn-all-out-',
          use_all=True, bn=True, G_iter=1, D_iter=1,
          beta=0.01, method='b', train=True, show_step=100, device="cuda:0"):
    if path_source is None or  path_target is None:
        raise ValueError("Please assign valid path of source /target data")
    # hyperparameter
    image_size = 28
    nc = 1
    class_num = 10
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")
    checkpoint_dir = checkpoint_dir + str(out_dim)

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("checkpoint_dir:", os.path.realpath(checkpoint_dir))

    # prepare data
    s_tr, s_te = mnist.get_train_and_test(batch_size, use_all, None, path=path_source,
                                          image_size=image_size, gray=True, aug=False)
    t_tr, t_te = usps.get_train_and_test(batch_size, use_all, None, path=path_target,
                                         image_size=image_size)
    ad_data = unit_data.unit_data_by_loader(s_tr, t_tr, ad_epochs)
    # create graph
    Gs = mnist_usps.CNN_Extract(nc, out_dim, bn).to(device)
    Gt = mnist_usps.CNN_Extract(nc, out_dim, bn).to(device)
    ps = prototype.Simple_Prototype(class_num, out_dim,device).to(device)

    if method == 'b':
        # D = mnist_usps.Discriminator(class_num, bn).to(device)
        D = disc.Discriminator(class_num, 256, 128, 64, True).to(device)
        # D = disc.Discriminator(class_num, 256, 128, False, True).to(device)
    else:
        # D = mnist_usps.Discriminator(out_dim, bn).to(device)
        D = disc.Discriminator(out_dim, 256, 128, 64, True).to(device)
        # D = disc.Discriminator(out_dim, 256, 128, False, True).to(device)

    g_optim = optim.Adam(Gt.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_optim = optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))
    # g_optim = optim.SGD(D.parameters(), lr=g_lr, momentum=0.9, weight_decay=5e-4)
    # d_optim = optim.SGD(D.parameters(), lr=d_lr, momentum=0.9, weight_decay=5e-4)
    
    model = adapt_model.ProtoDA(device=device, bn=bn, show_step=show_step)
    model.step2(Gs, ps, Gt, D, ad_data, t_te,
                g_optim, d_optim, checkpoint_dir,
                ad_epochs, G_iter, D_iter, beta,
                method=method, train=train, need_dce=False)


if __name__ == "__main__":
    import random
    manualSeed = 128
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    path_mnist=r'/media/antec/data/zw/data/mnist'
    path_usps = r'/media/antec/data/zw/data/usps/usps.h5'

    checkpoint_dir = './checkpoint/MNIST2USPS/bn-all-out-'
    
    step1(train=True, c_lr=0.01, lamb=0.01, path_source=path_mnist, path_target=path_usps,
          out_dim=10, c_epochs=20, bn=True,
          checkpoint_dir=checkpoint_dir, device='cuda:0')
    
    method = 'a'
    beta = 0.01
    step2(method='a', ad_epochs=20000, path_source=path_mnist, path_target=path_usps,
          g_lr=0.0001, d_lr=0.0001,
          beta=beta, out_dim=10, bn=True, checkpoint_dir=checkpoint_dir, device="cuda:0")
