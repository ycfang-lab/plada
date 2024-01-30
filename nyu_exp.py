from networks.vggnet import VGGNet
from networks import prototype
from networks import discriminator
import torch
import torch.nn as nn  
import torch.optim as optim
import os 
import matplotlib.pyplot as plt
import utils
import adapt_model_v2 as adapt_model
from dataset import nyu_dataset, unit_data


def root_dir_choose(platform='2070'):
    if platform=='2070':
        root_dir=r'/media/antec/data/zw/data'
    elif platform=='p5000':
        root_dir=r'/media/lenovo/data/zw/data'
    elif platform == 'mylaptop':
        root_dir = r"E:\Mycode\Domain Adaptation\data"
    elif platform == '2070_1':
        root_dir =  r"/media/antec/data/zw/data"
    return root_dir


def nyu_loader(batch_size, root_dir):
    root_path = os.path.join(root_dir, 'nyu', 'NYUD_multimodal')
    s_tr = nyu_dataset.get_nyu(True, 'images', root_path, batch_size)
    s_te = nyu_dataset.get_nyu(False, 'images', root_path, batch_size)
    t_tr = nyu_dataset.get_nyu(True, 'depths', root_path, batch_size)
    t_te = nyu_dataset.get_nyu(False, 'depths', root_path, batch_size)
    return s_tr, s_te, t_tr, t_te


def step1(checkpoint_dir="./checkpoint/plada/nyu/", 
          batch_size=32, out_dim=19, c_lr=1e-3, c_epoch=20,
          lamb=1e-3, gamma=1.0, train=True, angle_loss=None, device='cpu',
          root_dir=".", net='vgg16'):
    class_num = 19
    # checkpoint_dir = checkpoint_dir + net + "out"+ str(out_dim)
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))
    s_tr, s_te, t_tr, t_te = nyu_loader(batch_size, root_dir)

    if net == 'vgg16':
        Gs = VGGNet(True, 19, bn=True).to(device)
    elif net == 'res50':
        raise NotImplementedError
    else:
        raise ValueError("don't support this networks bone")

    ps = prototype.Simple_Prototype(class_num, out_dim, device).to(device).to(device)
    prototype.proto_init(ps, Gs, s_tr, device)

    c_optim = optim.SGD([{"params": Gs.parameters(), "lr": c_lr},
                        {"params": ps.parameters(), "lr": c_lr}],
                        lr=c_lr, momentum=0.9, weight_decay=1e-5)

    model = adapt_model.ProtoDA(device=device, gamma=gamma, show_step=10)

    model.step1(Gs, ps, s_tr, s_te, t_te,
                c_optim, c_epoch, checkpoint_dir,
                lamb, train, angle_loss=angle_loss)


def step2(root_dir='.', device='cpu', batch_size=32, out_dim=19, 
          g_lr=1e-4, d_lr=1e-4, 
          ad_epochs=10000, checkpoint_dir='./checkpoint/plada/nyu/',
          beta=1e-2, method='b', train=True, net='vgg16', multicuda=True):
    class_num = 19

    # if torch.cuda.device_count() > 1:
    #     print("use device count:", torch.cuda.device_count())
    #     gpus = device.split(',')
    #     net = nn.DataParallel(net, device_ids=[int(i) for i in gpus])
    # net.cuda()

    # device = torch.device(device if (torch.cuda.is_available()) else 'cpu')

    # checkpoint_dir = checkpoint_dir + net + "out"+ str(out_dim)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("checkpoint_dir:", os.path.realpath(checkpoint_dir))

    s_tr, s_te, t_tr, t_te = nyu_loader(batch_size, root_dir)

    ad_data_tr = unit_data.unit_data_by_loader(s_tr, t_tr, ad_epochs)

    if net == 'vgg16':
        Gs = VGGNet(True, 19, bn=True)
        Gt = VGGNet(True, 19, bn=True)
    elif net == 'res50':
        raise NotImplementedError
    else:
        raise ValueError("don't support this networks bone")
    
    p = prototype.Simple_Prototype(class_num, out_dim, device='cuda:0')
    
    if method == 'a':
        D = discriminator.Discriminator(out_dim, 1024, 1024, 1024, bn=False, sn=True)
        # D = discriminator.Discriminator(out_dim, 100, 100, 100, bn=True).to(device)
        # D = discriminator.Discriminator_(out_dim, 100).to(device)

    elif method == 'b':
        D = discriminator.Discriminator(class_num, 1024, 1024, 1024, bn=False, sn=True)
        # D = discriminator.Discriminator(out_dim, 100, 100, 100, bn=True).to(device)
        # D = discriminator.Discriminator_(out_dim, 100).to(device)

    else:
        raise ValueError('Something gets error in methods')

    if torch.cuda.device_count() > 1:
        print("use device count:", torch.cuda.device_count())
        gpus = device.split(',')
        Gs = nn.DataParallel(Gs, device_ids=[int(i) for i in gpus])
        Gt = nn.DataParallel(Gt, device_ids=[int(i) for i in gpus])
        p = nn.DataParallel(p, device_ids=[int(i) for i in gpus])
        D = nn.DataParallel(D, device_ids=[int(i) for i in gpus])
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else 'cpu')
    Gs.cuda()
    Gt.cuda()
    p.cuda()
    D.cuda()

    g_optim = optim.SGD(Gt.parameters(), lr=g_lr, momentum=0.9)
    d_optim = optim.SGD(D.parameters(), lr=d_lr, momentum=0.9)

    print(device)
    model = adapt_model.ProtoDA(device, bn=True)

    model.step2(Gs, p, Gt, D, ad_data_tr, t_te, g_optim, d_optim,
                checkpoint_dir, ad_epochs, beta=beta, method=method,
                train=train, g_lr=g_lr, d_lr=d_lr, lr_sche=True, multicuda=multicuda, is_nyud=True)

def vgg16c2d(platform, device, method='a', step='1', net='vgg16', beta=0.01):
    checkpoint_dir = "./checkpoint/nyu2/{}/out-19/".format(net)
    root_dir = root_dir_choose(platform)
    if step == "1":
        device = "cuda:{}".format(device)
        step1(checkpoint_dir, 32, 19, 1e-3, c_epoch=20, lamb=1e-2,
                train=True, angle_loss=False, device=device, 
                root_dir=root_dir)
    else:
        step2(root_dir, device, 32, 19, g_lr=1e-5,
                d_lr=1e-5, ad_epochs=100000, checkpoint_dir=checkpoint_dir,
                beta=beta, method=method, train=True, net=net, multicuda=True)


def vgg16c2d_eval(platform, device, method, net):
    checkpoint_dir = "./checkpoint/nyu2/{}/out-19/".format(net)
    root_dir = root_dir_choose(platform)
    step2(root_dir,device, 32, 19,
         checkpoint_dir=checkpoint_dir, method=method,train=False,net=net, multicuda=True)

if __name__ == "__main__":
    # vgg16c2d('2070_1', '0,1', 'b', step='2', beta=0.001)

    vgg16c2d_eval('2070_1', '0,1', 'a', 'vgg16')

