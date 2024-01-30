from networks import prototype, resnet, discriminator
import torch.optim as optim
from dataset import Imageclef, unit_data
import os
import matplotlib.pyplot as plt
import utils
import adapt_model_v2 as adapt_model
# import train_models as adapt_model
import torch

def step1(source, target,
          checkpoint_dir="./checkpoint/I2P/all-out-",
          batch_size=32, out_dim=10, c_lr=1e-3, c_epoch=20,
          lamb=1e-2, gamma=1.0, train=True):
    class_num=12
    checkpoint_dir = checkpoint_dir + str(out_dim)

    device = torch.device(device_this_program if (torch.cuda.is_available()) else "cpu")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    s_tr = Imageclef.get_clef(True, source, path, batch_size)
    s_te = Imageclef.get_clef(False, source, path, batch_size)
    t_te = Imageclef.get_clef(False, target, path, batch_size)

    Gs = resnet.ResNetFc('ResNet50', use_bottleneck=False, new_cls=True, class_num=out_dim).to(device)

    ps = prototype.Simple_Prototype(class_num, out_dim, device).to(device)
    prototype.proto_init(ps, Gs, s_tr, device)

    # optimizer
    c_optim = optim.SGD([{"params": Gs.feature_layers.parameters(), "lr": c_lr},
                         {"params": Gs.fc.parameters(), "lr": 10*c_lr},
                         {"params": ps.parameters(), "lr": 10*c_lr}],
                        lr=c_lr, momentum=0.9, weight_decay=1e-5)
    model = adapt_model.ProtoDA(device=device, gamma=gamma, show_step=10)
    model.step1(Gs, ps, s_tr, s_te, t_te,
                c_optim, c_epoch, checkpoint_dir,
                lamb, train)


def step2(source, target, batch_size=32, out_dim=10,
          g_lr=1e-4, d_lr=1e-4, ad_epochs=10000,
          checkpoint_dir="./checkpoint/I2P/all-out-",
          beta=1e-2, method='a', train=True, d_iter=5, dce_train=False):
    class_num=12
    device = torch.device(device_this_program if (torch.cuda.is_available()) else 'cpu')
    checkpoint_dir = checkpoint_dir + str(out_dim)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("checkpoint_dir:", os.path.realpath(checkpoint_dir))
    s_tr = Imageclef.get_clef(True, source, path, batch_size)
    t_tr = Imageclef.get_clef(True, target, path, batch_size)
    t_te = Imageclef.get_clef(False, target, path, batch_size)

    ad_data_tr = unit_data.unit_data_by_loader(s_tr, t_tr, ad_epochs)

    # create networks
    Gs = resnet.ResNetFc('ResNet50', use_bottleneck=False, new_cls=True, class_num=out_dim).to(device)
    Gt = resnet.ResNetFc('ResNet50', use_bottleneck=False, new_cls=True, class_num=out_dim).to(device)
    p = prototype.Simple_Prototype(class_num, out_dim, device).to(device)

    if method == 'a':
        D = discriminator.Discriminator(out_dim, 3072, 2048, 1024, bn=False, sn=True, wgan=False).to(device)
        # D = discriminator.Discriminator__(out_dim, 100).to(device)
    elif method == 'b':
        D = discriminator.Discriminator(class_num, 3072, 2048, 1024, bn=False, sn=True, wgan=False).to(device)
        # D = discriminator.Discriminator__(class_num, 100).to(device)
    else:
        raise ValueError('Something gets error in methods')

    g_optim = optim.SGD(Gt.parameters(), lr=g_lr, momentum=0.9)
    d_optim = optim.SGD(D.parameters(), lr=d_lr, momentum=0.9)
    # g_optim = optim.Adam(Gt.parameters(), lr=g_lr, betas=(0.5, 0.999))
    # d_optim = optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

    model = adapt_model.ProtoDA(device=device, bn=True, show_step=300)
    model.step2(Gs, p, Gt, D, ad_data_tr, t_te,
                g_optim, d_optim, checkpoint_dir, ad_epochs,
                beta=beta, method=method, train=train, D_iter=d_iter, need_dce=dce_train)


if __name__ == "__main__":
    # path = '/media/rtx/DA18EBFA09C1B27D/zw/data/image_CLEF'
    path = '/media/lenovo/系统/zw/DomainAdaptation/data/image_CLEF'
    # path = '/media/antec/TOSHIBA EXT/domainadaptation/data/image_CLEF'
    lamb = 1.0
    checkpoint_dir = "./checkpoint/I2P/all-out-"
    device_this_program = "cuda:0"
    step1('i', 'p', train=True, c_lr=1e-3, out_dim=12, c_epoch=30, lamb=lamb, checkpoint_dir=checkpoint_dir)
    beta = 0.01
    step2('i', 'p', out_dim=12, g_lr=1e-4, d_lr=1e-4, method='b',
          beta=beta, ad_epochs=30000, checkpoint_dir=checkpoint_dir)

    # checkpoint_dir = "./checkpoint/P2I/all-out-"
    # device_this_program = "cuda:0"
    # lamb = 1.0
    # step1('p', 'i', c_lr=1e-3, out_dim=12, c_epoch=60, lamb=lamb,
    #       checkpoint_dir=checkpoint_dir, train=True)
    # beta = 1.0
    # step2('p', 'i', out_dim=12, g_lr=1e-6, d_lr=1e-6, method='b',
    #       beta=beta, ad_epochs=60000, checkpoint_dir=checkpoint_dir, d_iter=1, dce_train=False)

    
    # lamb = 1.0
    # device_this_program = 'cuda:1'
    # checkpoint_dir = "./checkpoint/I2C/all-out-"
    # step1('i','c', train=False, c_lr=1e-3, out_dim=12, c_epoch=60, lamb=lamb, checkpoint_dir=checkpoint_dir)
    # beta = 1.0
    # step2('i', 'c', out_dim=12, g_lr=1e-6, d_lr=1e-6, method='b',
    #       beta=beta, ad_epochs=30000, checkpoint_dir=checkpoint_dir)


    # lamb = 1.0
    # device_this_program = 'cuda:0'
    # checkpoint_dir = "./checkpoint/C2I/all-out-"
    # step1('c','i', train=False, c_lr=1e-3, out_dim=12, c_epoch=60, lamb=lamb, checkpoint_dir=checkpoint_dir)
    # beta = 1.0
    # step2('c', 'i', out_dim=12, g_lr=1e-6, d_lr=1e-6, method='b',
    #       beta=beta, ad_epochs=30000, checkpoint_dir=checkpoint_dir)

    # checkpoint_dir = "./checkpoint/C2P/all-out-"
    # device_this_program = "cuda:0"
    # lamb = 1.0
    # step1('c', 'p', c_lr=1e-3, out_dim=12, c_epoch=60, lamb=lamb,
    #       checkpoint_dir=checkpoint_dir, train=True)
    # beta = 1.0
    # step2('c', 'p', out_dim=12, g_lr=1e-6, d_lr=1e-6, method='b',
    #       beta=beta, ad_epochs=60000, checkpoint_dir=checkpoint_dir, d_iter=1)

    # checkpoint_dir = "./checkpoint/P2C/all-out-"
    # device_this_program = "cuda:1"
    # lamb = 1.0
    # step1('p', 'c', c_lr=1e-3, out_dim=12, c_epoch=60, lamb=lamb,
    #       checkpoint_dir=checkpoint_dir, train=True)
    # beta = 1.0
    # step2('p', 'c', out_dim=12, g_lr=1e-6, d_lr=1e-6, method='b',
    #       beta=beta, ad_epochs=60000, checkpoint_dir=checkpoint_dir, d_iter=1)
