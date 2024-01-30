from networks.resnet import ResNetFc
from networks import prototype
from networks import discriminator
import torch
import torch.optim as optim
import os 
import matplotlib.pyplot as plt
import utils
import adapt_model_v2 as adapt_model
from dataset import asl_dataset, unit_data


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


def asl_loader(batch_size, root_dir, source, target, out_user):
    data_path = os.path.join(root_dir, "sign", "dataset5")
    s_tr = asl_dataset.get_by_domain(True, source, data_path, batch_size, data_path, out_user)
    s_te = asl_dataset.get_by_domain(False, source, data_path, batch_size, data_path, out_user)
    t_tr = asl_dataset.get_by_domain(True, target, data_path, batch_size, data_path, out_user)
    t_te = asl_dataset.get_by_domain(False, target, data_path, batch_size, data_path, out_user)
    return s_tr, s_te, t_tr, t_te

def step1(source, target, out_user='A',
          checkpoint_dir="./checkpoint/plada-d/Color2depth_A/",
          batch_size=32, out_dim=24, c_lr=1e-3, c_epoch=20,
          lamb=1e-2, gamma=1.0, train=True, angle_loss=None,
          device='cpu', root_dir="."):
    class_num = 24
    checkpoint_dir = checkpoint_dir + str(out_dim)
    device = torch.device(device if (torch.cuda.is_available()) else "cpu")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    s_tr, s_te, t_tr, t_te = asl_loader(batch_size, root_dir, source, target,
                                        out_user)
    Gs = ResNetFc("ResNet18", True, 256, True, 24, True).to(device)
    ps = prototype.Simple_Prototype(class_num, out_dim, device).to(device)
    prototype.proto_init(ps, Gs, s_tr, device)

    c_optim = optim.SGD([{"params": Gs.feature_layers.parameters(), "lr": c_lr},
                        {"params": Gs.fc.parameters(), "lr": c_lr},
                        {"params": ps.parameters(), "lr": c_lr}],
                        lr=c_lr, momentum=0.9, weight_decay=1e-5)
    model = adapt_model.ProtoDA(device=device, gamma=gamma, show_step=10)
    model.step1(Gs, ps, s_tr, s_te, t_te,
                c_optim, c_epoch, checkpoint_dir,
                lamb, train, angle_loss=angle_loss)


def step2(source, target, out_user='A',root_dir='.', device='cpu',
          batch_size=32, out_dim=10, g_lr=1e-4, d_lr=1e-4,
          ad_epochs=10000, checkpoint_dir="./checkpoint/plada-d/Color2depth_A/all-out-",
          beta=1e-2, method='b', train=True):
    class_num=24
    device = torch.device(device if (torch.cuda.is_available()) else 'cpu')
    checkpoint_dir = checkpoint_dir + str(out_dim)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("checkpoint_dir:", os.path.realpath(checkpoint_dir))
    s_tr, s_te, t_tr, t_te = asl_loader(batch_size, root_dir, source, target,
                                        out_user)

    ad_data_tr = unit_data.unit_data_by_loader(s_tr, t_tr, ad_epochs)

    Gs = ResNetFc("ResNet18", True, 256, True, 24, True).to(device)
    Gt = ResNetFc("ResNet18", True, 256, True, 24, True).to(device)

    p = prototype.Simple_Prototype(class_num, out_dim, device).to(device)


    if method == 'a':
        D = discriminator.Discriminator(out_dim, 3072, 2048, 1024, bn=False, sn=True).to(device)
        # D = discriminator.Discriminator(out_dim, 100, 100, 100, bn=True).to(device)
        # D = discriminator.Discriminator_(out_dim, 100).to(device)

    elif method == 'b':
        D = discriminator.Discriminator(class_num, 3072, 2048, 1024, bn=False, sn=True).to(device)
        # D = discriminator.Discriminator(out_dim, 100, 100, 100, bn=True).to(device)
        # D = discriminator.Discriminator_(out_dim, 100).to(device)

    else:
        raise ValueError('Something gets error in methods')

    g_optim = optim.SGD(Gt.parameters(), lr=g_lr, momentum=0.9)
    d_optim = optim.SGD(D.parameters(), lr=d_lr, momentum=0.9)


    model = adapt_model.ProtoDA(device, bn=True)
    model.step2(Gs, p, Gt, D, ad_data_tr, t_te, g_optim, g_optim, 
                checkpoint_dir, ad_epochs, beta=beta, method=method,
                train=train, g_lr=g_lr, d_lr=d_lr, lr_sche=True)


def Color2Depth_out_X(platform, device, out_user, step='1'):
    if step=='1':
        root_dir = root_dir_choose(platform)
        checkpoint_dir = "./checkpoint/color2depth_{}/step1/".format(out_user)
        step1(source='color', target='depth', out_user=out_user,
              checkpoint_dir=checkpoint_dir, batch_size=32, 
              out_dim=24, c_lr=1e-3, c_epoch=20, lamb=1e-2,
              gamma=1.0, train=True, angle_loss=False, device=device, 
              root_dir=root_dir)
    elif step=='2':
        root_dir = root_dir_choose(platform)
        beta=0.01
        checkpoint_dir = "./checkpoint/color2depth_{}/step1/".format(out_user)
        step2(source='color', target='depth', out_user=out_user, root_dir=root_dir, device=device,
              batch_size=32, out_dim=24, g_lr=1e-5, d_lr=1e-5, ad_epochs=10000,
              checkpoint_dir=checkpoint_dir, beta=beta, method='a', train=True)


def Depth2Color_out_X(platform, device, out_user, step='1'):
    if step=='1':
        root_dir = root_dir_choose(platform)
        checkpoint_dir = "./checkpoint/depth2color_{}/step1/".format(out_user)
        step1(source='depth', target='color', out_user=out_user,
              checkpoint_dir=checkpoint_dir, batch_size=32, 
              out_dim=24, c_lr=1e-3, c_epoch=20, lamb=1e-2,
              gamma=1.0, train=True, angle_loss=False, device=device, 
              root_dir=root_dir)
    elif step=='2':
        root_dir = root_dir_choose(platform)
        beta=0.001
        checkpoint_dir = "./checkpoint/depth2color_{}/step1/".format(out_user)
        step2(source='depth', target='color', out_user=out_user, root_dir=root_dir,device=device,
              batch_size=32, out_dim=24, g_lr=1e-5, d_lr=1e-5, ad_epochs=10000,
              checkpoint_dir=checkpoint_dir, beta=beta, method='a', train=True)


if __name__ == "__main__":
    # Color2Depth_out_X('2070_1', 'cuda:0', 'E', '2')

    
    
    Depth2Color_out_X('2070_1', 'cuda:1', 'E', '2')



    # Color2Depth_out_X('p5000', 'cuda:0', 'E', '2')
    # Depth2Color_out_X('p5000', 'cuda:0', 'E', '2')
