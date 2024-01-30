import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np  
from dataset import unit_data,usps,mnist,mnist_m
import os 
import matplotlib.pyplot as plt
import utils
from networks import prototype,mnist_usps
import model_func

class ADDA():
    def __init__(self,device='cpu'):
        self.device=device 
        self.ce_criterion = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

    def gan_g_loss(self,d_fake,real_label):
        g_loss = self.bce(d_fake,real_label)
        return g_loss

    def gan_d_loss(self,d_real,real_label,d_fake,fake_label):
        d_loss_real = self.bce(d_real,real_label)
        d_loss_fake = self.bce(d_fake,fake_label)
        d_loss = d_loss_real + d_loss_fake 
        return d_loss 

    #------adversarial loss acording to lsGAN============== 
    def ls_d_loss(self,d_real,b,d_fake,c):
        '''use ls d_loss according to lsgan'''
        ls_d_loss = 0.5 * torch.sum(torch.pow(d_real-b,exponent=2))\
                    + 0.5 * torch.sum(torch.pow(d_fake-c,exponent=2))
        return ls_d_loss 

    def ls_g_loss(self,d_fake,a):
        '''use ls g_loss according to lsGAN'''
        a = torch.ones_like(d_fake)
        ls_g_loss = 0.5 * torch.sum(torch.pow(d_fake-a,exponent=2))
        return ls_g_loss 

    def step1(self, tr_s_dataloader, te_s_dataloader, te_t_dataloader,
              Gs, optim, train, bn, c_epochs, checkpoint_dir, show_step=10):

        if train==True:
            best_s_acc = 0
            best_t_acc = 0
            Gs.train()
            for epoch in range(c_epochs):
                for i,(image, label) in enumerate(tr_s_dataloader):
                    # if bn:
                    #     Gs.train()
                       
                    image, label = image.to(self.device),label.to(self.device)
                    optim.zero_grad()

                    # do inference
                    feat = Gs(image)
                    loss = self.ce_criterion(feat, label)

                    loss.backward()
                    optim.step()

                    # do summary for this iteration
                    _, predict = torch.max(feat.data, 1)
                    correct = (predict==label).sum().item()
                    
                    acc = correct / label.size(0)
                    if i % show_step == 0:
                        print("i/epoch:{}/{},loss:{:.4f},train_acc:{:.4%}".format(i,epoch,loss.item(),acc))
                # if bn:
                #     Gs.eval()

                with torch.no_grad():
                    correct, total=0,0
                    for _,(image,label) in enumerate(te_s_dataloader):
                        image,label = image.to(self.device),label.to(self.device)
                        feat = Gs(image)
                        pos, predict = torch.max(feat.data,1)
                        correct += (predict==label).sum().item()
                        total += label.size(0)
                    acc = correct / total
                    if acc > best_s_acc:
                        best_s_acc = acc
                        torch.save(Gs.state_dict(),os.path.join(checkpoint_dir,"source-best-Gs.pt"))
                    print("epoch:{}, s_accuracy:{:.4%}, s_best accuracy:{:.4%}".format(epoch, acc, best_s_acc))
                    print("--------------------------------")
                    correct, total=0,0
                    for k,(image, label) in enumerate(te_t_dataloader):
                        image, label = image.to(self.device), label.to(self.device)
                        feat = Gs(image)
                        pos, predict = torch.max(feat.data, 1)
                        correct += (predict==label).sum().item()
                        total += label.size(0)
                    acc = correct / total
                    if acc > best_t_acc:
                        best_t_acc = acc 
                    print("epoch:{}, t_accuracy:{:.4%}, t_best accuracy:{:.4%}".format(epoch, acc, best_t_acc))
                print("--------------------------------")
            torch.save(Gs.state_dict(), os.path.join(checkpoint_dir, "Gs.pt"))

        Gs.load_state_dict(torch.load(os.path.join(checkpoint_dir, "Gs.pt")))

        if bn:
            Gs.eval()

        with torch.no_grad():
            correct,total=0,0
            for _,(image,label) in enumerate(te_s_dataloader):
                image,label = image.to(self.device),label.to(self.device)
                feat = Gs(image)
                pos,predict = torch.max(feat.data,1)
                correct += (predict==label).sum().item()
                total += label.size(0)
            acc = correct / total
            print("s_accuracy:{}".format(acc))
            
            correct,total=0,0
            for _,(image,label) in enumerate(te_t_dataloader):
                image,label = image.to(self.device),label.to(self.device)
                feat = Gs(image)
                pos,predict = torch.max(feat.data,1)
                correct += (predict==label).sum().item()
                total += label.size(0)
            acc = correct / total
            print("t_accuracy:{}".format(acc))
    
    def step2(self,Gs,Gt,D,
            ad_data_tr,t_data_te,g_optim,d_optim,checkpoint_dir,ad_epochs,
            bn=False,G_iter=1,D_iter=1,test_step=30):
        #g_loss_function,d_loss_function = self.loss_chooser(loss_style)
        # step2 adversrial train cnn feature and prototype
        # load data best model in source network
        Gs.load_state_dict(torch.load(os.path.join(checkpoint_dir,"source-best-Gs.pt")))
        #Gt.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'source-best-Gs.pt')))
        model_func.model_load(Gt, os.path.join(checkpoint_dir, 'source-best-Gs.pt'), [""])
        if bn:
            Gs.eval()
            Gt.train()
            D.train()

        best_adapt_acc = 0
        adapt_acc=[]
        adapt_step=[]
        plt.ion()
        # start aversarial training
        for i, (xs,ys,xt,yt) in enumerate(ad_data_tr,0):
            xs, ys = xs.to(self.device),ys.to(self.device)
            xt, yt = xt.to(self.device),yt.to(self.device)

            # set zero to device
            #self.Gt.train()
            if bn:
                Gt.train()
                D.train()
            for _ in range(D_iter):
                D.zero_grad()

                # do inference
                feat_gs = Gs(xs)
                feat_gt = Gt(xt)

                ds = D(feat_gs)
                dt = D(feat_gt)


                # make d loss
                real_label = torch.ones_like(ds)
                fake_label = torch.zeros_like(dt)
                d_loss = self.gan_d_loss(ds,real_label,dt,fake_label)
                #d_loss = self.ls_d_loss(ds,real_label,dt,fake_label)
                # do backward and step
                d_loss.backward()
                d_optim.step()

                # summary item
                d_loss_ = d_loss.detach().item()
                ds_ = ds.detach().mean().item()
                dt_ = dt.detach().mean().item()

            # (2) train Gt
            for _ in range(G_iter):
                Gt.zero_grad()
               

                # do inference
                feat_gt = Gt(xt)
                dt = D(feat_gt)
                real_label = torch.ones_like(dt)

                # make g_loss
                g_loss = self.gan_g_loss(dt,real_label)
                #g_loss = self.ls_g_loss(dt,real_label)

                # do backward and step
                g_loss.backward()
                
                g_optim.step()

                # summary item
                g_loss_ = g_loss.detach().item()
                dtg_ = dt.detach().mean().item()

            if i % 30 == 0:
                # print("i/epoch:{}/{},D_loss:{:.4f},G_loss:{:.4f},D(Gs(xs):{:.4f},D(Gt(xt)):{:.4f}/{:.4f}"\
                #     .format(i,ad_epochs,d_loss_,g_loss_,ds_,dt_,dtg_))
                print("i/epoch:{}/{},D_loss:{:.4f},G_loss:{:.4f},D(Gs(xs):{:.4f},D(Gt(xt)):{:.4f}/{:.4f}"\
                    .format(i,ad_epochs,d_loss_,g_loss_,ds_,dt_,dtg_))

            if i % test_step == 0 or (i>ad_epochs - 10):
                if bn:
                    Gt.eval()
                 
                with torch.no_grad():
                    correct, total = 0,0
                    for _,(x,y) in enumerate(t_data_te,0):
                        x,y = x.to(self.device),y.to(self.device)
                        feat = Gt(x)
                        pos,predict = torch.max(feat.data,1)
                        correct += (predict==y).sum().item()
                        total += y.size(0)  
                    acc = correct / total
                    adapt_acc.append(acc)
                    adapt_step.append(i)

                    np.save(checkpoint_dir+"/adapt_acc.npy",np.array(adapt_acc))
                    np.save(checkpoint_dir+"/adapt_step.npy",np.array(adapt_step))
                    plt.cla()
                    plt.plot(adapt_step,adapt_acc,c='red',marker='o')
                    plt.pause(0.5)
                    if best_adapt_acc < acc:
                        best_adapt_acc = acc
                        torch.save(Gt.state_dict(),os.path.join(checkpoint_dir,'Gt.pt'))
                       

                    print("Testing:iter/epoch:{}/{},acc:now/best:{:.4f}/{:.4f}"\
                        .format(i,ad_epochs,acc,best_adapt_acc))
        np.save(checkpoint_dir+"adapt_acc.npy",np.array(adapt_acc))
        np.save(checkpoint_dir+"adapt_step.npy",np.array(adapt_step))
        plt.ioff()
        plt.show()

def adda_step1(source='mnist',target='usps',batch_size=64,c_lr=0.001,c_epochs=20,
               checkpoint_dir="./checkpoint/MNIST2MNIST_M/adda_bn_useall",
               use_all=True,bn=False,train=True):
    image_size=28
    nc = 3
    class_num = 10
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(os.path.realpath(checkpoint_dir))

    # mnist dataset
    s_tr,s_te = mnist.get_train_and_test(batch_size,use_all,image_size=image_size,gray=False)
    # usps dataset 
    t_te = mnist_m.get_mnist_m(False,batch_size,None,image_size=image_size)

    Gs = mnist_usps.CNN_Extract(nc,class_num,bn).to(device)
    # prepare optimizer
    c_optim = optim.SGD(list(Gs.parameters()),lr=c_lr,momentum=0.9,weight_decay=0.0005)
    
    model =ADDA(device = device)
    model.step1(s_tr,s_te,t_te,
                Gs,c_optim,train,bn,c_epochs,checkpoint_dir)

def adda_step2(source='mnist',target='usps',batch_size=64,
              g_lr=0.00002,d_lr=0.00002,ad_epochs=2000,
              checkpoint_dir="./checkpoint/MNIST2MNIST_M/adda_bn_useall",
              use_all=True,bn=False,G_iter=1,D_iter=1):

    # hyperparameter
    image_size=28
    nc = 3
    class_num=10
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("checkpoint_dir:",os.path.realpath(checkpoint_dir))

    # prepare data
    s_data_tr = mnist.get_mnist(True,batch_size,use_all,None,None,image_size,False)
    t_data_tr = mnist_m.get_mnist_m(True,batch_size,None,None,image_size)
    ad_data_tr = unit_data.unit_data_by_loader(s_data_tr,t_data_tr,ad_epochs)
    
    t_data_te = mnist_m.get_mnist_m(False,batch_size,image_size=image_size)
    Gs = mnist_usps.CNN_Extract(nc,class_num,bn).to(device)
    Gt = mnist_usps.CNN_Extract(nc,class_num,bn).to(device)
    D = mnist_usps.Discriminator(class_num,bn).to(device)
    
    g_optim = optim.Adam(list(Gt.parameters()),lr=g_lr,betas=(0.5,0.999))
    d_optim = optim.Adam(D.parameters(),lr=d_lr,betas=(0.5,0.999))

    model = ADDA(device=device)
    model.step2(Gs,Gt,D,
                ad_data_tr,t_data_te,g_optim,d_optim,checkpoint_dir,ad_epochs,
                bn,G_iter,D_iter)


def feature_analysis(source="mnist",target="usps",
                    checkpoint_dir="./checkpoint/MNIST2USPS/adda_bn_useall"):
    image_size=28
    nc = 1
    class_num = 10
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    Gs = mnist_usps.CNN_Extract(nc,class_num,bn=True).to(device)
    Gt = mnist_usps.CNN_Extract(nc,class_num,bn=True).to(device)
    Gs.load_state_dict(torch.load(os.path.join(checkpoint_dir,"Gs.pt")))
    Gt.load_state_dict(torch.load(os.path.join(checkpoint_dir,'Gt.pt')))
    s_data_te = mnist.get_mnist(False,batch_size=-1,use_all=True,image_size=image_size,gray=True)
    t_data_te = usps.get_usps(False,batch_size=-1,use_all=True,image_size=image_size)

    img_s,label_s = next(iter(s_data_te))
    img_t,label_t = next(iter(t_data_te))
    img_s,img_t=img_s.to(device),img_t.to(device)
    label_s,label_t = label_s.detach().numpy(),label_t.detach().numpy()
    feat_s = Gs(img_s).to('cpu').detach().numpy()
    feat_t = Gs(img_t).to('cpu').detach().numpy()

    utils.plot_tsne_da(None,None,feat_s,label_s,feat_t,label_t,name="adda feature distribution")
    plt.show()
    
if __name__ == "__main__":
    #feature_analysis()
    adda_step2()
    #adda_step1()
