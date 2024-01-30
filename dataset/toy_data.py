import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np 
import os 
from PIL import Image 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
import scipy.ndimage
import math 
from matplotlib.colors import ListedColormap

class toy(Dataset):
    def __init__(self,train=True,n_samples=300,types='source',radians=30):
        seed_tr = 10
        seed_te_s = 20
        seed_tr = 10
        seed_te_t = 20
        if train:
            noisy_moons = datasets.make_moons(n_samples=2*n_samples,noise=0.05,random_state=seed_tr)
        else:
            noisy_moons = datasets.make_moons(n_samples=2*n_samples,noise=0.05,random_state=seed_tr)
        self.x,self.y = noisy_moons
        if types=='target':
            radians = math.radians(-1*radians)
            center = np.array([0.5,0.25])
            new_x0,new_x1 = self.rotate_around_point_highperf([self.x[:,0],self.x[:,1]], radians, origin=center)
            self.x = np.stack((new_x0,new_x1),axis=1)

    def rotate_around_point_highperf(self,xy, radians, origin=(0, 0)):
        """Rotate a point around a given point.
        
        I call this the "high performance" version since we're caching some
        values that are needed >1 time. It's less readable than the previous
        function but it's faster.
        """
        x, y = xy
        offset_x, offset_y = origin
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return qx, qy

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        X = torch.from_numpy(self.x[idx]).float()
        y = self.y[idx].astype(np.int64)
        return X,y
        
def get_toy(train,batch_size=64,n_samples=300,types='source',radians=30):
    dataset = toy(train,n_samples,types,radians)
    loader = DataLoader(dataset,batch_size,train,num_workers=2)
    return loader

if __name__ == "__main__":
    source_data = toy(types='source')
    target_data = toy(types='target')
    x,y=source_data.x,source_data.y 
    new_x,new_y = target_data.x,target_data.y
    fig = plt.figure() 
    
    cm_bright = ListedColormap(['red','green'])
    plt.scatter(x[:,0],x[:,1],s=10,c=y,cmap=cm_bright)
    plt.scatter(new_x[:,0],new_x[:,1],s=10,marker='x',c='blue')
    plt.show()





            



