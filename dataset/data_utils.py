import numpy as np 
import os 
from skimage import io,transform,img_as_ubyte
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


mnistm_path = "./data/mnist_m"

def save_as_npy(train):
    if train:
        data_dir = os.path.join(mnistm_path,"mnist_m_train")
        label_dir = os.path.join(mnistm_path,"mnist_m_train_labels.txt")
    else:
        data_dir = os.path.join(mnistm_path,"mnist_m_test")
        label_dir = os.path.join(mnistm_path,"mnist_m_test_labels.txt")

    label_f = open(label_dir)
    records = np.array(label_f.readlines())
    label_f.close()
    image_list,label_list = [],[]

    for i in records:
        print("now:{}".format(i))
        line = i.strip().split()
        img_path = os.path.join(data_dir,line[0])
        img = io.imread(img_path)
        #print("change before",img.shape,img.dtype )
        #print(np.max(img),np.min(img))
        img = img_as_ubyte(transform.resize(img,(28,28,3)))
        #print("change after",img.shape,img.dtype)
        #print(np.max(img),np.min(img))
        label = line[1]
        image_list.append(img)
        label_list.append(int(label))
        

    result_dict = {'images':np.array(image_list),'labels':np.array(label_list)}
    
    filename = 'train_uint8_28x28x3.npy' if train else 'test_uint8_28x28x3.npy'
    np.save(os.path.join(mnistm_path,'test',filename),result_dict)
    print('done!')

def imshow_grid(images, shape=[2,8]):
    '''Plot images in a grid of a given shape.'''
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0]*shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])

    plt.show()

def read_npy(path):
    data = np.load(path).item()
    images = data['images']
    label = data['labels']
    print(label[:30],)
    print(images[0,:,:,:].shape)
    imshow_grid(images,[3,10])
    



if __name__ == "__main__":
    #read_npy("./data/mnist_m/test/train28x28x3.npy")
    save_as_npy(False)








