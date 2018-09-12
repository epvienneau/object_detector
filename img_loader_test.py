import os
import numpy as np
from torch.utils import data
import scipy.misc
import cv2

def normalizeImage(img):
    img = img.astype('float')
    # Do not touch the alpha channel
    minval = img.min()
    maxval = img.max()
    if minval != maxval:
        img -= minval
        img /= (maxval-minval)
    return img*255


class img_loader(data.Dataset):
    def __init__(self, sub_list):
        self.sub_list = sub_list

    def __getitem__(self, index):
        # load image
        subinfo = self.sub_list
        img_name = subinfo[0][index]
        img_dir = subinfo[1][index]
        img_file = os.path.join(img_dir, img_name)
        gray_img = scipy.misc.imread(img_file)
        gray_img = normalizeImage(gray_img)
        gray_img = cv2.resize(gray_img, (224, 224)) #now its 224x224x3 for resnet
        gray_img = np.swapaxes(gray_img, 0, 2) #make it 3x512x512
        gray_img = gray_img.astype('float')
        return [gray_img]

    def __len__(self):
        self.total_count = len(self.sub_list)
        return self.total_count


