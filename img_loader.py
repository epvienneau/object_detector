import os
import numpy as np
from torch.utils import data
import scipy.misc

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
        subinfo = self.sub_list[index]
        coordinate1 = subinfo[0]
        coordinate2 = subinfo[1]
        img_name = subinfo[2]
        img_dir = subinfo[3]

        img_file = os.path.join(img_dir, img_name)
        gray_img = scipy.misc.imread(img_file)
        gray_img = normalizeImage(gray_img)
        data = np.zeros([326, 490, 3], np.float32)
        data[2, :, :] = gray_img
        data[1, :, :] = gray_img
        data[0, :, :] = gray_img

        return data, coordinate1, coordinate2, img_name

    def __len__(self):
        self.total_count = len(self.sub_list)
        return self.total_count


