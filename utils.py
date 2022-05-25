import os
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import random
import shutil
from PIL import Image
import numpy as np
import cv2
from PIL import Image
import PIL
import collections
import os
from pathlib import Path
import json


def mkdir(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)


def split_train_test(data_root, rate=0.1):
    image_dir = data_root + '/' + 'Images'
    mask_dir = data_root + '/' + 'Masks'

    mask_names = [i.name for i in Path(mask_dir).glob('*.png')]
    random.shuffle(mask_names)
    test_num = int(len(mask_names) * rate)

    test_mask_names = mask_names[:test_num]
    train_mask_names = mask_names[test_num:]

    save_dir = str(Path(mask_dir).parent)
    os.system('rm -rf ' + save_dir + '/train')
    os.system('rm -rf ' + save_dir + '/test')

    train_imgs = save_dir + '/train/images'
    train_masks = save_dir + '/train/masks'
    test_imgs = save_dir + '/test/images'
    test_masks = save_dir + '/test/masks'
    mkdir(train_imgs)
    mkdir(train_masks)
    mkdir(test_imgs)
    mkdir(test_masks)

    for name in train_mask_names:
        name = name[:-3]
        shutil.copy(image_dir + '/' + name + 'jpg', train_imgs)
        shutil.copy(mask_dir + '/' + name + 'png', train_masks)

    for name in test_mask_names:
        name = name[:-3]
        shutil.copy(image_dir + '/' + name + 'jpg', test_imgs)
        shutil.copy(mask_dir + '/' + name + 'png', test_masks)


def find_colors(data_root):
    png_dir = data_root + '/' + 'ImagesPNG'
    save_dir = data_root + '/' + 'Masks'
    mkdir(save_dir)

    colors = []
    img = Image.open(png_dir + '/wh0025.png').convert('RGB')
    img = np.array(img)

    h,w,c = img.shape
    for i in range(h):
        for j in range(w):
            color = img[i,j,:]
            color = tuple(color)
            if color not in colors:
                colors.append(color)
    print(colors)

    # color  = [(192, 192, 0), (0, 255, 0), (128, 128, 128), (255, 255, 0), (255, 0, 0), (0, 0, 255)]
    # name   = ['lemon',       'green',     'gray',          'yellow',      'red',       'blue']
    # class_ = ['pavement',    'vegetation','bare soil',     'road',        'building',  'water']
    # id     = [0,             1,           2,               3,             4,           5]

def copy_test_png(dataset_dir, save_dir):
    for path in Path(dataset_dir + '/test/masks').glob('*.png'):
        path2 = dataset_dir + '/ImagesPNG/' + path.name
        shutil.copy(path2, save_dir)

###############################################################################

def save_feature_map(x, model_name, save_name):
    # model_name = 'segnet'
    # save_name = '1'
    with torch.no_grad():
        n, c, h, w = x.shape
        for c in range(c):
            tmp = np.array(x[0, c, :, :].cpu()).copy()
            tag = model_name + '_stage' + save_name + '_ch' + str(c)
            min_ = np.min(tmp)
            max_ = np.max(tmp)
            tmp = (tmp - min_) / (max_ - min_)
            tmp = (tmp * 255).astype('uint8')
            print('saving', tag, tmp.shape)
            cv2.imwrite(tag + '.jpg', tmp)

def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:
        w_class = median_freq / freq_class,
    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes
    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq

#####################################################################################

def add_mask_to_source_multi_classes(source_np, mask_np, num_classes):
    colors = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [255, 255, 0]]
    foreground_mask_bool = mask_np.astype('bool')
    foreground_mask = mask_np * foreground_mask_bool
    foreground = np.zeros(source_np.shape, dtype='uint8')
    background = source_np.copy()

    for i in range(1, num_classes + 1):
        fg_tmp = np.where(foreground_mask == i, 1, 0)
        fg_tmp_mask_bool = fg_tmp.astype('bool')

        fg_color_tmp = np.zeros(source_np.shape, dtype='uint8')
        fg_color_tmp[:, :] = colors[i]
        for c in range(3):
            fg_color_tmp[:, :, c] *= fg_tmp_mask_bool
        foreground += fg_color_tmp
    foreground = cv2.addWeighted(source_np, 0.1, foreground, 0.9, 0)

    for i in range(3):
        foreground[:, :, i] *= foreground_mask_bool
        background[:, :, i] *= ~foreground_mask_bool

    show = foreground + background
    # plt.imshow(show)
    # plt.pause(0.5)
    return show


def vis_pred(mask_np, num_classes):
    h, w = mask_np.shape
    show = np.zeros((h,w,3), dtype='uint8')
    colors  = [[192, 192, 0], [0, 255, 0], [128, 128, 128], [255, 255, 0], [255, 0, 0], [0, 0, 255]]

    for i in range(num_classes):
        fg_mask_tmp = (np.where(mask_np == i, 1, 0)).astype('bool')

        fg_color_tmp = np.zeros((h,w,3), dtype='uint8')
        for j in range(3):
            fg_color_tmp[:,:,j] = colors[i][j]
            fg_color_tmp[:,:,j] = fg_color_tmp[:,:,j] * fg_mask_tmp
        show = show + fg_color_tmp

    show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
    return show

def add_mask_to_source(source_np, mask_np, color):
    mask_bool = (np.ones(mask_np.shape, dtype='uint8') & mask_np).astype('bool')

    foreground = np.zeros(source_np.shape, dtype='uint8')
    for i in range(3):
        foreground[:, :, i] = color[i]
    foreground = cv2.addWeighted(source_np, 0.5, foreground, 0.5, 0)

    background = source_np.copy()
    for i in range(3):
        foreground[:, :, i] *= mask_bool
        background[:, :, i] *= (~mask_bool)

    return background + foreground

##########################################################################

def plot_compare_curves(pt_root):
    pt_dirs = [i for i in Path(pt_root).iterdir() if i.is_dir() and 'ori' in i.name]
    Es, Ls, Ms, Ps = [], [], [], []
    Ns = []
    for pt_dir in pt_dirs:
        logfile_path = str(pt_dir) + '/val_log.json'
        print(logfile_path)
        with open(logfile_path) as f:
            logs = json.load(f)
        E, L, M, P = [], [], [], []
        for log in logs:
            epoch = log['epoch']
            loss = log['loss']
            miou = log['miou']
            pa = log['pa']
            # print(log)
            E.append(epoch)
            L.append(loss)
            M.append(miou)
            P.append(pa)
        Es.append(E)
        Ls.append(L)
        Ms.append(M)
        Ps.append(P)
        # Ns.append(pt_dir.name.split('-')[1])
        Ns.append(pt_dir.name)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(Es)):
        E = Es[i]
        N = Ns[i]
        M = Ms[i]
        ax.plot(E, M, '-', label=N)
    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    # ax.set_ylim(25, 35)
    plt.savefig(pt_root + "/mIoU_compare.png")
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(Es)):
        E = Es[i]
        N = Ns[i]
        P = Ps[i]
        ax.plot(E, P, '-', label=N)
    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PA")
    # ax.set_ylim(25, 35)
    plt.savefig(pt_root + "/PA_compare.png")
    # plt.show()


def training_curves(logfile_path):
    with open(logfile_path) as f:
        l1 = json.load(f)

    logfile_path2 = logfile_path.replace('l1', 'mse')
    with open(logfile_path2) as f2:
        l2 = json.load(f2)

    epochs = [i for i in range(50)]
    losses_l1 = [i['loss'] * 1e4 for i in l1]
    psnrs_l1 = [i['psnr'] for i in l1]
    ssims_l1 = [i['ssim'] for i in l1]

    psnrs_l2 = [i['psnr'] for i in l2]
    ssims_l2 = [i['ssim'] for i in l2]
    losses_l2 = [i['loss'] * 1e5 for i in l2]

    def show_psnr():
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(epochs, psnrs_l1, '-', label='L1')
        ax.plot(epochs, psnrs_l2, '-r', label='L2')

        ax.legend(loc=0)
        ax.grid()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR (dB)")
        ax.set_ylim(25, 35)

        plt.savefig("psnr.png")
        plt.show()

    def show_ssim():
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(epochs, ssims_l1, '-', label='L1')
        ax.plot(epochs, ssims_l2, '-r', label='L2')

        ax.legend(loc=0)
        ax.grid()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SSIM")
        ax.set_ylim(0.7, 1)

        plt.savefig("ssim.png")
        plt.show()

    def show_loss():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Epoch")

        lns1 = ax.plot(epochs, losses_l1, '-', label='L1 loss')

        ax.set_ylim(130, 380)
        plt.yticks([])
        ax2 = ax.twinx()
        lns2 = ax2.plot(epochs, losses_l2, '-r', label='L2 loss')

        ax2.set_ylim(45, 200)

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)

        plt.yticks([])

        plt.savefig("loss.png")
        plt.show()

    show_psnr()
    show_ssim()
    show_loss()

if __name__ == '__main__':
    # find_colors('/home/SENSETIME/sunxin/3_datasets/WHDLD')

    # split_train_test('/home/SENSETIME/sunxin/3_datasets/WHDLD')

    # copy_test_png('/home/SENSETIME/sunxin/3_datasets/WHDLD', '/home/SENSETIME/sunxin/2_myrepo/0_orders/whdld_segment/Result_images')

    plot_compare_curves('/home/SENSETIME/sunxin/2_myrepo/0_orders/whdld_segment/Results')
