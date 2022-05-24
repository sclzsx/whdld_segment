import random
import shutil
from PIL import Image
import numpy as np
import cv2
from PIL import Image
import PIL
from utils import add_mask_to_source_multi_classes
import collections
import os
from pathlib import Path
import json


def check_label_in_jsons():
    classes = ('background', 'yumi', 'yumiye', 'yumixin')

    image_dir = 'Data/sourceChangeNamePathGoodAnnoAgain'
    json_dir = 'Data/sourceChangeNamePathGoodAnnoAgain'

    out_images_dir = 'Data/sourceChangeNamePathGoodAnnoAgainCheck'
    out_labels_dir = 'Data/sourceChangeNamePathGoodAnnoAgainCheck'

    if not os.path.exists(out_images_dir):
        os.makedirs(out_labels_dir)
    if not os.path.exists(out_labels_dir):
        os.makedirs(out_labels_dir)

    class_dict = dict(zip(classes, range(len(classes))))
    json_paths = [i for i in Path(json_dir).rglob('*.json')]
    for json_path in json_paths:
        file_name = json_path.name[:-5]
        image = cv2.imread(image_dir + '/' + file_name + '.jpg')
        h, w, c = image.shape
        # cv2.imwrite(image_dir + '/' + file_name + '.jpg', image)
        shutil.copy(image_dir + '/' + file_name + '.jpg', out_images_dir)
        with open(str(json_path), 'r') as f:
            data = json.load(open(str(json_path)))
        top_layer = np.zeros((h, w), dtype='uint8')
        bottom_layer = np.zeros((h, w), dtype='uint8')
        # print(file_name)
        for shape in data['shapes']:
            class_name = shape['label']
            if class_name in classes:
                mask = PIL.Image.fromarray(np.zeros((h, w), dtype=np.uint8))
                draw = PIL.ImageDraw.Draw(mask)
                xy = [tuple(point) for point in shape['points']]
                draw.polygon(xy=xy, outline=1, fill=1)
                mask = np.array(mask, dtype=bool)
                layer_tmp = np.full((h, w), class_dict[class_name], dtype='uint8') * mask
                top_layer += layer_tmp
        full_mask = top_layer + bottom_layer * (~top_layer.astype('bool'))

        print(file_name, collections.Counter(np.array(full_mask).flatten()))

        show = add_mask_to_source_multi_classes(image, full_mask, len(classes))
        cv2.imwrite(out_images_dir + '/' + file_name + '.png', show)


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


def change_name_path_in_jsons():
    root = 'Data/source/labels'  # 根目录，py文件所在的路径
    save_root = 'Data/sourceChangeNamePath'

    for json_path in Path(root).rglob('*.json'):  # 遍历文件夹及子文件下下所有符合条件的文件
        with open(str(json_path), "r") as file:  # str是把windows路径转为string, r是只读
            dict = json.load(file)

            # for k,v in dict.items():
            # if k != "imageData" and k!= 'shapes':
            #     print(k,v)

            dict['imagePath'] = dict['imagePath'].replace('bmp', 'jpg').replace('jpeg', 'jpg').replace("..//biaozhu//",
                                                                                                       '')
            print(dict['imagePath'])
            # dict['imageData'] = ""

        with open(save_root + '/' + json_path.name, "w") as f:  # str是把windows路径转为string, r是只读
            json.dump(dict, f)


def labelme_jsons_to_masks_by_poly():
    classes = ('background', 'yumi', 'yumiye', 'yumixin')
    # classes = ('background', 'roads', 'roadside', 'ground_mark', 'zebra_crs')

    image_dir = 'Data/sourceChangeNamePathGoodAnnoAgain'
    # json_dir = 'Data/sourceChangeNamePathGoodAnnoAgain'

    out_images_dir = image_dir + 'MyAug'

    out_vis_dir = out_images_dir + 'Vis'

    if not os.path.exists(out_images_dir):
        os.makedirs(out_images_dir)

    if not os.path.exists(out_vis_dir):
        os.makedirs(out_vis_dir)

    cnt = 0

    class_dict = dict(zip(classes, range(len(classes))))
    json_paths = [i for i in Path(image_dir).rglob('*.json')]
    for json_path in json_paths:
        file_name = json_path.name[:-5]
        image = cv2.imread(image_dir + '/' + file_name + '.jpg')
        h, w, c = image.shape

        # shutil.copy(image_dir + '/' + file_name + '.jpg', out_images_dir)
        with open(str(json_path), 'r') as f:
            data = json.load(open(str(json_path)))
        # top_layer = np.zeros((h, w), dtype='uint8')
        full_mask = np.zeros((h, w), dtype='uint8')

        assert h == data["imageHeight"] and w == data["imageWidth"]
        # print(file_name)
        for shape in data['shapes']:
            class_name = shape['label']

            if class_name in classes:
                mask = PIL.Image.fromarray(np.zeros((h, w), dtype=np.uint8))
                draw = PIL.ImageDraw.Draw(mask)
                xy = [tuple(point) for point in shape['points']]
                draw.polygon(xy=xy, outline=1, fill=1)
                mask = np.array(mask, dtype=bool)
                layer_tmp = np.full((h, w), class_dict[class_name], dtype='uint8') * mask

                full_mask += layer_tmp

        def MyAug(data, label):
            results = []
            N = 40
            while N > 0:
                if np.random.rand() > 0.5:
                    data = np.flipud(data)
                    label = np.flipud(label)
                if np.random.rand() > 0.5:
                    data = np.fliplr(data)
                    label = np.fliplr(label)
                if np.random.rand() > 0.5:
                    data = np.rot90(data, k=1)
                    label = np.rot90(label, k=1)
                if np.random.rand() > 0.5:
                    data = np.rot90(data, k=3)
                    label = np.rot90(label, k=3)
                if np.random.rand() > 0.5:
                    data = np.rot90(data, k=3)
                    label = np.rot90(label, k=3)
                results.append((data, label))
                N -= 1
            return results

        def MyHandleCrop(img1, img2):
            # 2448 2048
            X = [0, 800, 1424]
            Y = [0, 700, 1024]
            assert len(X) == len(Y)

            results = []
            for i in range(len(X)):
                x = X[i]
                y = Y[i]

                y1 = y + win
                x1 = x + win

                rand_patch1 = img1[y: y1, x:x1, :]
                rand_patch2 = img2[y: y1, x:x1]
                results.append((rand_patch1, rand_patch2))

                return results

        results = MyAug(image, full_mask)
        results.append((image, full_mask))  # without augment
        for result in results:
            data, label = result[0], result[1]
            win = 1024
            pairs = MyHandleCrop(data, label)
            pairs.append((data, label))  # without crop
            for pair in pairs:
                img, lab = pair[0], pair[1]

                img = cv2.resize(img, (256, 256))

                lab = Image.fromarray(lab)
                lab = lab.resize((256, 256))

                lab_dict = collections.Counter(np.array(lab).flatten())
                lab_list = list(lab_dict.keys())
                class_list = [0, 1, 2, 3]
                errors = [i for i in lab_list if i not in class_list]
                if len(errors) == 0:
                    print(cnt, lab_dict)
                    cv2.imwrite(out_images_dir + '/' + str(cnt) + '.jpg', img)
                    lab.save(out_images_dir + '/' + str(cnt) + '.png')
                    cv2.imwrite(out_vis_dir + '/' + str(cnt) + '.jpg', np.array(lab) * 100)
                else:
                    print('################ PASS #########################', cnt, lab_dict)

                cnt += 1


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

if __name__ == '__main__':
    # find_colors('/home/SENSETIME/sunxin/3_datasets/WHDLD')

    split_train_test('/home/SENSETIME/sunxin/3_datasets/WHDLD')
