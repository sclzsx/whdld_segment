import torch
import time
import numpy as np
import cv2
import os
from torchvision import transforms
from utils import add_mask_to_source_multi_classes, vis_pred
from pathlib import Path
from dataset import SegDataset
from choices import get_criterion
from matplotlib import pyplot as plt
from metric import SegmentationMetric
import json
from collections import Counter
from tqdm import tqdm
from scipy import interpolate


def predict_a_batch(net, out_channels, batch_data, batch_label, class_weights, do_criterion, do_metric):
    if batch_label is None:  # 针对图片或视频帧的预测，没有对应的label，随机生成一个和data等大的label
        batch_label = torch.randn(batch_data.shape[0], out_channels, batch_data.shape[2], batch_data.shape[3])
    with torch.no_grad():
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        output = net(batch_data)
        # if batch_data.shape[3] != output.shape[3]:  # 通过H判定输出是否比输入小，主要针对deeplab系列，将输出上采样至输入大小
        #     output = F.interpolate(output, size=(batch_data.shape[2], batch_data.shape[3]), mode="bilinear", align_corners=False)
        if out_channels == 1:
            batch_label = batch_label.float()  # labels默认为long，通道为1时采用逻辑损失，需要data和label均为float
            output = torch.sigmoid(output).squeeze().cpu()  # Sigmod回归后去掉批次维N
            prediction_np = np.where(np.array(output) > 0.5, 1, 0)  # 阈值默认为0.5
        else:
            batch_label = batch_label.squeeze(1)  # 交叉熵损失需要去掉通道维C

            prediction_np = np.array(torch.max(output.data, 1)[1].squeeze(0).cpu())  # 取最大值的索引作为标签，并去掉批次维N

            # fg_idx = 1
            # net_output = torch.abs(output.cpu())  # NCHW
            # fg = torch.max(net_output.data, 1)[1].squeeze(0)
            # sum_map = torch.sum(net_output.squeeze(0), dim=0) + 1e-6
            # sum_map = np.array(sum_map)
            # fg_mask = np.array(fg).astype('bool')
            # select_map = np.array(net_output.squeeze(0)[fg_idx])
            # score_map = (select_map / sum_map) * fg_mask
            # prediction_np = np.where(score_map > 0.9, 1, 0)

        loss, pa, miou = None, None, None

        criterion = get_criterion(out_channels, class_weights)
        if do_criterion:
            loss = criterion(output.cuda(), batch_label).item()

        if do_metric:
            metric = SegmentationMetric(out_channels)
            metric.update(output, batch_label)
            # metric.update(output.cuda(), batch_label.cuda())
            pa, miou = metric.get()

        return prediction_np, loss, (pa, miou)


def eval_dataset_full(net, out_channels, loader, class_weights=None, save_dir=None):
    mious, pas, losses, batch_data_shape = [], [], [], ()
    for i, (batch_data, batch_label) in enumerate(loader):
        if i == 0:
            batch_data_shape = batch_data.shape
        _, loss, (pa, miou) = predict_a_batch(net, out_channels, batch_data, batch_label, class_weights=class_weights,
                                              do_criterion=True, do_metric=True)
        losses.append(loss)
        mious.append(miou)
        pas.append(pa)
        print('Predicted batch [{}/{}], Loss:{}, IoU:{}, PA:{}'.format(i, len(loader), round(loss, 3), round(miou, 3),
                                                                       round(pa, 3)))
    mean_iou = round(float(np.mean(mious)), 3)
    pixel_acc = round(float(np.mean(pas)), 3)
    avg_loss = round(float(np.mean(losses)), 3)
    print('Average loss:{}, Mean IoU:{}, Pixel accuracy:{}'.format(avg_loss, mean_iou, pixel_acc))
    if save_dir is None:
        return avg_loss, (mean_iou, pixel_acc)
    else:
        from ptflops import get_model_complexity_info
        image = (batch_data_shape[1], batch_data_shape[2], batch_data_shape[3])
        GFLOPs, Parameters = get_model_complexity_info(net.cuda(), image, as_strings=True, print_per_layer_stat=False,
                                                       verbose=False)
        save_dict = {}
        save_dict.setdefault('GFLOPs', GFLOPs)
        save_dict.setdefault('Parameters', Parameters)
        save_dict.setdefault('Average loss', avg_loss)
        save_dict.setdefault('Mean IoU', mean_iou)
        save_dict.setdefault('Pixel accuracy', pixel_acc)
        with open(save_dir + '/metrics.json', 'w') as f:
            import json
            json.dump(save_dict, f, indent=2)


def predict_images(net, args, dst_size=(256, 256), save_dir=None):
    if not args.test_images:
        print('Test image path is not specific!')
        return

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    pred_dicts = []

    times = []
    paths = [i for i in Path(args.test_images).glob('*.jpg')]
    for path in paths:
        frame = cv2.imread(str(path))
        start = time.time()

        img_transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
        img_tensor = img_transform(frame).unsqueeze(0)
        prediction_np, _, _ = predict_a_batch(net, args.out_channels, img_tensor, class_weights=None, batch_label=None,
                                              do_criterion=False, do_metric=False)
        prediction_np = prediction_np.astype('uint8')
        if args.erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.erode, args.erode))
            prediction_np = cv2.erode(prediction_np, kernel)
        dst_frame = cv2.resize(frame, dst_size)
        dst_prediction = cv2.resize(prediction_np, dst_size)
        # print(dst_prediction.shape)

        # pred_dict = Counter(dst_prediction.flatten())
        # pred_dicts.append(pred_dict)
        # print(pred_dict)

        # dst_frame = np.zeros_like(dst_frame)
        # # dst_show = add_mask_to_source_multi_classes(dst_frame, dst_prediction, args.out_channels)
        dst_show = vis_pred(dst_prediction, args.out_channels)

        torch.cuda.synchronize()
        end = time.time()
        cost_time = end - start
        # times.append(cost_time)
        print('Processed image:{}\t\tTime:{}'.format(path.name, cost_time))
        if save_dir is not None:
            cv2.imwrite(save_dir + '/' + path.name[:-4] + '_' + args.pt_dir + '.jpg' + path.name, dst_show)
        else:
            plt.imshow(dst_show)
            plt.pause(0.5)
        
        # break

    # if save_dir is not None:
    #     with open(save_dir + '/pred_dicts-' + args.pt_dir + '-.json', 'w') as f:
    #         json.dump(pred_dicts, f, indent=2)
