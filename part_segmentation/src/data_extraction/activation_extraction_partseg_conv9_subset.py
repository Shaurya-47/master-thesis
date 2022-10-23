from __future__ import print_function
import os
import argparse
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart
from model import DGCNN_partseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement

global class_cnts
class_indexs = np.zeros((16,), dtype=int)
global visual_warning
visual_warning = True

class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'):
        os.makedirs('outputs/'+args.exp_name+'/'+'visualization')
    os.system('cp main_partseg.py outputs'+'/'+args.exp_name+'/'+'main_partseg.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, visual=False):
    if not visual:
        label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def visualization(visu, visu_format, data, pred, seg, label, partseg_colors, class_choice):
    global class_indexs
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        classname = class_choices[int(label[i])]
        class_index = class_indexs[int(label[i])]
        if visu[0] != 'all':
            if len(visu) != 1:
                if visu[0] != classname or visu[1] != str(class_index):
                    skip = True 
                else:
                    visual_warning = False
            elif visu[0] != classname:
                skip = True 
            else:
                visual_warning = False
        elif class_choice != None:
            skip = True
        else:
            visual_warning = False
        if skip:
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1
        else:  
            if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname):
                os.makedirs('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname)
            for j in range(0, data.shape[2]):
                RGB.append(partseg_colors[int(pred[i][j])])
                RGB_gt.append(partseg_colors[int(seg[i][j])])
            pred_np = []
            seg_np = []
            pred_np.append(pred[i].cpu().numpy())
            seg_np.append(seg[i].cpu().numpy())
            xyz_np = data[i].cpu().numpy()
            xyzRGB = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB_gt)), axis=1)
            IoU = calculate_shape_IoU(np.array(pred_np), np.array(seg_np), label[i].cpu().numpy(), class_choice, visual=True)
            IoU = str(round(IoU[0], 4))
            filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_pred_'+IoU+'.'+visu_format
            filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_gt.'+visu_format
            if visu_format=='txt':
                np.savetxt(filepath, xyzRGB, fmt='%s', delimiter=' ') 
                np.savetxt(filepath_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
                print('TXT visualization file saved in', filepath)
                print('TXT visualization file saved in', filepath_gt)
            elif visu_format=='ply':
                xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath)
                vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath_gt)
                print('PLY visualization file saved in', filepath)
                print('PLY visualization file saved in', filepath_gt)
            else:
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                (visu_format))
                exit()
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1


def train(args, io):
    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, label, seg in train_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in test_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name) # saving model


def test(args, io):
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    partseg_colors = test_loader.dataset.partseg_colors
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device) # model structure loaded here
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path)) # pretrained model weights added to model here
    model = model.eval()                               # model set to evaluate mode
    test_acc = 0.0
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in test_loader:
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
        # visiualization
        visualization(args.visu, args.visu_format, data, pred, seg, label, partseg_colors, args.class_choice) 
    if visual_warning and args.visu != '':
        print('Visualization Failed: You can only choose a point cloud shape to visualize within the scope of the test class')
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_acc,
                                                                             avg_per_class_acc,
                                                                             np.mean(test_ious))
    io.cprint(outstr)
    # saving the model prediction and label tensors
    torch.save(test_pred_seg, 'outputs/%s/test_pred_seg.pt' % args.exp_name)
    torch.save(test_true_seg, 'outputs/%s/test_true_seg.pt' % args.exp_name)
    torch.save(test_label_seg, 'outputs/%s/test_label_seg.pt' % args.exp_name)

#### -------------- hidden layer information extraction ----------------- #####
   
    # set device
    device = 'cuda' # or 'cpu'
    
    # function for feature extraction
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # selecting the neural network layer
    model.module.conv9.register_forward_hook(get_features('feats'))

    # batch-wise feature extraction loop
    
    # placeholders
    preds = []
    features = []
    part_labels = []
    labels = []
    
    # placeholder for batch features
    features_batch = {}
    
    # main loop
    for data_batch, label_batch, seg_batch in test_loader:
        label_one_hot_batch = np.zeros((label_batch.shape[0], 16))
        for idx in range(label_batch.shape[0]):
            label_one_hot_batch[idx, label_batch[idx]] = 1
        label_one_hot_batch = torch.from_numpy(label_one_hot_batch.astype(np.float32))
        data_batch, label_one_hot_batch, seg = data_batch.to(device), label_one_hot_batch.to(device), seg_batch.to(device)
        data_batch = data_batch.permute(0, 2, 1)
        batch_size = data_batch.size()[0]

        # forward pass with feature extraction
        preds_batch = model(data_batch, label_one_hot_batch)
    
        # append batch features, predictions, part and object labels to lists
        preds.append(preds_batch.detach().cpu().numpy())
        features.append(features_batch['feats'].cpu().numpy())
        part_labels.append(seg_batch.detach().cpu().numpy())
        labels.append(label_batch.detach().cpu().numpy())

    # getting object indices using labels
    indices_airplane = np.array(labels == 0).flatten()
    indices_bag = np.array(labels == 1).flatten()
    indices_cap = np.array(labels == 2).flatten()
    indices_car = np.array(labels == 3).flatten()
    indices_chair = np.array(labels == 4).flatten()
    indices_earphone = np.array(labels == 5).flatten()
    indices_guitar = np.array(labels == 6).flatten()
    indices_knife = np.array(labels == 7).flatten()
    indices_lamp = np.array(labels == 8).flatten()
    indices_laptop = np.array(labels == 9).flatten()
    indices_motorbike = np.array(labels == 10).flatten()
    indices_mug = np.array(labels == 11).flatten()
    indices_pistol = np.array(labels == 12).flatten()
    indices_rocket = np.array(labels == 13).flatten()
    indices_skateboard = np.array(labels == 14).flatten()
    indices_table = np.array(labels == 15).flatten()
    
    # selecting 10 examples per object (10 x 16 = 160 overall)
    
    # getting predictions using the indices
    airplane_preds_subset = preds[indices_airplane][0:10]
    bag_preds_subset = preds[indices_bag][0:10]
    cap_preds_subset = preds[indices_cap][0:10]
    car_preds_subset = preds[indices_car][0:10]
    chair_preds_subset = preds[indices_chair][0:10]
    earphone_preds_subset = preds[indices_earphone][0:10]
    guitar_preds_subset = preds[indices_guitar][0:10]
    knife_preds_subset = preds[indices_knife][0:10]
    lamp_preds_subset = preds[indices_lamp][0:10]
    laptop_preds_subset = preds[indices_laptop][0:10]
    motorbike_preds_subset = preds[indices_motorbike][0:10]
    mug_preds_subset = preds[indices_mug][0:10]
    pistol_preds_subset = preds[indices_pistol][0:10]
    rocket_preds_subset = preds[indices_rocket][0:10]
    skateboard_preds_subset = preds[indices_skateboard][0:10]
    table_preds_subset = preds[indices_table][0:10]

    # getting hidden layer outputs using the indices
    airplane_conv9_hidden_output_subset = features[indices_airplane][0:10]
    bag_conv9_hidden_output_subset = features[indices_bag][0:10]
    cap_conv9_hidden_output_subset = features[indices_cap][0:10]
    car_conv9_hidden_output_subset = features[indices_car][0:10]
    chair_conv9_hidden_output_subset = features[indices_chair][0:10]
    earphone_conv9_hidden_output_subset = features[indices_earphone][0:10]
    guitar_conv9_hidden_output_subset = features[indices_guitar][0:10]
    knife_conv9_hidden_output_subset = features[indices_knife][0:10]
    lamp_conv9_hidden_output_subset = features[indices_lamp][0:10]
    laptop_conv9_hidden_output_subset = features[indices_laptop][0:10]
    motorbike_conv9_hidden_output_subset = features[indices_motorbike][0:10]
    mug_conv9_hidden_output_subset = features[indices_mug][0:10]
    pistol_conv9_hidden_output_subset = features[indices_pistol][0:10]
    rocket_conv9_hidden_output_subset = features[indices_rocket][0:10]
    skateboard_conv9_hidden_output_subset = features[indices_skateboard][0:10]
    table_conv9_hidden_output_subset = features[indices_table][0:10]

    # getting part labels using the indices
    airplane_part_labels_subset = part_labels[indices_airplane][0:10]
    bag_part_labels_subset = part_labels[indices_bag][0:10]
    cap_part_labels_subset = part_labels[indices_cap][0:10]
    car_part_labels_subset = part_labels[indices_car][0:10]
    chair_part_labels_subset = part_labels[indices_chair][0:10]
    earphone_part_labels_subset = part_labels[indices_earphone][0:10]
    guitar_part_labels_subset = part_labels[indices_guitar][0:10]
    knife_part_labels_subset = part_labels[indices_knife][0:10]
    lamp_part_labels_subset = part_labels[indices_lamp][0:10]
    laptop_part_labels_subset = part_labels[indices_laptop][0:10]
    motorbike_part_labels_subset = part_labels[indices_motorbike][0:10]
    mug_part_labels_subset = part_labels[indices_mug][0:10]
    pistol_part_labels_subset = part_labels[indices_pistol][0:10]
    rocket_part_labels_subset = part_labels[indices_rocket][0:10]
    skateboard_part_labels_subset = part_labels[indices_skateboard][0:10]
    table_part_labels_subset = part_labels[indices_table][0:10]
    
    

    
    # combining all respective object-wise tensors into single tensors
    
    # hidden output
    conv9_hidden_output_subset = np.vstack((airplane_conv9_hidden_output_subset, 
                                            bag_conv9_hidden_output_subset,
                                            cap_conv9_hidden_output_subset,
                                            car_conv9_hidden_output_subset,
                                            chair_conv9_hidden_output_subset,
                                            earphone_conv9_hidden_output_subset,
                                            guitar_conv9_hidden_output_subset,
                                            knife_conv9_hidden_output_subset,
                                            lamp_conv9_hidden_output_subset,
                                            laptop_conv9_hidden_output_subset,
                                            motorbike_conv9_hidden_output_subset,
                                            mug_conv9_hidden_output_subset,
                                            pistol_conv9_hidden_output_subset,
                                            rocket_conv9_hidden_output_subset,
                                            skateboard_conv9_hidden_output_subset,
                                            table_conv9_hidden_output_subset
                                            ))
    
    # dropping the batch size dimension from the tensor
    conv9_hidden_output_subset = np.moveaxis(conv9_hidden_output_subset, 2, 1)
    conv9_hidden_output_subset = np.resize(conv9_hidden_output_subset, (163840,256))
    
    # predictions
    preds_subset = np.vstack((airplane_preds_subset, 
                              bag_preds_subset,
                              cap_preds_subset,
                              car_preds_subset,
                              chair_preds_subset,
                              earphone_preds_subset,
                              guitar_preds_subset,
                              knife_preds_subset,
                              lamp_preds_subset,
                              laptop_preds_subset,
                              motorbike_preds_subset,
                              mug_preds_subset,
                              pistol_preds_subset,
                              rocket_preds_subset,
                              skateboard_preds_subset,
                              table_preds_subset
                              ))
    
    # dropping the batch size dimension from the tensor
    preds_subset = th.from_numpy(preds_subset)
    preds_subset = preds_subset.permute(0, 2, 1).contiguous()
    preds_subset = preds_subset.max(dim=2)[1]
    preds_subset = np.resize(preds_subset, (163840,1)).flatten()
    
    # part labels
    part_labels_subset = np.vstack((airplane_part_labels_subset, 
                                    bag_part_labels_subset,
                                    cap_part_labels_subset,
                                    car_part_labels_subset,
                                    chair_part_labels_subset,
                                    earphone_part_labels_subset,
                                    guitar_part_labels_subset,
                                    knife_part_labels_subset,
                                    lamp_part_labels_subset,
                                    laptop_part_labels_subset,
                                    motorbike_part_labels_subset,
                                    mug_part_labels_subset,
                                    pistol_part_labels_subset,
                                    rocket_part_labels_subset,
                                    skateboard_part_labels_subset,
                                    table_part_labels_subset
                                    ))
    
    # dropping the batch size dimension from the tensor
    part_labels_subset = np.resize(part_labels_subset, (163840,1)).flatten()
    
    # converting all to pytorch tensors and saving locally    
    torch.save(conv9_hidden_output_subset, 'outputs/%s/conv9_hidden_output_subset.pt' % args.exp_name)
    torch.save(preds_subset, 'outputs/%s/preds_subset.pt' % args.exp_name)
    torch.save(part_labels_subset, 'outputs/%s/part_labels_subset.pt' % args.exp_name)

#### -------------------------------------------------------------------- #####

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)