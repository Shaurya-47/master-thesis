#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import torch
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
    
###############################################################################    

###############                  Experiments                 ##################

###############################################################################
 
    # hidden layer information extraction - using pretrained model
    
    # set device
    device = 'cuda' # or 'cpu'
    
    ##### HELPER FUNCTION FOR FEATURE EXTRACTION
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    ##### REGISTER HOOK
    # extracting the hidden layer features for the last EdgeConv layer
    model.module.conv9.register_forward_hook(get_features('feats'))

    ##### FEATURE EXTRACTION LOOP - batch wise
    
    # placeholders
    PREDS = []
    FEATS = []
    PART_LABELS = []
    LABELS = []
    
    # placeholder for batch features
    features = {}
    
    for data, label, seg in test_loader:
        #seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        # forward pass [with feature extraction]
        preds = model(data, label_one_hot)
    
        # add feats and preds to lists
        PREDS.append(preds.detach().cpu().numpy())
        FEATS.append(features['feats'].cpu().numpy())
        
        # add part labels and parent labels to list
        PART_LABELS.append(seg.detach().cpu().numpy())
        LABELS.append(label.detach().cpu().numpy())

##############################################################################

    ##### FEATURE EXTRACTION LOOP - if we only have one batch
    
    # # placeholders
    # PREDS = []
    # FEATS = []
        
    # # placeholder for batch features
    # features = {}
           
    # # forward pass [with feature extraction]
    # preds = model(test_pts_small)
        
    # # add feats and preds to lists
    # PREDS.append(preds.detach().cpu().numpy())
    # FEATS.append(features['feats'].cpu().numpy())

##############################################################################

    ##### INSPECT FEATURES
    
    io.cprint(str(type(PREDS)))
    io.cprint(str(type(FEATS)))
    io.cprint(str(type(PART_LABELS)))
    io.cprint(str(type(LABELS)))
    
    PREDS = np.concatenate(PREDS)
    FEATS = np.concatenate(FEATS)
    PART_LABELS = np.concatenate(PART_LABELS)
    LABELS = np.concatenate(LABELS)
    
    io.cprint(str(PREDS.shape))
    io.cprint(str(FEATS.shape))
    io.cprint(str(PART_LABELS.shape))
    io.cprint(str(LABELS.shape))
    
    # converting to pytorch tensor
    #PREDS = torch.from_numpy(PREDS)
    #FEATS = torch.from_numpy(FEATS)
    #PART_LABELS = torch.from_numpy(PART_LABELS)
    #LABELS = torch.from_numpy(LABELS)

    # saving locally
    # torch.save(PREDS, 'outputs/%s/test_subset_predictions.pt' % args.exp_name)
    # torch.save(FEATS, 'outputs/%s/test_subset_conv9_hidden_output.pt' % args.exp_name)
    # torch.save(PART_LABELS, 'outputs/%s/test_subset_part_labels.pt' % args.exp_name)
    # torch.save(LABELS, 'outputs/%s/test_subset_labels.pt' % args.exp_name)
    
##############################################################################
    
    # reduce the number of points in the test set and save all related tensors

    # creating a subset with 10 airplanes and 10 chairs (Boolean mask)
    
    # getting all arplane and chair prediction indices
    indices_airplane = np.array(LABELS == 0).flatten()
    indices_bag = np.array(LABELS == 1).flatten()
    indices_cap = np.array(LABELS == 2).flatten()
    indices_car = np.array(LABELS == 3).flatten()
    indices_chair = np.array(LABELS == 4).flatten()
    indices_earphone = np.array(LABELS == 5).flatten()
    indices_guitar = np.array(LABELS == 6).flatten()
    indices_knife = np.array(LABELS == 7).flatten()
    indices_lamp = np.array(LABELS == 8).flatten()
    indices_laptop = np.array(LABELS == 9).flatten()
    indices_motorbike = np.array(LABELS == 10).flatten()
    indices_mug = np.array(LABELS == 11).flatten()
    indices_pistol = np.array(LABELS == 12).flatten()
    indices_rocket = np.array(LABELS == 13).flatten()
    indices_skateboard = np.array(LABELS == 14).flatten()
    indices_table = np.array(LABELS == 15).flatten()
    
    ##### predictions
    
    # predictions - all airplanes and chairs
    airplane_preds = PREDS[indices_airplane]
    bag_preds = PREDS[indices_bag]
    cap_preds = PREDS[indices_cap]
    car_preds = PREDS[indices_car]
    chair_preds = PREDS[indices_chair]
    earphone_preds = PREDS[indices_earphone]
    guitar_preds = PREDS[indices_guitar]
    knife_preds = PREDS[indices_knife]
    lamp_preds = PREDS[indices_lamp]
    laptop_preds = PREDS[indices_laptop]
    motorbike_preds = PREDS[indices_motorbike]
    mug_preds = PREDS[indices_mug]
    pistol_preds = PREDS[indices_pistol]
    rocket_preds = PREDS[indices_rocket]
    skateboard_preds = PREDS[indices_skateboard]
    table_preds = PREDS[indices_table]
    io.cprint(str((airplane_preds.shape)))
    io.cprint(str((chair_preds.shape)))
        
    # predictions - subset of 10 airplanes and 10 chairs
    airplane_preds_subset = airplane_preds[0:100]
    bag_preds_subset = bag_preds[0:100]
    cap_preds_subset = cap_preds[0:100]
    car_preds_subset = car_preds[0:100]
    chair_preds_subset = chair_preds[0:100]
    earphone_preds_subset = earphone_preds[0:100]
    guitar_preds_subset = guitar_preds[0:100]
    knife_preds_subset = knife_preds[0:100]
    lamp_preds_subset = lamp_preds[0:100]
    laptop_preds_subset = laptop_preds[0:100]
    motorbike_preds_subset = motorbike_preds[0:100]
    mug_preds_subset = mug_preds[0:100]
    pistol_preds_subset = pistol_preds[0:100]
    rocket_preds_subset = rocket_preds[0:100]
    skateboard_preds_subset = skateboard_preds[0:100]
    table_preds_subset = table_preds[0:100]
    io.cprint(str(airplane_preds_subset.shape))
    io.cprint(str(chair_preds_subset.shape))

    ##### hidden output

    # hidden - all airplanes and chairs
    airplane_conv9_hidden_output = FEATS[indices_airplane]
    bag_conv9_hidden_output = FEATS[indices_bag]
    cap_conv9_hidden_output = FEATS[indices_cap]
    car_conv9_hidden_output = FEATS[indices_car]
    chair_conv9_hidden_output = FEATS[indices_chair]
    earphone_conv9_hidden_output = FEATS[indices_earphone]
    guitar_conv9_hidden_output = FEATS[indices_guitar]
    knife_conv9_hidden_output = FEATS[indices_knife]
    lamp_conv9_hidden_output = FEATS[indices_lamp]
    laptop_conv9_hidden_output = FEATS[indices_laptop]
    motorbike_conv9_hidden_output = FEATS[indices_motorbike]
    mug_conv9_hidden_output = FEATS[indices_mug]
    pistol_conv9_hidden_output = FEATS[indices_pistol]
    rocket_conv9_hidden_output = FEATS[indices_rocket]
    skateboard_conv9_hidden_output = FEATS[indices_skateboard]
    table_conv9_hidden_output = FEATS[indices_table]
    io.cprint(str((airplane_conv9_hidden_output.shape)))
    io.cprint(str((chair_conv9_hidden_output.shape)))
    
    # hidden - subset of 10 airplanes and 10 chairs
    airplane_conv9_hidden_output_subset = airplane_conv9_hidden_output[0:100]
    bag_conv9_hidden_output_subset = bag_conv9_hidden_output[0:100]
    cap_conv9_hidden_output_subset = cap_conv9_hidden_output[0:100]
    car_conv9_hidden_output_subset = car_conv9_hidden_output[0:100]
    chair_conv9_hidden_output_subset = chair_conv9_hidden_output[0:100]
    earphone_conv9_hidden_output_subset = earphone_conv9_hidden_output[0:100] 
    guitar_conv9_hidden_output_subset = guitar_conv9_hidden_output[0:100][0:100]
    knife_conv9_hidden_output_subset = knife_conv9_hidden_output[0:100]
    lamp_conv9_hidden_output_subset = lamp_conv9_hidden_output[0:100]
    laptop_conv9_hidden_output_subset = laptop_conv9_hidden_output[0:100]
    motorbike_conv9_hidden_output_subset = motorbike_conv9_hidden_output[0:100]
    mug_conv9_hidden_output_subset = mug_conv9_hidden_output[0:100]
    pistol_conv9_hidden_output_subset = pistol_conv9_hidden_output[0:100]
    rocket_conv9_hidden_output_subset = rocket_conv9_hidden_output[0:100]
    skateboard_conv9_hidden_output_subset = skateboard_conv9_hidden_output[0:100]
    table_conv9_hidden_output_subset = table_conv9_hidden_output[0:100]
    io.cprint(str(airplane_conv9_hidden_output_subset.shape))
    io.cprint(str(chair_conv9_hidden_output_subset.shape))

    ##### part labels

    # part labels - all airplanes and chairs
    airplane_part_labels = PART_LABELS[indices_airplane]
    bag_part_labels = PART_LABELS[indices_bag]
    cap_part_labels = PART_LABELS[indices_cap]
    car_part_labels = PART_LABELS[indices_car]
    chair_part_labels = PART_LABELS[indices_chair]
    earphone_part_labels = PART_LABELS[indices_earphone]
    guitar_part_labels = PART_LABELS[indices_guitar]
    knife_part_labels = PART_LABELS[indices_knife]
    lamp_part_labels = PART_LABELS[indices_lamp]
    laptop_part_labels = PART_LABELS[indices_laptop]
    motorbike_part_labels = PART_LABELS[indices_motorbike]
    mug_part_labels = PART_LABELS[indices_mug]
    pistol_part_labels = PART_LABELS[indices_pistol]
    rocket_part_labels = PART_LABELS[indices_rocket]
    skateboard_part_labels = PART_LABELS[indices_skateboard]
    table_part_labels = PART_LABELS[indices_table]
    io.cprint(str((airplane_part_labels.shape)))
    io.cprint(str((chair_part_labels.shape)))
    
    # part labels - subset of 10 airplanes and 10 chairs
    airplane_part_labels_subset = airplane_part_labels[0:100]
    bag_part_labels_subset = bag_part_labels[0:100]
    cap_part_labels_subset = cap_part_labels[0:100]
    car_part_labels_subset = car_part_labels[0:100]
    chair_part_labels_subset = chair_part_labels[0:100]
    earphone_part_labels_subset = earphone_part_labels[0:100]
    guitar_part_labels_subset = guitar_part_labels[0:100]
    knife_part_labels_subset = knife_part_labels[0:100]
    lamp_part_labels_subset = lamp_part_labels[0:100]
    laptop_part_labels_subset = laptop_part_labels[0:100]
    motorbike_part_labels_subset = motorbike_part_labels[0:100]
    mug_part_labels_subset = mug_part_labels[0:100]
    pistol_part_labels_subset = pistol_part_labels[0:100]
    rocket_part_labels_subset = rocket_part_labels[0:100]
    skateboard_part_labels_subset = skateboard_part_labels[0:100]
    table_part_labels_subset = table_part_labels[0:100]
    io.cprint(str(airplane_part_labels_subset.shape))
    io.cprint(str(chair_part_labels_subset.shape))

    io.cprint('Shape counts')
    io.cprint(str(airplane_part_labels_subset.shape))
    io.cprint(str(bag_part_labels_subset.shape))
    io.cprint(str(cap_part_labels_subset.shape))
    io.cprint(str(car_part_labels_subset.shape))
    io.cprint(str(chair_part_labels_subset.shape))
    io.cprint(str(earphone_part_labels_subset.shape))
    io.cprint(str(guitar_part_labels_subset.shape))
    io.cprint(str(knife_part_labels_subset.shape))
    io.cprint(str(lamp_part_labels_subset.shape))
    io.cprint(str(laptop_part_labels_subset.shape))
    io.cprint(str(motorbike_part_labels_subset.shape))
    io.cprint(str(mug_part_labels_subset.shape))
    io.cprint(str(pistol_part_labels_subset.shape))
    io.cprint(str(rocket_part_labels_subset.shape))
    io.cprint(str(skateboard_part_labels_subset.shape))
    io.cprint(str(table_part_labels_subset.shape))
    
    torch.save(airplane_preds_subset, 'outputs/%s/airplane_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(bag_preds_subset, 'outputs/%s/bag_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(cap_preds_subset, 'outputs/%s/cap_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(car_preds_subset, 'outputs/%s/car_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(chair_preds_subset, 'outputs/%s/chair_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(earphone_preds_subset, 'outputs/%s/earphone_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(guitar_preds_subset, 'outputs/%s/guitar_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(knife_preds_subset, 'outputs/%s/knife_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(lamp_preds_subset, 'outputs/%s/lamp_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(laptop_preds_subset, 'outputs/%s/laptop_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(motorbike_preds_subset, 'outputs/%s/motorbike_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(mug_preds_subset, 'outputs/%s/mug_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(pistol_preds_subset, 'outputs/%s/pistol_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(rocket_preds_subset, 'outputs/%s/rocket_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(skateboard_preds_subset, 'outputs/%s/skateboard_test_subset_predictions_big_dataset.pt' % args.exp_name)
    torch.save(table_preds_subset, 'outputs/%s/table_test_subset_predictions_big_dataset.pt' % args.exp_name)
    
    torch.save(airplane_conv9_hidden_output_subset, 'outputs/%s/airplane_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(bag_conv9_hidden_output_subset, 'outputs/%s/bag_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(cap_conv9_hidden_output_subset, 'outputs/%s/cap_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(car_conv9_hidden_output_subset, 'outputs/%s/car_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(chair_conv9_hidden_output_subset, 'outputs/%s/chair_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(earphone_conv9_hidden_output_subset, 'outputs/%s/earphone_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(guitar_conv9_hidden_output_subset, 'outputs/%s/guitar_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(knife_conv9_hidden_output_subset, 'outputs/%s/knife_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(lamp_conv9_hidden_output_subset, 'outputs/%s/lamp_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(laptop_conv9_hidden_output_subset, 'outputs/%s/laptop_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(motorbike_conv9_hidden_output_subset, 'outputs/%s/motorbike_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(mug_conv9_hidden_output_subset, 'outputs/%s/mug_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(pistol_conv9_hidden_output_subset, 'outputs/%s/pistol_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(rocket_conv9_hidden_output_subset, 'outputs/%s/rocket_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(skateboard_conv9_hidden_output_subset, 'outputs/%s/skateboard_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    torch.save(table_conv9_hidden_output_subset, 'outputs/%s/table_test_subset_conv9_hidden_output_big_dataset.pt' % args.exp_name)
    
    torch.save(airplane_part_labels_subset, 'outputs/%s/airplane_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(bag_part_labels_subset, 'outputs/%s/bag_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(cap_part_labels_subset, 'outputs/%s/cap_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(car_part_labels_subset, 'outputs/%s/car_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(chair_part_labels_subset, 'outputs/%s/chair_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(earphone_part_labels_subset, 'outputs/%s/earphone_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(guitar_part_labels_subset, 'outputs/%s/guitar_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(knife_part_labels_subset, 'outputs/%s/knife_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(lamp_part_labels_subset, 'outputs/%s/lamp_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(laptop_part_labels_subset, 'outputs/%s/laptop_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(motorbike_part_labels_subset, 'outputs/%s/motorbike_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(mug_part_labels_subset, 'outputs/%s/mug_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(pistol_part_labels_subset, 'outputs/%s/pistol_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(rocket_part_labels_subset, 'outputs/%s/rocket_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(skateboard_part_labels_subset, 'outputs/%s/skateboard_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    torch.save(table_part_labels_subset, 'outputs/%s/table_test_subset_part_labels_big_dataset.pt' % args.exp_name)
    
    # saving in parts as size is too big 
    #chair_conv9_hidden_output_subset_part_1 = chair_conv9_hidden_output_subset[:400]
    #chair_conv9_hidden_output_subset_part_2 = chair_conv9_hidden_output_subset[400:]
    #table_conv9_hidden_output_subset_part_1 = table_conv9_hidden_output_subset[:400]
    #table_conv9_hidden_output_subset_part_2 = table_conv9_hidden_output_subset[400:]
    
    #torch.save(chair_conv9_hidden_output_subset_part_1, 'outputs/%s/chair_test_subset_conv9_hidden_output_big_dataset_part_1.pt' % args.exp_name)
    #torch.save(chair_conv9_hidden_output_subset_part_2, 'outputs/%s/chair_test_subset_conv9_hidden_output_big_dataset_part_2.pt' % args.exp_name)
    #torch.save(table_conv9_hidden_output_subset_part_1, 'outputs/%s/table_test_subset_conv9_hidden_output_big_dataset_part_1.pt' % args.exp_name)
    #torch.save(table_conv9_hidden_output_subset_part_2, 'outputs/%s/table_test_subset_conv9_hidden_output_big_dataset_part_2.pt' % args.exp_name)

    
##############################################################################

    
    # Shape analysis
    
    # 6 - number of test points
    # 64 - 64 separate edge convolutions applied - max operator applied and single number stored in the end
    # 1024 - number of points in a single test example point cloud
    # 20 - chosen k for KNN
    
    # In words: for each point (out of 1024) in a test point cloud (out of 6), we construct the KNN (20) graph, and then get a 64-dimensional edge 
    # representation for each NN by applying the Edge Conv operation
    
    # After that (not done till this stage, but can be seen in code block 7, we take the max edge and only keep that, thus droping the dimension k)
    
    # # getting it as a Torch tensor
    # hidden = torch.from_numpy(FEATS)
    
    # # sending to cuda/cpu
    # hidden = hidden.cuda() # cuda
    
    # # getting it in the same format as the model
    # hidden = hidden.type('torch.cuda.FloatTensor')#test_pts_small.type('torch.cuda.FloatTensor')
    
    # # Inspecting the shape again
    # io.cprint(hidden.size())
    
    # # inspecting the values
    # io.cprint(hidden[0][:][:][:].size())
    
    # io.cprint(hidden[0][0][0][:].size())
    
    # # One EdgeConv output (out of 64 of them) - 20 NNs of a single point
    # io.cprint(hidden[0][0][0][:])
    # # Edge feature value 1 (out of 64) for all 20 neighbours - only max is kept at the end (step after conv) - THIS IS THE EDGE EMBEDDING
    
    # io.cprint(hidden[0][63][0][:])
    # # Edge feature value 64 (out of 64)
    



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
