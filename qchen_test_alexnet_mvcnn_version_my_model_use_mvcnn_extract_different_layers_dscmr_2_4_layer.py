import datetime
import sys

from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from datasets import *
from utils import meter
from models import *
n_output=40
#pytorch version

# we get the top5 as one feature
#we get the lowest 5 as seconde feature
# then we use dscmr +contrastive center loss to train a better feature repsentative.
#add by qchen 2021.4.19 .hope it works.
#
# This is the code for the paper submit to MTAP.
# Multi-view 3D model retrieval based on enhanced detail features with contrastive center loss
# Qiang CHEN1, Yinong CHEN2*
# 1 College of Computer and Information Science, Southwest University, Chongqing, 400175, China
# 2 School of Computing, Informatics and Decision Systems Engineering, Arizona State University, Tempe, AZ, USA
# May 13,2021

from center_loss import CenterLoss
from models.MVCNN import BaseClassifierNet
from my_contrastive_center_loss import MyContrastiveCenterLoss
class MVCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(MVCNN, self).__init__()
        base_model_name = config.base_model_name
        num_classes = config.view_net.num_classes
        print(f'\ninit {base_model_name} model...\n')
        self.features = BaseFeatureNet(base_model_name, pretrained)
        self.classifier = BaseClassifierNet(base_model_name, num_classes, pretrained)

        #self.classifier_output1 = nn.Linear(4096, 256)

        #self.classifier_output2 = nn.Linear(256, n_output)
        #self.conv2d1 = conv_2d(5, 1, 1)

    def forward(self, x):
        batch_sz = x.size(0)

        mvcnn_last_fc, x_view = self.features(x)  #mvcnn_last_fc 4096
        mvcnn_preds= self.classifier(mvcnn_last_fc)  # output 40

        total_layers=12
        #second_view
        x_sort, _ = torch.topk(x_view, total_layers, 1)
        fc2=x_sort[:,1,:]
        preds2 = self.classifier(fc2)  # 256

        #3rd_view

        fc3=x_sort[:,2,:]
        preds3 = self.classifier(fc3)  # 256

        #4th_view

        fc4=x_sort[:,3,:]
        preds4 = self.classifier(fc4)  # 256

        #5th_view

        fc5=x_sort[:,4,:]
        preds5 = self.classifier(fc5)  # 256

        # 6th_view

        fc6 = x_sort[:, 5, :]
        preds6 = self.classifier(fc6)  # 256

        # 7th_view

        fc7 = x_sort[:, 6, :]
        preds7 = self.classifier(fc7)  # 256

        # 8th_view

        fc8 = x_sort[:, 7, :]
        preds8 = self.classifier(fc8)  # 256

        # 9th_view

        fc9 = x_sort[:, 8, :]
        preds9 = self.classifier(fc9)  # 256

        # 10th_view

        fc10 = x_sort[:, 9, :]
        preds10 = self.classifier(fc10)  # 256


        # 11th_view

        fc11 = x_sort[:, 10, :]
        preds11 = self.classifier(fc11)  # 256


        # 12th_view

        fc12 = x_sort[:, 11, :]
        preds12 = self.classifier(fc12)  # 256

        return mvcnn_preds,preds2,preds3,preds4,preds5,preds6,preds7,preds8,preds9,preds10,preds11,preds12,mvcnn_last_fc,fc2,fc3,fc4,fc5,fc6,fc7,fc8,fc9,fc10,fc11,fc12

input_dim=4096
class DSCMR(nn.Module):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=256, img_output_dim=2048,
                 text_input_dim=256, text_output_dim=2048, minus_one_dim=2048, output_dim=40):
        super(DSCMR, self).__init__()
        self.denseL1 = nn.Linear(input_dim, 2048)
        self.img_net = MVCNN()
        #self.dgcnn_net=DGCNN(n_neighbor=config.pc_net.n_neighbor, num_classes=config.pc_net.num_classes)

        for p in self.img_net.parameters():
            p.requires_grad = False

        #for p in self.dgcnn_net.parameters():  #no need to update
         #   p.requires_grad = False

        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

    def forward(self, views):

        mvcnn_preds, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, mvcnn_last_fc, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10, fc11, fc12 = self.img_net(
            views)
        fc_feature1 = F.relu(self.denseL1(fc2))
        fc_feature2 = F.relu(self.denseL1(fc4))

        fc_feature1 = self.linearLayer(fc_feature1)
        fc_feature2 = self.linearLayer(fc_feature2)

        predict1 = self.linearLayer2(fc_feature1)
        predict2 = self.linearLayer2(fc_feature2)
        return predict1, predict2,fc_feature1, fc_feature2


def validate(val_loader, net, epoch):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec1 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec2 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec3 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec4 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec5 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec6 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec7 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec8 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec9 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec10 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec11 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec12 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    prec10_12 = meter.ClassErrorMeter(topk=[1], accuracy=True)
    retrieval_map = meter.RetrievalMAPMeter()
    retrieval1 = meter.RetrievalMAPMeter()
    retrieval2 = meter.RetrievalMAPMeter()
    retrieval3 = meter.RetrievalMAPMeter()
    retrieval4 = meter.RetrievalMAPMeter()
    retrieval5 = meter.RetrievalMAPMeter()

    # testing mode
    net.eval()

    for i, (views, labels,viewlist) in enumerate(val_loader):
        batch_time.reset()
        # bz x 12 x 3 x 224 x 224
        views = views.to(device=config.device)
        labels = labels.to(device=config.device)

        #view1_predict, view2_predict, view1_feature, view2_feature = net(views)
        preds1, preds2,fc_feature1, fc_feature2= net(views)
        #mvcnn_preds, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, mvcnn_last_fc, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10, fc11, fc12= net(views)

        #output_feature = torch.cat((mvcnn_last_fc, fc2,fc3,fc4,fc5, fc6,fc7,fc8,fc9,fc10,fc11,fc12), 1)
        output_feature = torch.cat((fc_feature1, fc_feature2), 1)

        #prec.add(mvcnn_preds+preds2+preds3+preds4+preds5, labels.data)
        prec.add(preds1 + preds2 , labels.data)

        prec1.add(preds1.data, labels.data)

        prec2.add(preds2.data, labels.data)
        # prec3.add(preds3.data, labels.data)
        # prec4.add(preds4.data, labels.data)
        # prec5.add(preds5.data, labels.data)
        # prec6.add(preds6.data, labels.data)
        # prec7.add(preds7.data, labels.data)
        # prec8.add(preds8.data, labels.data)
        # prec9.add(preds9.data, labels.data)
        # prec10.add(preds10.data, labels.data)
        # prec11.add(preds11.data, labels.data)
        # prec12.add(preds12.data, labels.data)
        # prec10_12.add(preds12.data+preds11.data+preds10.data, labels.data)

        retrieval_map.add5(output_feature.detach() / torch.norm(output_feature.detach(), 2, 1, True), labels.detach(),viewlist,(preds1 + preds2).data)
        retrieval1.add5(fc_feature1.detach() / torch.norm(fc_feature1.detach(), 2, 1, True), labels.detach(),viewlist,preds1.data)

        retrieval2.add5(fc_feature2.detach() / torch.norm(fc_feature2.detach(), 2, 1, True), labels.detach(),viewlist,preds2.data)
        # retrieval3.add(fc3.detach() / torch.norm(fc3.detach(), 2, 1, True), labels.detach())
        # retrieval4.add(fc4.detach() / torch.norm(fc4.detach(), 2, 1, True), labels.detach())
        # retrieval5.add(fc5.detach() / torch.norm(fc5.detach(), 2, 1, True), labels.detach())

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@all {prec.value(1):.3f}\t'
                  f'Prec@2 {prec1.value(1):.3f}\t'
                  f'Prec@4 {prec2.value(1):.3f}\t')

    matname='mvcnn_different_layer_test_dscmr_2_4_'+str(epoch)+'.mat'
    if prec.value(1) > 60:
        mAP = retrieval_map.mAP(matname)
        mAP1 = retrieval1.mAP('_t_mvcnn_xxxx.mat')
        mAP2= retrieval2.mAP('_t_second_xxxx.mat')
        #mAP3 = retrieval3.mAP('_t_second_xxxx.mat')
        #mAP4 = retrieval4.mAP('_t_second_xxxx.mat')
        #mAP5 = retrieval5.mAP('_t_second_xxxx.mat')

        print(f'map_all at epoch : {mAP} ')
        print(f'map_F2 at epoch : {mAP1} ')
        print(f'map_F4 at epoch : {mAP2} ')
        #print(f'map_3 at epoch : {mAP3} ')
        #print(f'map_4 at epoch : {mAP4} ')
        #print(f'map_5 at epoch : {mAP5} ')

    # print(
    #     f'mean class accuracy at epoch {epoch}: all {prec.value(1)}  mvcnn {prec_mvcnn.value(1)} @2 {prec2.value(1)} '
    #     f'@3 {prec3.value(1)} @4 {prec4.value(1)} @5 {prec5.value(1)}'
    #     f'@6 {prec6.value(1)} @7 {prec7.value(1)} @8 {prec8.value(1)}'
    #     f'@9 {prec9.value(1)} @10 {prec10.value(1)} @11 {prec11.value(1)} '
    #     f'@12 {prec12.value(1)} @10_12 {prec10_12.value(1)}')

    print(
        f'mean class accuracy at epoch {epoch}: all {prec.value(1)}  @F2 {prec1.value(1)} @F4 {prec2.value(1)} ')
    return prec.value(1)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def save_record(epoch, prec1, net: nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict, osp.join(config.view_net.ckpt_record_folder, f'epoch{epoch}_{prec1:.2f}.pth'))


def save_ckpt(epoch, best_prec1, net, optimizer, training_conf=config.view_net):
    ckpt = dict(
        epoch=epoch,
        best_prec1=best_prec1,
        model=net.state_dict(),
        optimizer=optimizer.state_dict(),
        training_conf=training_conf
        )
    #torch.save(ckpt, 'mvcnn_different_layer_test_2_4_dscmr_'+str(epoch)+'.ckpt.pth')


if __name__ == '__main__':
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].


    config.init_env()

    train_dataset = data_pth.view_data(config.view_net.data_root,
                                       status=STATUS_TRAIN,
                                       base_model_name=config.base_model_name)
    train_loader = DataLoader(train_dataset, batch_size=config.view_net.train.batch_sz,
                              num_workers=config.num_workers, shuffle=True)


    val_dataset = data_pth.view_data(config.view_net.data_root,
                                     status=STATUS_TEST,
                                     base_model_name=config.base_model_name)
    val_loader = DataLoader(val_dataset, batch_size=config.view_net.train.batch_sz,
                            num_workers=config.num_workers,shuffle=True)
    #dataiter = iter(train_loader)
    #images, labels = dataiter.next()

    #sys.exit(0)

    import torch.optim as optim


    #available_gpus = "0"      #multi-gpus "0,1"
    #os.environ['CUDA_VISIBLE_DEVICES'] = available_gpus  # available_gpus
    config.init_env()

    dscmr_net=DSCMR()
    print(dscmr_net)
    dscmr_net = torch.nn.DataParallel(dscmr_net)
    dscmr_net = dscmr_net.to(device=config.device)
    optimizer_dscmr = optim.Adam(dscmr_net.parameters(), config.pc_net.train.lr,
                                 weight_decay=config.pc_net.train.weight_decay)
    lr_scheduler_dscmr = torch.optim.lr_scheduler.StepLR(optimizer_dscmr, 20, 0.7)

    criterion_dscmr = nn.CrossEntropyLoss()
    criterion_dscmr = criterion_dscmr.to(device=config.device)

    resume_epoch=0


    # print(f'loading pretrained model from {config.view_net.ckpt_load_file}')
    # checkpoint = torch.load(config.view_net.ckpt_load_file)
    # dscmr_net.module.img_net.load_state_dict(checkpoint['model'],False)
    # best_prec1 = checkpoint['best_prec1']

    print(f'loading pretrained model from mvcnn_different_layer_test_2_4_dscmr_18.ckpt.pth')
    checkpoint = torch.load('mvcnn_different_layer_test_2_4_dscmr_18.ckpt.pth')
    dscmr_net.load_state_dict(checkpoint['model'],True)


    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.
    with torch.no_grad():
       prec1 = validate(val_loader, dscmr_net, 999)
    sys.exit()

    center_loss = MyContrastiveCenterLoss(num_classes=40, feat_dim=4096, use_gpu=True)
    optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=0.0001)

    dscmr_net.train()

    best_prec1=0
    if(best_prec1 is None):
        best_prec1=0
    resume_epoch=0

    for epoch in range(500):  # loop over the dataset multiple times

        running_loss = 0.0
        #if epoch >= 5:
        #    for p in net.parameters():
        #        p.requires_grad = True
        #lr_scheduler.step(epoch=epoch)

        prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
        prec_view1 = meter.ClassErrorMeter(topk=[1], accuracy=True)
        prec_view2 = meter.ClassErrorMeter(topk=[1], accuracy=True)
        #for i, data in enumerate(train_loader, 0):
        for i, (views, labels,viewlist) in enumerate(train_loader):
            # get the inputs
            labels = labels.to(device=config.device)

            #pcs = pcs.to(device=config.device)
            inputs_views = views.to(config.device)

            view1_predict, view2_predict, view1_feature, view2_feature = dscmr_net(inputs_views)

            output_feature = torch.cat((view1_feature, view2_feature), 1)

            prec.add(view1_predict.data + view2_predict.data, labels.data)

            prec_view1.add(view1_predict.data, labels.data)
            prec_view2.add(view2_predict.data, labels.data)

            loss_view2=criterion_dscmr(view2_predict, labels)
            loss_view1 = criterion_dscmr(view1_predict, labels)

            #loss_mse_dgcnn=mse_criterion(view1_feature, view2_feature)
            #loss_mse_img = mse_criterion(view1_predict, labels)

            #prec_dgcnn.add(view2_predict.data, labels.data)
            optimizer_dscmr.zero_grad()
            loss=loss_view1+loss_view2

            #alpha=0.01
            alpha=1
            #c_loss = center_loss(view1_predict.data + view2_predict.data, labels) * alpha + loss+0.1*(loss_mse_dgcnn+loss_mse_img)
            c_loss = center_loss(torch.cat((view1_feature,view2_feature),dim=1), labels) * alpha + loss# + 0.1 * (
                        #loss_mse_dgcnn )

            optimizer_centloss.zero_grad()

            c_loss.backward()
            for param in center_loss.parameters():
                param.grad.data *= (1. / alpha)
            optimizer_centloss.step()

            #loss.backward()
            #backward()
            optimizer_dscmr.step()
            # for param in center_loss.parameters():
            #     param.grad.data *= (1. / alpha)
            # optimizer_centloss.step()

            # print statistics
            running_loss += loss.item()
            if i % 15 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f acc_all: %.3f acc_F2 : %.3f  acc_F4 : %.3f %s' %
                      (epoch + 1, i + 1, running_loss / 15, prec.value(1), prec_view1.value(1),prec_view2.value(1),datetime.datetime.now()))
                running_loss = 0.0

        with torch.no_grad():
            prec1 = validate(val_loader, dscmr_net, epoch)

        #if (epoch %20==0 and epoch > 0 ) or epoch==499:


        if(prec1>best_prec1 and prec1>90):
            best_prec1=prec1
            save_ckpt(epoch + resume_epoch, best_prec1, dscmr_net, optimizer_dscmr)

        print('Best accuracy is ',best_prec1)


    # data_input = Variable(torch.randn([32, 3, 224, 224])) # input size
    # print(data_input.size())
    # net(data_input)

    print(dscmr_net)
    print(get_parameter_number(dscmr_net))
    #from torchsummary import summary
    #summary(your_model, input_size=(channels, H, W))
    #summary(net,(3,227,227))
