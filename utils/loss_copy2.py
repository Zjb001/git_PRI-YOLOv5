
'''val loss'''

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
import numpy as np
import cv2 as cv
from skimage import color
import os
import math

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



class ComputeLoss1:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss1, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # BCEcls
        # BCEobj
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        # cn 0.0 cp 1.0

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, indexn,i,imgs):  # predictions, targets, model  # p 0 torch.size([8,3,80,80,85]) 1 torch.size([8,3,40,40,85]) 2 ([8,3,20,20,85])
        #　print(indexn)
        device = targets.device
        lcls1, lcls2, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)  # 用于存放loss的
        # lcls = 0

        Pd = torch.ones_like(targets[:, 0, None])
        aa = Pd.cuda()
        # aa = Pd

        new_targets = torch.cat((targets,aa),1)

        bb = torch.zeros(new_targets.size(0),1)
        for i in range(new_targets.size(0)):
            temp_num = new_targets[i, 1].item()
            temp_num1= int(temp_num/10)
            temp_num2 = temp_num%10
            new_targets[i, 1] = temp_num1
            bb[i,0]=temp_num2
        bb = bb.cuda()

        new_targets2 = torch.cat((new_targets, bb), 1)

        tcls1, tcls2, tbox, indices, anchors = self.build_targets(p, new_targets2)
        tcls2[:] = [x-3 for x in tcls2]

        # targets = targets.to(device='cpu', non_blocking=True).numpy()

        # computer Losses
        for i, pi in enumerate(p):  # layer index, layer predictions；
            b, a, gj, gi, pd = indices[i]  # image index, anchor index0\1\2, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj  ([8,3,80,80])
            # final_pro = np.load('dict/changed.npy')

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)  # sorce_iou: torch.size([135])
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
                pd = pd.to(torch.float16)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + pd * score_iou

                #  ----------------------------------------------------------------------
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:8], self.cn, device=device)  # targets
                    t2 = torch.full_like(ps[:, 8:], self.cn, device=device)  # targets
                    t[range(n), tcls1[i]] = self.cp
                    t2[range(n), tcls2[i]] = self.cp

                    lcls1 += self.BCEcls(ps[:, 5:8], t)  # BCE
                    lcls2 += self.BCEcls(ps[:, 8:], t2)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls1 *= self.hyp['cls']
        lcls2 *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls1 + lcls2) * bs, torch.cat((lbox, lobj, lcls1, lcls2)).detach()
        # ---------------------------------------------------------------------------------------------
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors 3, targets
        tcls1,tcls2, tbox, indices, anch = [], [], [], [],[]  # →list
        gain = torch.ones(9, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)  # ([[0,0...],[1,1...],[2,2...]]);shape [3,103]
        # print(ai[:, :, None].shape)
        # print(targets.repeat(na, 1, 1).shape)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        targets[:, :, [0,1,2,3,4,5,6,7,8]] = targets[:, :, [0,1,2,3,4,5,8,6,7]]  #　img，cls1，xy，wh，anchor，pd，cls2

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # wu
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:  # have targets，targets num not zeros
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio

                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = torch.max(r, 1. / r).max(2)[0] < 8.0
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy （[149,2]）
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # ([5,149])
                # print(t.repeat((5, 1, 1)))
                t = t.repeat((5, 1, 1))[j]  # ([445,7])
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # offsets = (torch.zeros_like(gxy)[None] + off[:, None]) # ([5,149,2])
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c= t[:, :2].long().T  # image index, class index
            d = t[:,8].long().T
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1),t[:,7]))  # image index, anchor index, grid indices？ y,x
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls1.append(c)  # class1
            tcls2.append(d)  # class1

        return tcls1,tcls2, tbox, indices, anch


    def ssim1(self, im1, im2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ux, uy = np.mean(im1), np.mean(im2)
        ox, oy = np.std(im1), np.std(im2)
        oxy = np.cov(im1, im2, ddof=0)[1][0]
        ssim_map = ((2 * ux * uy + C1) * (2 * oxy + C2)) / (
                    (np.square(ux) + np.square(uy) + C1) * (np.square(ox) + np.square(oy) + C2))
        return ssim_map

    def colorzabo(self,I,D):
        if np.shape(I)[0]==np.shape(D)[0] and np.shape(I)[1]==np.shape(D)[1]:
            c11 = self.ssim1(I[:,:,0],D[:,:,0])
            c22 = self.ssim1(I[:,:,1],D[:,:,1])
            c33 = self.ssim1(I[:, :, 2], D[:, :, 2])
            cc = 0.9*np.square(c11)+0.05*(np.square(c22)+np.square(c33))
        else:
            cc=0
        return cc

    def cal_index(self, img, tem):  # img:numpy;tem:list xywh
        [h, w, ch] = np.shape(img)
        x, y, w1, h1 = float(tem[0]) * w, float(tem[1]) * h, float(tem[2]) * w, float(tem[3]) * h

        x = x - w1 / 2
        y = y - h1 / 2
        rect = [math.ceil(x), math.ceil(y), math.ceil(w1), math.ceil(h1)]
        new_x, new_y, new_w, new_h = rect[0], rect[1], rect[2], rect[3]
        x_min = 0 if (new_x - new_w) < 0 else new_x - new_w
        y_min = 0 if (new_y - new_h) < 0 else new_y - new_h
        x_max = h if (new_x + 2 * new_w) > h else new_x + 2 * new_w
        y_max = h if (new_y + 2 * new_h) > h else new_y + 2 * new_h
        y_max2 = h-1 if (new_y + new_h) >= h else new_y + new_h
        x_max2 = h-1 if (new_x + new_w) >= h else new_x + new_w

        ares1 = img[y_min:new_y, x_min:new_x, 0:3]
        ares4 = img[new_y:y_max2, x_min:new_x, 0:3]
        ares7 = img[y_max2:y_max, x_min:new_x, 0:3]
        ares2 = img[y_min:new_y, new_x:x_max2, 0:3]
        ares3 = img[y_min:new_y, x_max2:x_max, 0:3]
        ares5 = img[new_y:y_max2, new_x:x_max2, 0:3]
        # cv.imshow('ares1', ares5)
        # cv.waitKey(0)
        ares6 = img[new_y:y_max2, x_max2:x_max, 0:3]
        ares8 = img[y_max2:y_max, new_x:x_max2, 0:3]
        ares9 = img[y_max2:y_max, x_max2:x_max, 0:3]

        temp_list = [ares1.astype(np.float32), ares2.astype(np.float32), ares3.astype(np.float32), ares4.astype(np.float32),
                     ares5.astype(np.float32), ares6.astype(np.float32), ares7.astype(np.float32), ares8.astype(np.float32), ares9.astype(np.float32)]
        over_list = []
        for temp_i in temp_list:
            if np.any(temp_i):
                over_list.append(color.rgb2lab(temp_i))
            else:
                over_list.append(temp_i)
        zone1, zone2, zone3, zone4 = over_list[0], over_list[1], over_list[2], over_list[3]
        zone5, zone6, zone7, zone8 = over_list[4], over_list[5], over_list[6], over_list[7]
        zone9 = over_list[8]

        L1, A1, B1 = zone1[:, :, 0], zone1[:, :, 1], zone1[:, :, 2]
        l1, a1, b1 = np.mean(L1), np.mean(A1), np.mean(B1)
        mo1 = np.sqrt(np.square(l1) + np.square(a1) + np.square(b1))

        L2, A2, B2 = zone2[:, :, 0], zone2[:, :, 1], zone2[:, :, 2]
        l2, a2, b2 = np.mean(L2), np.mean(A2), np.mean(B2)
        mo2 = np.sqrt(np.square(l2) + np.square(a2) + np.square(b2))

        L3, A3, B3 = zone3[:, :, 0], zone3[:, :, 1], zone3[:, :, 2]
        l3, a3, b3 = np.mean(L3), np.mean(A3), np.mean(B3)
        mo3 = np.sqrt(np.square(l3) + np.square(a3) + np.square(b3))

        L4, A4, B4 = zone4[:, :, 0], zone4[:, :, 1], zone4[:, :, 2]
        l4, a4, b4 = np.mean(L4), np.mean(A4), np.mean(B4)
        mo4 = np.sqrt(np.square(l4) + np.square(a4) + np.square(b4))

        L5, A5, B5 = zone5[:, :, 0], zone5[:, :, 1], zone5[:, :, 2]
        l5, a5, b5 = np.mean(L5), np.mean(A5), np.mean(B5)
        mo5 = np.sqrt(np.square(l5) + np.square(a5) + np.square(b5))

        L6, A6, B6 = zone6[:, :, 0], zone6[:, :, 1], zone6[:, :, 2]
        l6, a6, b6 = np.mean(L6), np.mean(A6), np.mean(B6)
        mo6 = np.sqrt(np.square(l6) + np.square(a6) + np.square(b6))

        L7, A7, B7 = zone7[:, :, 0], zone7[:, :, 1], zone7[:, :, 2]
        l7, a7, b7 = np.mean(L7), np.mean(A7), np.mean(B7)
        mo7 = np.sqrt(np.square(l7) + np.square(a7) + np.square(b7))

        L8, A8, B8 = zone8[:, :, 0], zone8[:, :, 1], zone8[:, :, 2]
        l8, a8, b8 = np.mean(L8), np.mean(A8), np.mean(B8)
        mo8 = np.sqrt(np.square(l8) + np.square(a8) + np.square(b8))

        L9, A9, B9 = zone9[:, :, 0], zone9[:, :, 1], zone9[:, :, 2]
        l9, a9, b9 = np.mean(L9), np.mean(A9), np.mean(B9)
        mo9 = np.sqrt(np.square(l9) + np.square(a9) + np.square(b9))
        [new_hh, new_ww] = np.shape(L5)
        c1, c2, c3, c4 = np.zeros([new_hh, new_ww]), np.zeros([new_hh, new_ww]), np.zeros([new_hh, new_ww]), np.zeros(
            [new_hh, new_ww])
        c5, c6, c7, c8, c9 = np.zeros([new_hh, new_ww]), np.zeros([new_hh, new_ww]), np.zeros([new_hh, new_ww]), np.zeros(
            [new_hh, new_ww]), np.zeros([new_hh, new_ww])
        for i in range(0, new_hh):
            for j in range(0, new_ww):
                c1[i, j] = (np.sqrt((L5[i, j] - l1) ** 2 + (A5[i, j] - a1) ** 2 + (B5[i, j] - b1) ** 2)) / (
                            np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo1+0.00001)
                c2[i, j] = (np.sqrt((L5[i, j] - l2) ** 2 + (A5[i, j] - a2) ** 2 + (B5[i, j] - b2) ** 2)) / (
                        np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo2+0.00001)
                c3[i, j] = (np.sqrt((L5[i, j] - l3) ** 2 + (A5[i, j] - a3) ** 2 + (B5[i, j] - b3) ** 2)) / (
                        np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo3+0.00001)
                c4[i, j] = (np.sqrt((L5[i, j] - l4) ** 2 + (A5[i, j] - a4) ** 2 + (B5[i, j] - b4) ** 2)) / (
                        np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo4+0.00001)
                c5[i, j] = (np.sqrt((L5[i, j] - l5) ** 2 + (A5[i, j] - a5) ** 2 + (B5[i, j] - b5) ** 2)) / (
                        np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo5+0.00001)
                c6[i, j] = (np.sqrt((L5[i, j] - l6) ** 2 + (A5[i, j] - a6) ** 2 + (B5[i, j] - b6) ** 2)) / (
                        np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo6+0.00001)
                c7[i, j] = (np.sqrt((L5[i, j] - l7) ** 2 + (A5[i, j] - a7) ** 2 + (B5[i, j] - b7) ** 2)) / (
                        np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo7+0.00001)
                c8[i, j] = (np.sqrt((L5[i, j] - l8) ** 2 + (A5[i, j] - a8) ** 2 + (B5[i, j] - b8) ** 2)) / (
                        np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo8+0.00001)
                c9[i, j] = (np.sqrt((L5[i, j] - l9) ** 2 + (A5[i, j] - a9) ** 2 + (B5[i, j] - b9) ** 2)) / (
                        np.sqrt(L5[i, j] ** 2 + A5[i, j] ** 2 + B5[i, j] ** 2) + mo9+0.00001)

        C1, C2, C3, C4 = np.sum(c1) / (new_hh * new_ww), np.sum(c2) / (new_hh * new_ww), np.sum(c3) / (
                    new_hh * new_ww), np.sum(c4) / (new_hh * new_ww)
        C5, C6, C7, C8, C9 = np.sum(c5) / (new_hh * new_ww), np.sum(c6) / (new_hh * new_ww), np.sum(c7) / (
                    new_hh * new_ww), np.sum(
            c8) / (new_hh * new_ww), np.sum(c9) / (new_hh * new_ww)
        D1, D2, D3, D4 = self.colorzabo(zone5, zone1), self.colorzabo(zone5, zone2), self.colorzabo(zone5,
                                                                                                    zone3), self.colorzabo(
            zone5,
            zone4)
        D9, D6, D7, D8 = self.colorzabo(zone5, zone9), self.colorzabo(zone5, zone6), self.colorzabo(zone5,
                                                                                                    zone7), self.colorzabo(
            zone5,
            zone8)

        if np.shape(ares9)[0] == 1 and np.shape(ares9)[1] == 1:
            color_dbd = np.min([C1, C2, C4])
            Zabo = np.max(np.abs([D1, D2, D4]))
        elif np.shape(ares9)[0] == 1 and np.shape(ares9)[1] != 1:
            color_dbd = np.min([C1, C2, C3, C4, C6])
            Zabo = np.max(np.abs([D1, D2, D3, D4, D6]))
        elif np.shape(ares9)[0] != 1 and np.shape(ares9)[1] == 1:
            color_dbd = np.min([C1, C2, C4, C7, C8])
            Zabo = np.max(np.abs([D1, D2, D4, D7, D8]))
        else:
            color_dbd = np.min([C1, C2, C3, C4, C6, C7, C8, C9])
            Zabo = np.max(np.abs([D1, D2, D3, D4, D6, D7, D8, D9]))
        target_area = (w1 * h1) / (w * h)

        sum_index = 0.2 * math.exp(-np.sqrt(Zabo)) + 0.2 * np.sqrt(color_dbd / 0.372) + 0.6 * np.sqrt(
            target_area / 0.22)
        return sum_index



