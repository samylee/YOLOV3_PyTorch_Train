import torch
import torch.nn as nn
import numpy as np
from utils.utils import bbox_iou


class YOLOV3Loss(nn.Module):
    def __init__(self, B=3, C=20, anchors=None, device=None):
        super(YOLOV3Loss, self).__init__()
        self.B, self.C = B, C
        self.device = device
        self.anchors = torch.from_numpy(np.asarray(anchors, dtype=np.float32)).view(-1, self.B, 2).to(self.device)
        self.anchors_total = self.anchors.view(-1, 2)
        self.ignore_thresh = 0.7
        self.masks = torch.tensor(range(self.anchors_total.size(0)), dtype=torch.int32).view(-1, self.B).to(self.device)

        self.class_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.noobj_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.obj_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.coords_criterion = nn.MSELoss(reduction='sum').to(self.device)

    def target_box_encode(self, box, grid_size, img_size, i, j, index):
        box_out = box.clone()
        box_out[0] = box[0] * grid_size - i
        box_out[1] = box[1] * grid_size - j
        box_out[2] = torch.log(box[2] * img_size / self.anchors_total[index, 0])
        box_out[3] = torch.log(box[3] * img_size / self.anchors_total[index, 1])
        return box_out

    def make_grid(self, nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, preds, targets, img_size):
        loss = 0
        for idx, pred in enumerate(preds):
            batch_size, _, grid_size, _ = pred.shape

            # num_samples, 3(anchors), 13(grid), 13(grid), 25 (tx, ty, tw, th, conf, classes)
            pred_permute = (
                pred.view(batch_size, self.B, self.C+5, grid_size, grid_size)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
            )

            # tx, ty
            pred_permute[..., 0:2] = torch.sigmoid(pred_permute[..., 0:2])
            # conf, class
            pred_permute[..., 4:] = torch.sigmoid(pred_permute[..., 4:])

            # decode boxes for forward
            preds_coords = torch.empty((batch_size, self.B, grid_size, grid_size, 4), dtype=torch.float32).to(self.device)
            grid = self.make_grid(grid_size, grid_size).to(self.device)
            anchor_grid = self.anchors[idx].view(1, -1, 1, 1, 2)
            preds_coords[..., 0:2] = (pred_permute[..., 0:2] + grid) / grid_size
            preds_coords[..., 2:4] = torch.exp(pred_permute[..., 2:4]) * anchor_grid / img_size

            # find noobj
            noobj_mask_list = []
            for b in range(batch_size):
                preds_coords_batch = preds_coords[b]
                targets_coords_batch = targets[b, :, 1:]
                preds_ious_list = []
                for target_coords_batch in targets_coords_batch:
                    if target_coords_batch[0] == 0:
                        break
                    preds_ious_noobj = bbox_iou(preds_coords_batch.view(-1, 4), target_coords_batch.unsqueeze(0)).view(self.B, grid_size, grid_size)
                    preds_ious_list.append(preds_ious_noobj)
                preds_ious_tensor = torch.stack(preds_ious_list, dim=0)
                preds_ious_max = torch.max(preds_ious_tensor, dim=0)[0]

                noobj_mask = preds_ious_max <= self.ignore_thresh
                noobj_mask_list.append(noobj_mask)
            noobj_mask_tensor = torch.stack(noobj_mask_list)

            # generate obj mask and iou
            obj_mask_tensor = torch.empty_like(pred_permute[..., 4], dtype=torch.bool, requires_grad=False).fill_(False).to(self.device)
            obj_class_tensor = torch.zeros_like(pred_permute[..., 5:], requires_grad=False)
            targets_coords_encode = torch.zeros_like(preds_coords, requires_grad=False)
            loss_coords_wh_scale = torch.zeros_like(preds_coords, requires_grad=False)
            for b in range(batch_size):
                targets_batch = targets[b]
                for target_batch in targets_batch:
                    target_class_batch = int(target_batch[0])
                    assert target_class_batch < self.C, 'oh shit'
                    target_coords_batch = target_batch[1:]
                    if target_coords_batch[0] == 0:
                        break
                    i = int(target_coords_batch[0] * grid_size)
                    j = int(target_coords_batch[1] * grid_size)

                    target_coords_batch_shift = torch.zeros((1, 4), dtype=torch.float32).to(self.device)
                    target_coords_batch_shift[0, 2:] = target_coords_batch[2:]

                    anchors_match_batch_shift = torch.zeros((self.anchors_total.size(0), 4), dtype=torch.float32).to(self.device)
                    anchors_match_batch_shift[:, 2] = self.anchors_total[:, 0] / img_size
                    anchors_match_batch_shift[:, 3] = self.anchors_total[:, 1] / img_size

                    anchors_ious = bbox_iou(anchors_match_batch_shift, target_coords_batch_shift)

                    # get obj index
                    anchors_ious_index = torch.max(anchors_ious, dim=0)[1].item()
                    if anchors_ious_index in self.masks[idx]:
                        # target box encode
                        target_coords_batch_encode = self.target_box_encode(target_coords_batch, grid_size, img_size, i, j, anchors_ious_index)

                        # do not ignore second label in the same grid and same anchor_index
                        # if obj_mask_tensor[b, j, i, anchors_ious_index]:
                        #     continue

                        anchors_ious_index = anchors_ious_index if anchors_ious_index in [0,1,2] else anchors_ious_index - 3

                        obj_mask_tensor[b, anchors_ious_index, j, i] = True
                        noobj_mask_tensor[b, anchors_ious_index, j, i] = False
                        obj_class_tensor[b, anchors_ious_index, j, i, target_class_batch] = 1.0
                        targets_coords_encode[b, anchors_ious_index, j, i] = target_coords_batch_encode

                        current_scale_wh = 2 - target_coords_batch[2]*target_coords_batch[3]
                        loss_coords_wh_scale[b, anchors_ious_index, j, i] = current_scale_wh.repeat(4)

            # 1. noobj loss
            preds_conf_noobj_mask = pred_permute[..., 4][noobj_mask_tensor]
            loss_noobj = self.noobj_criterion(preds_conf_noobj_mask, torch.zeros_like(preds_conf_noobj_mask))

            # 2. obj loss
            preds_conf_obj_mask = pred_permute[..., 4][obj_mask_tensor]
            loss_obj = self.obj_criterion(preds_conf_obj_mask, torch.ones_like(preds_conf_obj_mask))

            # 3. class loss
            class_mask = obj_mask_tensor.unsqueeze(-1).expand_as(pred_permute[..., 5:])
            preds_class_mask = pred_permute[..., 5:][class_mask]
            obj_class_tensor_mask = obj_class_tensor[class_mask]
            loss_class = self.class_criterion(preds_class_mask, obj_class_tensor_mask)

            # 4. coords loss
            coords_mask = obj_mask_tensor.unsqueeze(-1).expand_as(preds_coords)
            preds_coords_obj_mask = pred_permute[..., 0:4][coords_mask]
            targets_coords_obj_encode_mask = targets_coords_encode[coords_mask]
            loss_coords_wh_scale_mask = loss_coords_wh_scale[coords_mask]
            loss_coords = self.coords_criterion(preds_coords_obj_mask * loss_coords_wh_scale_mask, targets_coords_obj_encode_mask * loss_coords_wh_scale_mask)

            loss += (loss_class + loss_obj + loss_noobj + loss_coords) / batch_size

        return loss