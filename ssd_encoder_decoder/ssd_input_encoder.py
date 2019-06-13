#coding:utf-8
'''
一个编码器:将ground truth注释转换成SSD兼容的训练目标

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np

from bounding_box_utils.bounding_box_utils import iou, convert_coordinates
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi

class SSDInputEncoder:
    
    # 将图像中目标检测的ground truth labels (2维 bounding box 坐标和类别标签) 转换成训练SSD模型需要的形式
    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 steps=None,
                 offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 border_pixels='half',
                 coords='centroids',
                 normalize_coords=True,
                 background_id=0):
        '''
        参数:
            img_height (int): 输入图像的高度.
            img_width (int):  输入图像的宽度.
            n_classes (int):  数据集类别数量（不包括背景类），比如 Pascal VOC为20, MS COCO为80.          
            predictor_sizes (list):包含卷积预测层输出的宽高 
            min_scale (float, optional): 必须 >0.          
            max_scale (float, optional): scaling factors可以通过最小与最大之间的线性插值得到，必须大于等于min_scale.            
            scales (list, optional): 每个预测层的scaling factors..           
            aspect_ratios_global (list, optional): 全局比例           
            aspect_ratios_per_layer (list, optional):每层比例          
            two_boxes_for_ar1 (bool, optional): box的比例是否包含1.           
            steps (list, optional): 输入到预测层缩放倍数            
            offsets (list, optional): 每个格子长宽的一半            
            clip_boxes (bool, optional): 如果 `True`,剪裁超过图像边界的框         
            variances (list, optional): 坐标偏移值会除以这个值

            matching_type (str, optional): 可以是`multi`或者`bipartite`.
                'bipartite', 每一个ground truth box会和与之IoU最高的一个default box匹配。 
                'multi',除了之前提到的bipartite匹配,所有IoU高于或者等于`pos_iou_threshold`的default box
                会和对应的ground truth box匹配.   
            pos_iou_threshold (float, optional): multi模式中判断是否匹配时的iou阈值，高于此值标为正样本。
            neg_iou_limit (float, optional): 低于这个值的被标记为负样本，
                如果default box不是正样本也不是负样本，不参与训练。            
            border_pixels (str, optional): 如何对待bounding boxes的边界像素。
                可以是 'include', 'exclude', 或者 'half'.
                如果是'include',边界像素属于boxes。如果是`exclude`，边界像素不属于boxes。
                如果是 'half',水平和垂直中的一个边界属于boxes,另一个不属于。
            coords (str, optional):模型中使用的坐标形式。（不是真实标签的输入形式） 
                'centroids':(cx, cy, w, h) (box center coordinates, width, height), 
                'minmax' :(xmin, xmax, ymin, ymax)
                'corners':(xmin, ymin, xmax, ymax)            
            normalize_coords (bool, optional): 如果为`True`,使用相对坐标而不是绝对坐标。
                 将坐标归一化到[0,1]之间。      
            background_id (int, optional): 确定背景类别的ID。
        '''
        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != predictor_sizes.shape[0] + 1): 
                raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}".format(len(scales), len(predictor_sizes)+1))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError("All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        else: 
            if not 0 < min_scale <= max_scale:
                raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(min_scale, max_scale))

        if not (aspect_ratios_per_layer is None):
            if (len(aspect_ratios_per_layer) != predictor_sizes.shape[0]):
                raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))
            for aspect_ratios in aspect_ratios_per_layer:
                if np.any(np.array(aspect_ratios) <= 0):
                    raise ValueError("All aspect ratios must be greater than zero.")
        else:
            if (aspect_ratios_global is None):
                raise ValueError("At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` must not be `None`.")
            if np.any(np.array(aspect_ratios_global) <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if not (coords == 'minmax' or coords == 'centroids' or coords == 'corners'):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

        if (not (steps is None)) and (len(steps) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one step value per predictor layer.")

        if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one offset value per predictor layer.")



        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1 # +1是背景类
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale

        # 如果`scales` 为空, 计算`min_scale` and `max_scale`之间的线性插值作为scaling factors
        if (scales is None):
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)# np.linspace创建等差数列         
        else:
            # 如果scales已经具体给出,直接使用就可以，不用从`min_scale`和`max_scale`计算得出.
            self.scales = scales

        
        # 如果 `aspect_ratios_per_layer`为空,每层使用相同的`aspect_ratios_global` 中的值
        if (aspect_ratios_per_layer is None):
            self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            # 如果每层的宽高比已给出，我们使用这些值就可以了。
            self.aspect_ratios = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1


        if not (steps is None):
            self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]
        if not (offsets is None):
            self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.background_id = background_id

        # 计算每个预测层的每个空间位置的boxes的数量
        # 例如，如果一个而预测层有三种比例[1.0, 0.5, 2.0],并且对于比例1.0，预测两个不同尺寸的boxes,
        # 预测层就会在特征图的每一个空间位置一共预测四个Boxs
        if not (aspect_ratios_per_layer is None):
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

        ##################################################################################
        # 计算每个预测层的defaults boxes list有n个预测层的数据，每个层为`(feature_map_height, feature_map_width, n_boxes, 4)`.
        ##################################################################################
        self.boxes_list = []
        for i in range(len(self.predictor_sizes)):
            boxes= self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                        this_scale=self.scales[i],
                                                        next_scale=self.scales[i+1],
                                                        this_steps=self.steps[i],
                                                        this_offsets=self.offsets[i])
            self.boxes_list.append(boxes)


    def __call__(self, ground_truth_labels):
        '''
        将真实数据转换成训练需要的格式
        参数:ground_truth_labels (list):(class_id, xmin, ymin, xmax, ymax)
        返回:y_encoded, (batch_size, #boxes, #classes + 4 + 4 + 4)
        '''

        #1. 真实标签顺序
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        batch_size = len(ground_truth_labels)

        # 整理anchor box的格式(batch_size, #boxes, #classes + 12)
        y_encoded = self.generate_encoding_template(batch_size=batch_size)

        # 匹配真实box和anchor box 
        y_encoded[:, :, self.background_id] = 1 # 所有boxes默认为背景.
        n_boxes = y_encoded.shape[1] 
        class_vectors = np.eye(self.n_classes) #  one-hot class vectors

        for i in range(batch_size): # For each batch item...
            if ground_truth_labels[i].size == 0: continue # If there is no ground truth for this batch item, there is nothing to match.
            labels = ground_truth_labels[i].astype(np.float) # The labels for this batch item

            # Check for degenerate ground truth bounding boxes before attempting any computations.
            if np.any(labels[:,[xmax]] - labels[:,[xmin]] <= 0) or np.any(labels[:,[ymax]] - labels[:,[ymin]] <= 0):
                raise DegenerateBoxError("SSDInputEncoder detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, labels) +
                                         "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. Degenerate ground truth " +
                                         "bounding boxes will lead to NaN errors during the training.")

            # normalize 
            if self.normalize_coords:
                labels[:,[ymin,ymax]] /= self.img_height 
                labels[:,[xmin,xmax]] /= self.img_width 

            # 可能需要转换坐标格式
            if self.coords == 'centroids':
                labels = convert_coordinates(labels, start_index=xmin, conversion='corners2centroids', border_pixels=self.border_pixels)
            elif self.coords == 'minmax':
                labels = convert_coordinates(labels, start_index=xmin, conversion='corners2minmax')

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)] # The one-hot class IDs for the ground truth boxes of this batch item
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin,ymin,xmax,ymax]]], axis=-1) # The one-hot version of the labels for this batch item

            #  计算IoU  `(num_ground_truth_boxes, num_anchor_boxes)`.
            similarities = iou(labels[:,[xmin,ymin,xmax,ymax]], y_encoded[i,:,-12:-8], coords=self.coords, mode='outer_product', border_pixels=self.border_pixels)

            # 1. 找到和每个真实框IOU最高的一个default box，这里保证了每一个真实框将至少匹配到一个default box.
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)
            # 将真实标签写入匹配到的default boxes中
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot
            # 将匹配到的default box设为0，表示已经匹配
            similarities[:, bipartite_matches] = 0

            #2. 剩余的default box会寻找与其IOU最大的真实框，如果IOU大于阈值pos_iou_threshold，匹配成功

            if self.matching_type == 'multi':
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)
                y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]
                similarities[:, matches[1]] = 0

            # 最后: 剩下的框中如果有IOU大于neg_iou_limit，设置为中立，因为和真实框比较接近，不适合作为背景类参与训练
            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        # 2.将坐标转换成偏移值
        if self.coords == 'centroids':
            y_encoded[:,:,[-12,-11]] -= y_encoded[:,:,[-8,-7]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:,:,[-12,-11]] /= y_encoded[:,:,[-6,-5]] * y_encoded[:,:,[-4,-3]] # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:,:,[-10,-9]] /= y_encoded[:,:,[-6,-5]] # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:,:,[-10,-9]] = np.log(y_encoded[:,:,[-10,-9]]) / y_encoded[:,:,[-2,-1]] # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        elif self.coords == 'corners':
            y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-10]] /= np.expand_dims(y_encoded[:,:,-6] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-11,-9]] /= np.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-7], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
        elif self.coords == 'minmax':
            y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-11]] /= np.expand_dims(y_encoded[:,:,-7] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-10,-9]] /= np.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-6], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
        return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps=None,
                                        this_offsets=None):
        '''
        Arguments:
            feature_map_size (tuple): [feature_map_height, feature_map_width]
            aspect_ratios (list): 生成的anchor boxes的比例
            this_scale (float)，next_scale (float): A float in [0, 1]
        Returns:
            (feature_map_height, feature_map_width, n_boxes_per_cell, 4)   4:坐标
        '''
        
        size = min(self.img_height, self.img_width)
        # 计算所有比例的box的宽和高
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list) #每个格子（cell)中有多少boxes

        # 计算box中心
        if (this_steps is None):
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        
        # this_offsets：anchor box 中心距左上角的像素值
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        # 计算default box中心坐标
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy) #生成网格
        cx_grid = np.expand_dims(cx_grid, -1) # np.tile()
        cy_grid = np.expand_dims(cy_grid, -1)

        # (feature_map_height, feature_map_width, n_boxes, 4） 最后一维4：(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # h

        # 将 (cx, cy, w, h) 转换成 (xmin, ymin, xmax, ymax)格式
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # 剪裁超出图像边界的boxes
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # 将坐标归一化到 [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        
        if self.coords == 'centroids':
            # (xmin, ymin, xmax, ymax)->(cx, cy, w, h)
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # (xmin, ymin, xmax, ymax)->(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')
        return boxes_tensor


    def generate_encoding_template(self, batch_size):
        '''
        这个函数中所有的tensor创建，reshape,concatenation操作以及调用的子函数都和SSD模型中的相同
        list 层 每层为`(feature_map_height, feature_map_width, n_boxes, 4)`.->`(batch_size, #boxes, #classes + 12)`
        Arguments:
            batch_size (int): The batch size.
        Returns:
            一个shape为 `(batch_size, #boxes, #classes + 12)`的数组，编码ground truth的标签的模板。最后一个轴的长度是
            `#classes + 12`，因为模型的输出不止是4个预测的坐标偏移值，还有4个default boxes的坐标，以及4个variance values.        
        '''  
        #1. anchor boxes
        boxes_batch = []
        for boxes in self.boxes_list:
            # self.boxes_list list (feature_map_height, feature_map_width, n_boxes, 4)
            # 5D tensor `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # 5D tensor -> 3D tensor `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. 
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # (batch, sum_per_predict_layer(feature_map_height * feature_map_width * n_boxes), 4)
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        # 2: one-hot class encodings `(batch, #boxes, #classes)`
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 3: variances. 和 `boxes_tensor` shape相同，只是简单的在最后一维的每个位置包含了4 variance值.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances # Long live broadcasting

        # 4：concat classes, boxes, variances 
        #  y_encoding_template` 理应和 SSD模型输出的tensor shape 相同，内容没用关系，所以又用`boxes_tensor`填充了一次
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)
        
        return y_encoding_template

class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass
