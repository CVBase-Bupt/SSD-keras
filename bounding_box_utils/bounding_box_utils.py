#coding:utf-8
'''
Includes:
* Function to compute the IoU similarity for axis-aligned, rectangular, 2D bounding boxes
* Function for coordinate conversion for axis-aligned, rectangular, 2D bounding boxes

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

def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    表示box位置的三种形式如下：
        1) (xmin, xmax, ymin, ymax) - 'minmax' 
        2) (xmin, ymin, xmax, ymax) - 'corners' 
        2) (cx, cy, w, h) - 'centroids'

    不同形式之间的转换

    Arguments:
        tensor (array): 包含需要转换形式的四个坐标的数组
        start_index (int): 最后一个轴中第一个坐标的索引
        conversion (str, optional):转换方向，从哪种形式转换成哪种形式
        border_pixels (str, optional):如何处理box的边界像素 ,include(包含边界像素),
        exclude（不包含）,half（水平或垂直包含一个边）

    Returns:改变形式后的数组
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        #(xmin, xmax, ymin, ymax)->(cx, cy, w, h)
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 #  cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 #  cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d #  w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d #  h
    elif conversion == 'centroids2minmax':
        #(cx, cy, w, h)->(xmin, xmax, ymin, ymax)
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # ymax
    elif conversion == 'corners2centroids':
        #(xmin, ymin, xmax, ymax)->(cx, cy, w, h)
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 #  cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d #  w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d #  h
    elif conversion == 'centroids2corners':
        #(cx, cy, w, h)-> (xmin, ymin, xmax, ymax)
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 #  xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 #  ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 #  xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 #  ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1

def convert_coordinates2(tensor, start_index, conversion):
    '''

    convert_coordinates()的矩阵乘法实现.
    直至'centroids' 和 'minmax' 形式之间的转换.

    这个函数平均比`convert_coordinates()`慢一点，可能因为有更多的不必要的算法操作（因为两个矩阵是稀疏的）

    '''
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        M = np.array([[0.5, 0. , -1.,  0.],
                      [0.5, 0. ,  1.,  0.],
                      [0. , 0.5,  0., -1.],
                      [0. , 0.5,  0.,  1.]])
        tensor1[..., ind:ind+4] = np.dot(tensor1[..., ind:ind+4], M)
    elif conversion == 'centroids2minmax':
        M = np.array([[ 1. , 1. ,  0. , 0. ],
                      [ 0. , 0. ,  1. , 1. ],
                      [-0.5, 0.5,  0. , 0. ],
                      [ 0. , 0. , -0.5, 0.5]]) # The multiplicative inverse of the matrix above
        tensor1[..., ind:ind+4] = np.dot(tensor1[..., ind:ind+4], M)
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1

def intersection_area(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''

    计算交集,假设boxes1和boxes2分别包含m和n个boxes


    参数:
        boxes1 (array): 1维(4, )或2维(m, 4)
        boxes2 (array): 1维(4, )或2维(n, 4)
        coords (str, optional): 坐标形式
        mode (str, optional): outer_product或element-wise 
            outer_product：返回(m,n)矩阵，boxes1和boxes2所有框之间的交集
            element-wise： 返回1维矩阵. 如果boxes1和boxes2都是m个框，返回一个长度为m的数组，
                           位置i是boxes1[i]和boxes2[i]的交集
        border_pixels (str, optional): 如何处理box边界值

    返回:
        boxes1和boxes2之间的交集
    '''

    # 确保boxes有正确的shapes
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}: raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.",format(mode))

    # 有必要的话转换坐标的格式
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    m = boxes1.shape[0] # `boxes1`中boxes的数量
    n = boxes2.shape[0] # `boxes2`中boxes的数量

    # 对应不同的坐标形式建立不同的下标索引
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 
    elif border_pixels == 'exclude':
        d = -1 

    # 计算相交区域

    if mode == 'outer_product':
        #--------------
        #|            |
        #|      min_xy|
        #|         ---|--------
        #----------|---       |
        #          |  max_xy  |
        #          ------------
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmin,ymin]], axis=0), reps=(m, 1, 1)))

        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmax,ymax]], axis=0), reps=(m, 1, 1)))

        # 相交区域的边长
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,:,0] * side_lengths[:,:,1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
        max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,0] * side_lengths[:,1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''
    计算两组轴对齐的2维矩形框的交并比（intersection-over-union，也称为Jaccard相似度）
    boxes1和boxes2分别包含m和n个boxes

    Arguments:
        boxes1 (array): 1维(4, )或2维(m, 4)
        boxes2 (array): 1维(4, )或2维(n, 4)
        coords (str, optional): 坐标形式
        mode (str, optional): outer_product或element-wise 
            outer_product：返回(m,n)矩阵，boxes1和boxes2所有框之间的IOU 
            element-wise： 返回1维矩阵. 如果boxes1和boxes2都是m个框，返回一个长度为m的数组，
                       位置i是boxes1[i]和boxes2[i]的IOU
        border_pixels (str, optional): 如何处理box边界值

    Returns:
        boxes1和boxes2之间的IOU
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}: raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.".format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # 计算IoU.

    # 计算区域交集.
    intersection_areas = intersection_area(boxes1, boxes2, coords=coords, mode=mode)

    m = boxes1.shape[0] #  `boxes1`中box的数量
    n = boxes2.shape[0] #  `boxes2`中box的数量

    # 计算区域并集.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 
    elif border_pixels == 'exclude':
        d = -1
    if mode == 'outer_product':

        boxes1_areas = np.tile(np.expand_dims((boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d), axis=1), reps=(1,n))
        boxes2_areas = np.tile(np.expand_dims((boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d), axis=0), reps=(m,1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d)
        boxes2_areas = (boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas
