#coding:utf-8
'''
图像几何变换

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
import cv2
import random

from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter, ImageValidator

class Resize:
    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] = np.round(labels[:, [ymin+1, ymax+1]] * (img_height / self.out_height), decimals=0)
                labels[:, [xmin+1, xmax+1]] = np.round(labels[:, [xmin+1, xmax+1]] * (img_width / self.out_width), decimals=0)
                return labels

        if labels is None:
            if return_inverter:
                return image, inverter
            else:
                return image
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / img_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / img_width), decimals=0)

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.out_height,
                                         image_width=self.out_width)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

class ResizeRandomInterp:
    '''
    和上面Resize实现功能相同，只不过插值方式是随机的
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4],
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):


        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.resize = Resize(height=self.height,
                             width=self.width,
                             box_filter=self.box_filter,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.resize.interpolation_mode = np.random.choice(self.interpolation_modes)
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels, return_inverter)

class Flip:
    '''
    水平或者垂直翻转图像
    '''
    def __init__(self,
                 dim='horizontal',
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        if not (dim in {'horizontal', 'vertical'}): raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        if self.dim == 'horizontal':
            image = image[:,::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [xmin, xmax]] = img_width - labels[:, [xmax, xmin]]
                return image, labels
        else:
            image = image[::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [ymin, ymax]] = img_height - labels[:, [ymax, ymin]]
                return image, labels

class RandomFlip:
    '''
    随机翻转图像
    '''
    def __init__(self,
                 dim='horizontal',
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Translate:
    '''
    在水平或垂直方向移动图像
    '''

    def __init__(self,
                 dy,
                 dx,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dy (float): 垂直方向移动的距离百分比，正值向下移动，负值向上移动
            dx (float): 水平方向移动的距离百分比，正值向右移动，负值向左移动
        '''

        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.dy_rel = dy
        self.dx_rel = dx
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        #这个矩阵将图像在x，y方向进行平移
        dy_abs = int(round(img_height * self.dy_rel))
        dx_abs = int(round(img_width * self.dx_rel))
        M = np.float32([[1, 0, dx_abs],
                        [0, 1, dy_abs]])
        #平移之后剪裁为(img_width, img_height)大小
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']
            labels = np.copy(labels)

            # 对标签进行相应的变换
            labels[:,[xmin,xmax]] += dx_abs
            labels[:,[ymin,ymax]] += dy_abs

            # 计算此patch中有效的box.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=img_height,
                                         image_width=img_width)

            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=img_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=img_width-1)

            return image, labels

class RandomTranslate:
    '''
    Randomly translates images horizontally and/or vertically.
    '''

    def __init__(self,
                 dy_minmax=(0.03,0.3),
                 dx_minmax=(0.03,0.3),
                 prob=0.5,
                 clip_boxes=True,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            n_trials_max (int, optional): 仅在给出边界框标签时才生效。确定生成有效图像的最大试验次数。 
                如果在'n_trials_max`试验中无法生成有效图像，则返回未改变的输入图像。
        '''
        if dy_minmax[0] > dy_minmax[1]:
            raise ValueError("It must be `dy_minmax[0] <= dy_minmax[1]`.")
        if dx_minmax[0] > dx_minmax[1]:
            raise ValueError("It must be `dx_minmax[0] <= dx_minmax[1]`.")
        if dy_minmax[0] < 0 or dx_minmax[0] < 0:
            raise ValueError("It must be `dy_minmax[0] >= 0` and `dx_minmax[0] >= 0`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
            
        self.dy_minmax = dy_minmax
        self.dx_minmax = dx_minmax
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.translate = Translate(dy=0,
                                   dx=0,
                                   clip_boxes=self.clip_boxes,
                                   box_filter=self.box_filter,
                                   background=self.background,
                                   labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.translate.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # 数值
                dy_abs = np.random.uniform(self.dy_minmax[0], self.dy_minmax[1])
                dx_abs = np.random.uniform(self.dx_minmax[0], self.dx_minmax[1])
                # 方向
                dy = np.random.choice([-dy_abs, dy_abs])
                dx = np.random.choice([-dx_abs, dx_abs])
                self.translate.dy_rel = dy
                self.translate.dx_rel = dx

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.translate(image, labels)
                else:
                    # Translate the box coordinates to the translated image's coordinate system.
                    new_labels = np.copy(labels)
                    new_labels[:, [ymin, ymax]] += int(round(img_height * dy))
                    new_labels[:, [xmin, xmax]] += int(round(img_width * dx))

                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=img_height,
                                            image_width=img_width):
                        return self.translate(image, labels)

            # If all attempts failed, return the unaltered input image.
            if labels is None:
                return image

            else:
                return image, labels

        elif labels is None:
            return image

        else:
            return image, labels

class Scale:
    '''
    Scales images, i.e. zooms in or out.
    '''

    def __init__(self,
                 factor,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0)):

        if factor <= 0:
            raise ValueError("It must be `factor > 0`.")
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.factor = factor
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # 计算旋转矩阵（这里将角度设置为0），所以只缩放了
        # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                    angle=0,
                                    scale=self.factor)

        # 缩放图像
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            labels = np.copy(labels)
            # 使用M得到对应的bbox左上角以及右下角坐标

            toplefts = np.array([labels[:,xmin], labels[:,ymin], np.ones(labels.shape[0])])
            bottomrights = np.array([labels[:,xmax], labels[:,ymax], np.ones(labels.shape[0])])
            new_toplefts = (np.dot(M, toplefts)).T
            new_bottomrights = (np.dot(M, bottomrights)).T
            labels[:,[xmin,ymin]] = np.round(new_toplefts, decimals=0).astype(np.int)
            labels[:,[xmax,ymax]] = np.round(new_bottomrights, decimals=0).astype(np.int)

            # Compute all valid boxes for this patch.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=img_height,
                                         image_width=img_width)

            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=img_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=img_width-1)

            return image, labels

class RandomScale:
    '''
    Randomly scales images.
    '''

    def __init__(self,
                 min_factor=0.5,
                 max_factor=1.5,
                 prob=0.5,
                 clip_boxes=True,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):


        if not (0 < min_factor <= max_factor):
            raise ValueError("It must be `0 < min_factor <= max_factor`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.scale = Scale(factor=1.0,
                           clip_boxes=self.clip_boxes,
                           box_filter=self.box_filter,
                           background=self.background,
                           labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.scale.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Pick a scaling factor.
                factor = np.random.uniform(self.min_factor, self.max_factor)
                self.scale.factor = factor

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.scale(image, labels)
                else:
                    # Scale the bounding boxes accordingly.
                    # Transform two opposite corner points of the rectangular boxes using the rotation matrix `M`.
                    toplefts = np.array([labels[:,xmin], labels[:,ymin], np.ones(labels.shape[0])])
                    bottomrights = np.array([labels[:,xmax], labels[:,ymax], np.ones(labels.shape[0])])

                    # Compute the rotation matrix.
                    M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                                angle=0,
                                                scale=factor)

                    new_toplefts = (np.dot(M, toplefts)).T
                    new_bottomrights = (np.dot(M, bottomrights)).T

                    new_labels = np.copy(labels)
                    new_labels[:,[xmin,ymin]] = np.around(new_toplefts, decimals=0).astype(np.int)
                    new_labels[:,[xmax,ymax]] = np.around(new_bottomrights, decimals=0).astype(np.int)

                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=img_height,
                                            image_width=img_width):
                        return self.scale(image, labels)

            # 如果所有尝试都失败，选择未发生改变的原始图片
            if labels is None:
                return image

            else:
                return image, labels

        elif labels is None:
            return image

        else:
            return image, labels

class Rotate:
    '''
    逆时针旋转图像90, 180, 或270度.
    '''

    def __init__(self,
                 angle,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            angle (int): 只有90,180,和270是有效值.

        '''

        if not angle in {90, 180, 270}:
            raise ValueError("`angle` must be in the set {90, 180, 270}.")
        self.angle = angle
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # 生成旋转矩阵，以图像中心为原点旋转图像
        M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                    angle=self.angle,
                                    scale=1)

        # 从旋转矩阵得到sine和cosine
        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])

        # Compute the new bounding dimensions of the image.计算图像新维度
        img_width_new = int(img_height * sin_angle + img_width * cos_angle)
        img_height_new = int(img_height * cos_angle + img_width * sin_angle)

        # 根据变换调整旋转矩阵
        M[1, 2] += (img_height_new - img_height) / 2
        M[0, 2] += (img_width_new - img_width) / 2

        # 旋转图像
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width_new, img_height_new))

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            labels = np.copy(labels)
            toplefts = np.array([labels[:,xmin], labels[:,ymin], np.ones(labels.shape[0])])
            bottomrights = np.array([labels[:,xmax], labels[:,ymax], np.ones(labels.shape[0])])
            new_toplefts = (np.dot(M, toplefts)).T
            new_bottomrights = (np.dot(M, bottomrights)).T
            labels[:,[xmin,ymin]] = np.round(new_toplefts, decimals=0).astype(np.int)
            labels[:,[xmax,ymax]] = np.round(new_bottomrights, decimals=0).astype(np.int)

            if self.angle == 90:
                # ymin and ymax were switched by the rotation.
                labels[:,[ymax,ymin]] = labels[:,[ymin,ymax]]
            elif self.angle == 180:
                # ymin and ymax were switched by the rotation,
                # and also xmin and xmax were switched.
                labels[:,[ymax,ymin]] = labels[:,[ymin,ymax]]
                labels[:,[xmax,xmin]] = labels[:,[xmin,xmax]]
            elif self.angle == 270:
                # xmin and xmax were switched by the rotation.
                labels[:,[xmax,xmin]] = labels[:,[xmin,xmax]]

            return image, labels

class RandomRotate:
    def __init__(self,
                 angles=[90, 180, 270],
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        for angle in angles:
            if not angle in {90, 180, 270}:
                raise ValueError("`angles` can only contain the values 90, 180, and 270.")
        self.angles = angles
        self.prob = prob
        self.labels_format = labels_format
        self.rotate = Rotate(angle=90, labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            # Pick a rotation angle.
            self.rotate.angle = random.choice(self.angles)
            self.rotate.labels_format = self.labels_format
            return self.rotate(image, labels)

        elif labels is None:
            return image

        else:
            return image, labels
