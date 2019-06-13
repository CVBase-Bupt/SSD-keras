#encoding:utf-8
'''
数据增强中多种patch采样操作

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

from data_generator.object_detection_2d_image_boxes_validation_utils import BoundGenerator, BoxFilter, ImageValidator

class PatchCoordinateGenerator:
    '''
    生成满足要求的随机patch坐标
    '''

    def __init__(self,
                 img_height=None,
                 img_width=None,
                 must_match='h_w',
                 min_scale=0.3,
                 max_scale=1.0,
                 scale_uniformly=False,
                 min_aspect_ratio = 0.5,
                 max_aspect_ratio = 2.0,
                 patch_ymin=None,
                 patch_xmin=None,
                 patch_height=None,
                 patch_width=None,
                 patch_aspect_ratio=None):
        '''
        Arguments:
            img_height (int)，img_width (int): 图像的高和宽
            must_match (str, optional): 'h_w'/'h_ar'/ 'w_ar'，给定两个，另一个计算得出
            min_scale (float, optional)，max_scale (float, optional): patch高/宽 = 图像高/宽*scale 
            scale_uniformly (bool, optional): 值为`True`且`must_match == 'h_w'时，patch的高和宽会被一致缩放
            min_aspect_ratio (float, optional),max_aspect_ratio (float, optional):生成patch的最小/最大比例
            patch_ymin (int, optional),patch_xmin (int, optional): `None`或者左上角垂直/水平坐标，如果是'None'，随机生成
            patch_height (int, optional), patch_width (int, optional),patch_aspect_ratio (float, optional):
                'None'或者生成patch的高，宽，比例
            
        '''

        if not (must_match in {'h_w', 'h_ar', 'w_ar'}):
            raise ValueError("`must_match` must be either of 'h_w', 'h_ar' and 'w_ar'.")
        if min_scale >= max_scale:
            raise ValueError("It must be `min_scale < max_scale`.")
        if min_aspect_ratio >= max_aspect_ratio:
            raise ValueError("It must be `min_aspect_ratio < max_aspect_ratio`.")
        if scale_uniformly and not ((patch_height is None) and (patch_width is None)):
            raise ValueError("If `scale_uniformly == True`, `patch_height` and `patch_width` must both be `None`.")
        self.img_height = img_height
        self.img_width = img_width
        self.must_match = must_match
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_uniformly = scale_uniformly
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_aspect_ratio = patch_aspect_ratio

    def __call__(self):
        '''
        Returns:
            返回生成patch的坐标：4-tuple (ymin, xmin, height, width)`
        '''

        # 1. 得到patch的高和宽
        if self.must_match == 'h_w': 
            if not self.scale_uniformly:
                # height.
                if self.patch_height is None:
                    patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_height)
                else:
                    patch_height = self.patch_height
                # width.
                if self.patch_width is None:
                    patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_width)
                else:
                    patch_width = self.patch_width
            else:
                scaling_factor = np.random.uniform(self.min_scale, self.max_scale)
                patch_height = int(scaling_factor * self.img_height)
                patch_width = int(scaling_factor * self.img_width)

        elif self.must_match == 'h_ar': # Width是独立变量
            # height.
            if self.patch_height is None:
                patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_height)
            else:
                patch_height = self.patch_height
            # aspect ratio.
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # width
            patch_width = int(patch_height * patch_aspect_ratio)

        elif self.must_match == 'w_ar': 
            # width.
            if self.patch_width is None:
                patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_width)
            else:
                patch_width = self.patch_width
            # aspect ratio.
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # height
            patch_height = int(patch_width / patch_aspect_ratio)

        # 2. 得到patch左上角坐标.
        if self.patch_ymin is None:
            y_range = self.img_height - patch_height #可以为负值
            if y_range >= 0: patch_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension.
            else: patch_ymin = np.random.randint(y_range, 1) # The possible positions for the image on the background canvas in the vertical dimension.
        else:
            patch_ymin = self.patch_ymin

        if self.patch_xmin is None:
            x_range = self.img_width - patch_width
            if x_range >= 0: patch_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension.
            else: patch_xmin = np.random.randint(x_range, 1) # The possible positions for the image on the background canvas in the horizontal dimension.
        else:
            patch_xmin = self.patch_xmin

        return (patch_ymin, patch_xmin, patch_height, patch_width)

class CropPad:
    def __init__(self,
                 patch_ymin,
                 patch_xmin,
                 patch_height,
                 patch_width,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        #if (patch_height <= 0) or (patch_width <= 0):
        #    raise ValueError("Patch height and width must both be positive.")
        #if (patch_ymin + patch_height < 0) or (patch_xmin + patch_width < 0):
        #    raise ValueError("A patch with the given coordinates cannot overlap with an input image.")
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        if (self.patch_ymin > img_height) or (self.patch_xmin > img_width):
            raise ValueError("The given patch doesn't overlap with the input image.")

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # patch左上角
        patch_ymin = self.patch_ymin
        patch_xmin = self.patch_xmin
        if image.ndim == 3:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width, 3), dtype=np.uint8)
            canvas[:, :] = self.background
        elif image.ndim == 2:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width), dtype=np.uint8)
            canvas[:, :] = self.background[0]

        # crop.
        if patch_ymin < 0 and patch_xmin < 0: # Pad the image at the top and on the left.
            image_crop_height = min(img_height, self.patch_height + patch_ymin)  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(img_width, self.patch_width + patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[-patch_ymin:-patch_ymin + image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[:image_crop_height, :image_crop_width]

        elif patch_ymin < 0 and patch_xmin >= 0: # Pad the image at the top and crop it on the left.
            image_crop_height = min(img_height, self.patch_height + patch_ymin)  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(self.patch_width, img_width - patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[-patch_ymin:-patch_ymin + image_crop_height, :image_crop_width] = image[:image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        elif patch_ymin >= 0 and patch_xmin < 0: # Crop the image at the top and pad it on the left.
            image_crop_height = min(self.patch_height, img_height - patch_ymin) # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(img_width, self.patch_width + patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[:image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height, :image_crop_width]

        elif patch_ymin >= 0 and patch_xmin >= 0: # Crop the image at the top and on the left.
            image_crop_height = min(self.patch_height, img_height - patch_ymin) # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(self.patch_width, img_width - patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[:image_crop_height, :image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        image = canvas

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] += patch_ymin
                labels[:, [xmin+1, xmax+1]] += patch_xmin
                return labels

        if not (labels is None):
            labels[:, [ymin, ymax]] -= patch_ymin
            labels[:, [xmin, xmax]] -= patch_xmin
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.patch_height,
                                         image_width=self.patch_width)
            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=self.patch_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=self.patch_width-1)
            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

        else:
            if return_inverter:
                return image, inverter
            else:
                return image

class Crop:
    '''
    剪裁掉图像边缘的一些像素，CropPad的接口
    '''

    def __init__(self,
                 crop_top,
                 crop_bottom,
                 crop_left,
                 crop_right,
                 clip_boxes=True,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.crop = CropPad(patch_ymin=self.crop_top,
                            patch_xmin=self.crop_left,
                            patch_height=None,
                            patch_width=None,
                            clip_boxes=self.clip_boxes,
                            box_filter=self.box_filter,
                            labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        self.crop.patch_height = img_height - self.crop_top - self.crop_bottom
        self.crop.patch_width = img_width - self.crop_left - self.crop_right
        self.crop.labels_format = self.labels_format

        return self.crop(image, labels, return_inverter)

class Pad:
    '''
    使用指定值pad图像的每个边，CropPad的接口
    '''

    def __init__(self,
                 pad_top,
                 pad_bottom,
                 pad_left,
                 pad_right,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.background = background
        self.labels_format = labels_format
        self.pad = CropPad(patch_ymin=-self.pad_top,
                           patch_xmin=-self.pad_left,
                           patch_height=None,
                           patch_width=None,
                           clip_boxes=False,
                           box_filter=None,
                           background=self.background,
                           labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        self.pad.patch_height = img_height + self.pad_top + self.pad_bottom
        self.pad.patch_width = img_width + self.pad_left + self.pad_right
        self.pad.labels_format = self.labels_format

        return self.pad(image, labels, return_inverter)

class RandomPatch:
    '''
    从图像中随机采样patch，这里的随机指的是patch coordinate generator, the box filter, 和patch validator带来的随机性
    如果此变换用于生成固定大小或宽高比的patch，但是没有满足设定大小和比例的输出图像，它将返回“None”。
    '''

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 clip_boxes=True,
                 prob=1.0,
                 background=(0,0,0),
                 can_fail=False,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            can_fail (bool, optional): 如果是'True',如果没有找到有效的patch则返回 `None`. 如果是 `False`,这种情况
                下会返回未经修改的输入图像。 
        '''
        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError("`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.can_fail = can_fail
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]
            self.patch_coord_generator.img_height = img_height
            self.patch_coord_generator.img_width = img_width

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.sample_patch.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Generate patch coordinates.
                patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                self.sample_patch.patch_ymin = patch_ymin
                self.sample_patch.patch_xmin = patch_xmin
                self.sample_patch.patch_height = patch_height
                self.sample_patch.patch_width = patch_width

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.sample_patch(image, labels, return_inverter)
                else:
                    # Translate the box coordinates to the patch's coordinate system.
                    new_labels = np.copy(labels)
                    new_labels[:, [ymin, ymax]] -= patch_ymin
                    new_labels[:, [xmin, xmax]] -= patch_xmin
                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=patch_height,
                                            image_width=patch_width):
                        return self.sample_patch(image, labels, return_inverter)

            # If we weren't able to sample a valid patch...
            if self.can_fail:
                # ...return `None`.
                if labels is None:
                    if return_inverter:
                        return None, None
                    else:
                        return None
                else:
                    if return_inverter:
                        return None, None, None
                    else:
                        return None, None
            else:
                # ...return the unaltered input image.
                if labels is None:
                    if return_inverter:
                        return image, None
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, None
                    else:
                        return image, labels

        else:
            if return_inverter:
                def inverter(labels):
                    return labels

            if labels is None:
                if return_inverter:
                    return image, inverter
                else:
                    return image
            else:
                if return_inverter:
                    return image, labels, inverter
                else:
                    return image, labels

class RandomPatchInf:
    '''
    这个操作和`RandomPatch`相似, 除了:
    1. 这个操作会无限运行直到找到有效的patch，或者返回未经修改的输入图像
    2. 如果给定边界生成器，每个`n_trials_max`迭代都会生成新的边界
    '''

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 bound_generator=None,
                 n_trials_max=50,
                 clip_boxes=True,
                 prob=0.857,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):


        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError("`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        if not (isinstance(bound_generator, BoundGenerator) or bound_generator is None):
            raise ValueError("`bound_generator` must be either `None` or a `BoundGenerator` object.")
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.bound_generator = bound_generator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]
        self.patch_coord_generator.img_height = img_height
        self.patch_coord_generator.img_width = img_width

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Override the preset labels format.
        if not self.image_validator is None:
            self.image_validator.labels_format = self.labels_format
        self.sample_patch.labels_format = self.labels_format

        while True: # Keep going until we either find a valid patch or return the original image.

            p = np.random.uniform(0,1)
            if p >= (1.0-self.prob):

                # In case we have a bound generator, pick a lower and upper bound for the patch validator.
                if not ((self.image_validator is None) or (self.bound_generator is None)):
                    self.image_validator.bounds = self.bound_generator()

                # Use at most `self.n_trials_max` attempts to find a crop
                # that meets our requirements.
                for _ in range(max(1, self.n_trials_max)):

                    # Generate patch coordinates.
                    patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                    self.sample_patch.patch_ymin = patch_ymin
                    self.sample_patch.patch_xmin = patch_xmin
                    self.sample_patch.patch_height = patch_height
                    self.sample_patch.patch_width = patch_width

                    # Check if the resulting patch meets the aspect ratio requirements.
                    aspect_ratio = patch_width / patch_height
                    if not (self.patch_coord_generator.min_aspect_ratio <= aspect_ratio <= self.patch_coord_generator.max_aspect_ratio):
                        continue

                    if (labels is None) or (self.image_validator is None):
                        # We either don't have any boxes or if we do, we will accept any outcome as valid.
                        return self.sample_patch(image, labels, return_inverter)
                    else:
                        # Translate the box coordinates to the patch's coordinate system.
                        new_labels = np.copy(labels)
                        new_labels[:, [ymin, ymax]] -= patch_ymin
                        new_labels[:, [xmin, xmax]] -= patch_xmin
                        # Check if the patch contains the minimum number of boxes we require.
                        if self.image_validator(labels=new_labels,
                                                image_height=patch_height,
                                                image_width=patch_width):
                            return self.sample_patch(image, labels, return_inverter)
            else:
                if return_inverter:
                    def inverter(labels):
                        return labels

                if labels is None:
                    if return_inverter:
                        return image, inverter
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, inverter
                    else:
                        return image, labels

class RandomMaxCropFixedAR:
    '''
    从图像中裁剪给定固定宽高比的情况下最大的补丁。
    '''

    def __init__(self,
                 patch_aspect_ratio,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 clip_boxes=True,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_aspect_ratio (float): 所有采样的patch都有固定的比例

        '''

        self.patch_aspect_ratio = patch_aspect_ratio
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.labels_format = labels_format
        self.random_patch = RandomPatch(patch_coord_generator=PatchCoordinateGenerator(), # Just a dummy object
                                        box_filter=self.box_filter,
                                        image_validator=self.image_validator,
                                        n_trials_max=self.n_trials_max,
                                        clip_boxes=self.clip_boxes,
                                        prob=1.0,
                                        can_fail=False,
                                        labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        # The ratio of the input image aspect ratio and patch aspect ratio determines the maximal possible crop.
        image_aspect_ratio = img_width / img_height

        if image_aspect_ratio < self.patch_aspect_ratio:
            patch_width = img_width
            patch_height = int(round(patch_width / self.patch_aspect_ratio))
        else:
            patch_height = img_height
            patch_width = int(round(patch_height * self.patch_aspect_ratio))

        # Now that we know the desired height and width for the patch,
        # instantiate an appropriate patch coordinate generator.
        patch_coord_generator = PatchCoordinateGenerator(img_height=img_height,
                                                         img_width=img_width,
                                                         must_match='h_w',
                                                         patch_height=patch_height,
                                                         patch_width=patch_width)

        # The rest of the work is done by `RandomPatch`.
        self.random_patch.patch_coord_generator = patch_coord_generator
        self.random_patch.labels_format = self.labels_format
        return self.random_patch(image, labels, return_inverter)

class RandomPadFixedAR:
    '''
    向图像添加最小可能的填充，从而生成包含整个图像的给定固定宽高比的patch。
    由于所得图像的纵横比是恒定的，因此可以随后将它们调整为相同尺寸而不会失真。
    '''

    def __init__(self,
                 patch_aspect_ratio,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):


        self.patch_aspect_ratio = patch_aspect_ratio
        self.background = background
        self.labels_format = labels_format
        self.random_patch = RandomPatch(patch_coord_generator=PatchCoordinateGenerator(), # Just a dummy object
                                        box_filter=None,
                                        image_validator=None,
                                        n_trials_max=1,
                                        clip_boxes=False,
                                        background=self.background,
                                        prob=1.0,
                                        labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        if img_width < img_height:
            patch_height = img_height
            patch_width = int(round(patch_height * self.patch_aspect_ratio))
        else:
            patch_width = img_width
            patch_height = int(round(patch_width / self.patch_aspect_ratio))

        # Now that we know the desired height and width for the patch,
        # instantiate an appropriate patch coordinate generator.
        patch_coord_generator = PatchCoordinateGenerator(img_height=img_height,
                                                         img_width=img_width,
                                                         must_match='h_w',
                                                         patch_height=patch_height,
                                                         patch_width=patch_width)

        # The rest of the work is done by `RandomPatch`.
        self.random_patch.patch_coord_generator = patch_coord_generator
        self.random_patch.labels_format = self.labels_format
        return self.random_patch(image, labels, return_inverter)
