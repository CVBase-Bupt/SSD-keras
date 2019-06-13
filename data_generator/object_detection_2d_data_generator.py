# encoding:utf-8
'''
解析数据，生成训练数据

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
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys
import h5py
from bs4 import BeautifulSoup
import pickle

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter

class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass

class DatasetError(Exception):
    '''
    An exception class to be raised if a anything is wrong with the dataset,
    in particular if you try to generate batches when no dataset was loaded.
    '''
    pass

class DataGenerator:
    '''
    生成样本和对应标签

    每次传递数据之后可以打乱数据的顺序
    Can shuffle the dataset consistently after each complete pass.
    
    Can perform image transformations for data conversion and data augmentation,
    '''

    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        '''
        初始化数据生成器，可以直接从这里的构造器中直接读取数据，比如一个HDF5数据集,或者使用这里的parser methods
        读取数据集

        参数:
            load_images_into_memory (bool, optional): 如果是True,所有的数据集都会加载到内存中（请确保你有足够的内存）
                这样比一批一批加载数据到内存中要快
                
            hdf5_dataset_path (str, optional): 包含数据集的HDF5文件的路径 ，create_hdf5_dataset()模型制作。
            如果你加载了一个这样的HDF5数据集, 不需要再使用任何的parser methods，HDF5数据集已经包含了所有相关数据。

            filenames (string or list, optional): 图像路径
            filenames_type (string, optional): pickled文件或text文件.
            images_dir (string, optional): 
                如果为`filenames`传递文本文件，则图像的完整路径将由`images_dir`和文本文件中的名称组成，
                即这应该是包含文本文件所引用的图像的目录。 
                如果`filenames_type`不是'text'，那么这个参数是无关紧要的。
            labels (string or list, optional): 数据集标签
            image_ids (string or list, optional): 数据集中图像的ID
            eval_neutral (string or list, optional): 是否应该在评估期间将该对象视为中立。
            labels_output_format (list, optional): 下面几项数据的顺序，期望的数据顺序是 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
            verbose (bool, optional): 如果True,打印出耗时较长的进程
        '''
        self.labels_output_format = labels_output_format
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')} 

        self.dataset_size = 0 #现在还没有加载数据
        self.load_images_into_memory = load_images_into_memory
        self.images = None # 除非load_images_into_memory == True，这个值才不会是None

        # `self.filenames`是一个包含图像样本的所有文件名的列表（完整路径）。
        # 请注意，它不包含实际的图像文件本身。
        # 此列表是解析器方法的输出之一
        # 如果要加载HDF5数据集，此列表将为“None”。
        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            if load_images_into_memory:
                self.images = []
                it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        # 如果ground truth是可用的, 
        #`self.labels` 是一个包括每张图像ground truth bounding boxes的列表
        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

        if not eval_neutral is None:
            if isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as f:
                    self.eval_neutral = pickle.load(f)
            elif isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.eval_neutral = None

        if not hdf5_dataset_path is None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset()
        else:
            self.hdf5_dataset = None

    def load_hdf5_dataset(self, ):
        '''
        加载create_hdf5_dataset()制作的HDF5数据集

        Returns:
            None.
        '''
        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        self.dataset_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32) # Instead of shuffling the HDF5 dataset or images in memory, we will shuffle this index list.

        if self.load_images_into_memory:
            self.images = []
            tr = range(self.dataset_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])


    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=[],
                  classes=['background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  include_classes = 'all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False,
                  verbose=True):
        '''
        这是Pascal VOC数据集的XML解析器， 对代码进行微小更改后可能适用于其它数据集。
        但在当前代码，适用于Pascal VOC数据集的数据格式和XML标记。

        参数:
            images_dirs (list):Pascal VOC 2007和2012的图像路径
            image_set_filenames (list):文本路径，文本包含训练集与测试集图像ID
            annotations_dirs (list, optional):xml文件名称为对应的图像ID，文件包含每张图像的标签，包括目标类别以及坐标
            classes (list, optional): 目标类别，列表第一项为背景类，类别的顺序确定了类别的ID
            include_classes (list, optional):  'all' 或者是训练集中包括的类别ID的列表.
             If 'all', all ground truth boxes will be included in the dataset.
            exclude_truncated (bool, optional): 如果是 `True`, 不包括标记为只包含物体一部分的框.
            exclude_difficult (bool, optional): 如果是 `True`, 不包括标记为难以判断的框.
            ret (bool, optional): 是否返回parser的输出。
            verbose (bool, optional): 如果是`True`, 打印出可能需要较长时间操作的进度.

        Returns:
            默认没有返回值，可选images, image filenames, labels, image IDs,以及标记为"difficult"的boxes的列表
        '''
        # Set class members.
        self.images_dirs = images_dirs
        self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        self.classes = classes
        self.include_classes = include_classes

        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.eval_neutral = []
        if not annotations_dirs:
            self.labels = None
            self.eval_neutral = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            
            # 遍历文件得到训练集或测试集所有图像ID
            with open(image_set_filename) as f:
                image_ids = [line.strip() for line in f] # Note: 这些是字符串，不是整数
                self.image_ids += image_ids

            # 遍历数据集中所有图像
            for image_id in image_ids:
                filename = '{}'.format(image_id) + '.jpg' #图像名称
                self.filenames.append(os.path.join(images_dir, filename)) #图像路径

                # 解析标签文件
                if not annotations_dir is None:
                    #打开当前图像的标签文件并解析
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')
                    folder = soup.folder.text 

                    boxes = [] 
                    eval_neutr = [] 
                    objects = soup.find_all('object') # 得到图像包含的目标

                    # 解析每一个目标的数据
                    for obj in objects:
                        #类别
                        class_name = obj.find('name', recursive=False).text
                        #类别ID
                        class_id = self.classes.index(class_name)
                        # 检查我们是否计划训练此类别，若不训练，解析下一个目标
                        if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                        
                        #pose = obj.find('pose', recursive=False).text
                        truncated = int(obj.find('truncated', recursive=False).text)
                        if exclude_truncated and (truncated == 1): continue
                        difficult = int(obj.find('difficult', recursive=False).text)
                        if exclude_difficult and (difficult == 1): continue
                        # 得到bounding box的坐标.
                        bndbox = obj.find('bndbox', recursive=False)
                        xmin = int(bndbox.xmin.text)
                        ymin = int(bndbox.ymin.text)
                        xmax = int(bndbox.xmax.text)
                        ymax = int(bndbox.ymax.text)
                        item_dict = {'folder': folder,
                                     'image_name': filename,
                                     'image_id': image_id,
                                     'class_name': class_name,
                                     'class_id': class_id,
                                     #'pose': pose,
                                     #'truncated': truncated,
                                     #'difficult': difficult,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                        if difficult: eval_neutr.append(True)
                        else: eval_neutr.append(False)

                    self.labels.append(boxes)
                    self.eval_neutral.append(eval_neutr)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        
        if self.load_images_into_memory:
            self.images = []
            it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True):
        '''
        Arguments:
            images_dir (str): 图像目录
            labels_filename (str): csv路径，文件每一行为image file name, class ID, xmin, xmax, ymin, ymax.
            input_format (list):  输入标签的顺序'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'，
            include_classes (list, optional):'all' 或者数据集中包含类别的ID
            random_sample (float, optional): 随机采样，False生成器默认使用全部数据集，或者[0,1],随即采样数据集中的一部分
            ret (bool, optional): 是否返回parser的输出。
            verbose (bool, optional): 如果是`True`, 打印出可能需要较长时间操作的进度.

        Returns:
            默认返回空, 可选 images, image filenames, labels, and image IDs.
        '''

        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.include_classes = include_classes
        if self.labels_filename is None or self.input_format is None:
            raise ValueError("`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")
        # 清楚之前可能解析的数据
        self.filenames = []
        self.image_ids = []
        self.labels = []

        # 1.读取csv每行并且排序
        data = []
        with open(self.labels_filename, 'r') as csvfile:
        #with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            #next(csvread) # Skip the header row.
            for row in csvread: # 每行一个bbox
                if self.include_classes == 'all' or int(row[self.input_format.index('class_id')].strip()) in self.include_classes: # If the class_id is among the classes that are to be included in the dataset...
                    box = [] # 类别和坐标
                    box.append(row[self.input_format.index('image_name')].strip()) # Select the image name column in the input format and append its content to `box`
                    for element in self.labels_output_format: #('class_id', 'xmin', 'ymin', 'xmax', 'ymax')
                         # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                        box.append(int(row[self.input_format.index(element)].strip())) # ...select the respective column in the input format and append it to `box`.
                    data.append(box)
                    # data ('image_name')('class_id', 'xmin', 'ymin', 'xmax', 'ymax')

        data = sorted(data) # 根据图像名称排序

        # we can compile the actual samples and labels lists       
        current_file = data[0][0] # The current image for which we're collecting the ground truth boxes
        current_image_id = data[0][0].split('.')[0] # The image ID will be the portion of the image name before the first dot.
        current_labels = [] # The list where we collect all ground truth boxes for a given image
        add_to_dataset = False
        for i, box in enumerate(data):
            if box[0] == current_file: # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                        #第n张图像的labels，第n张图像文件名称，第n张图像id
            else: # If this box belongs to a new image file
                if random_sample: # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0,1)
                    if p >= (1-random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)
                current_labels = [] # Reset the labels list because this is a new file.
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            # if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret: # In case we want to return these
            return self.images, self.filenames, self.labels, self.image_ids


    def create_hdf5_dataset(self,
                            file_path='dataset.h5',
                            resize=False,
                            variable_image_size=True,
                            verbose=True):
        '''
        将当前加载的数据集转换为HDF5文件。


        此HDF5文件将所有图像作为未压缩数组存储在连续内存块中，从而可以更快地加载它们。

        这样一个未压缩的数据集，然而，在硬盘驱动器上占用的空间可能比压缩格式（如JPG或PNG）中的源图像总和大得多。
        推荐将数据转换成HDF5数据集，如果你有足够的硬盘空间，因为从HDF5加载数据显著增加数据生成的速度


        必须加载数据集（比如通过one of the parser methods），在从它建立HDF5之前

        这个建立的HDF5数据集会维持打开的状态，这样就可以立即使用

        参数:
            file_path (str, optional): 保存 HDF5 数据集的路径.
            resize (tuple, optional): `False` 或者 2-元组 `(高, 宽)`表示图像的目标尺寸。
            数据集中所有图像将会被resized到目标尺寸，在它们被写道HDF5文件之前。 如果是 `False`,不需要resize
            variable_image_size (bool, optional): 此参数的唯一目的是将其值存储在HDF5数据集中，
                                                  以便能够快速确定数据集中的图像是否都具有相同的大小。
            verbose (bool, optional): 是否打印出数据集创建的进程.

        Returns:
            None.
        '''

        self.hdf5_dataset_path = file_path

        dataset_size = len(self.filenames)

        # 建立HDF5文件.
        hdf5_dataset = h5py.File(file_path, 'w')

        # 创建一些属性，告诉我们此数据集包含的内容。
        # 数据集肯定是要包含图像的，但是也可能包含标签，图像id等
        hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_image_ids', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_eval_neutral', data=False, shape=None, dtype=np.bool_)
        # 快速检查数据集中的图像是否具有相同大小
        if variable_image_size and not resize:
            hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
        else:
            hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # 创建以展平数组存储图像的数据集，这使我们可以存储可变大小的图像。
        hdf5_images = hdf5_dataset.create_dataset(name='images',
                                                  shape=(dataset_size,),
                                                  maxshape=(None),
                                                  dtype=h5py.special_dtype(vlen=np.uint8))

        #创建数据集，该数据集将保存我们需要的图像高度，宽度和通道，以便稍后从展平的阵列重建图像。
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
                                                        shape=(dataset_size, 3),
                                                        maxshape=(None, 3),
                                                        dtype=np.int32)

        if not (self.labels is None):

            # 创建将标签以展平数组形式存储的数据集。
            hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                                      shape=(dataset_size,),
                                                      maxshape=(None),
                                                      dtype=h5py.special_dtype(vlen=np.int32))

            # 创建将保存每个图像的标签数组尺寸的数据集，以便我们稍后可以从展平的数组中恢复标签。
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes',
                                                            shape=(dataset_size, 2),
                                                            maxshape=(None, 2),
                                                            dtype=np.int32)

            hdf5_dataset.attrs.modify(name='has_labels', value=True)

        if not (self.image_ids is None):

            hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
                                                         shape=(dataset_size,),
                                                         maxshape=(None),
                                                         dtype=h5py.special_dtype(vlen=str))

            hdf5_dataset.attrs.modify(name='has_image_ids', value=True)

        if not (self.eval_neutral is None):

            # 创建一个数据集，其中标签以平铺数组的形式存储
            hdf5_eval_neutral = hdf5_dataset.create_dataset(name='eval_neutral',
                                                            shape=(dataset_size,),
                                                            maxshape=(None),
                                                            dtype=h5py.special_dtype(vlen=np.bool_))

            hdf5_dataset.attrs.modify(name='has_eval_neutral', value=True)

        tr = range(dataset_size)

        # 迭代数据集中所有图像.
        for i in tr:

            # 存储图像.
            with Image.open(self.filenames[i]) as image:

                image = np.asarray(image, dtype=np.uint8)

                # 确保所有的图像有三个通道.
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image] * 3, axis=-1)
                    elif image.shape[2] == 4:
                        image = image[:,:,:3]

                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]))

                # 展开图形数组并写入图像数据集中.
                hdf5_images[i] = image.reshape(-1)
                # 将图像shape写入图像shape的数据集中.
                hdf5_image_shapes[i] = image.shape

            # 存储ground truth，如果有的话.
            if not (self.labels is None):

                labels = np.asarray(self.labels[i])
                # 展开标签数组，并将其写入标签数据集.
                hdf5_labels[i] = labels.reshape(-1)
                # 将 labels的shape写入标签shapes数据集中.
                hdf5_label_shapes[i] = labels.shape

            # 如果我们有图片id的话，存储.
            if not (self.image_ids is None):

                hdf5_image_ids[i] = self.image_ids[i]

            # 如果我们有evaluation-neutrality annotations，存.
            if not (self.eval_neutral is None):

                hdf5_eval_neutral[i] = self.eval_neutral[i]

        hdf5_dataset.close()
        self.hdf5_dataset = h5py.File(file_path, 'r')
        self.hdf5_dataset_path = file_path
        self.dataset_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32) # Instead of shuffling the HDF5 dataset, we will shuffle this index list.

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        '''
        生成批量样本和对应的标签,shuffle,数据增强
        参数:
            batch_size (int, optional): 生成batch的大小
            shuffle (bool, optional): 在每次传递之前是否打乱数据集，训练时'True',调试或预测时可以关闭
            transformations (list, optional):将被应用到给定顺序的图像和标签上的transformations列表.
            label_encoder (callable, optional)：将标签从输入格式转换成训练需要的形式

            返回 (set, optional):生成器的输出 
                * 'processed_images': 处理过的图像，这个关键字你加不加都没关系，反正一定是会在输出中的。
                * 'encoded_labels': 编码后的标签
                * 'matched_anchors':只有当labels_encoder是一个SSDInputEncoder对象时可用。 
                    和'encoded_labels'相同,但包含所有匹配default box的坐标，而不是ground truth坐标。
                     这可以用于可视化与每个ground truth匹配的default box。
                    训练模式可用。
                * 'processed_labels': 经过处理后编码前的标签。
                * 'filenames': 文件名称（全路径）
                * 'image_ids': 图像ID
                * 'evaluation-neutral': A nested list of lists of booleans. Each list contains `True` or `False` for every ground truth
                    bounding box of the respective image depending on whether that bounding box is supposed to be evaluation-neutral (`True`)
                    or not (`False`). May return `None` if there exists no such concept for a given dataset. An example for
                    evaluation-neutrality are the ground truth boxes annotated as "difficult" in the Pascal VOC datasets, which are
                    usually treated to be neutral in a model evaluation.

                    布尔列表的嵌套列表。

                    每个列表包含相应图像的每个ground truth边界框的“True”或“False”，具
                    体取决于该边界框是否应该是评估中性（“True”）或不是（“False”）。

                    如果给定数据集不存在这样的概念，则可以返回“None”。
 
                    评估中立性的一个例子是在Pascal VOC数据集中注释为“difficult”的地面实况框，在模型评估中通常将其视为中性。
                * 'inverse_transform': 
                    嵌套列表，包含批次中每个项目的“逆变器”功能列表。
                     这些反相器函数将图像的（预测的）标签作为输入，并将应用于原始图像的变换的反转应用于它们。

                    这使得模型可以对变换后的图像进行预测，然后将这些预测转换回原始图像。

                    这主要与评估相关：如果要在具有不同图像大小的数据集上评估模型，则必须以某种方式（例如通过调整大小或裁剪）对图像进行变换，以使它们具有相同的大小。
                    模型将要在这些变形的图像上预测boxes,但是评估的时候，需要在原始图像上预测，而不是形变的图像，这意味着你需要将预测的坐标对应回原始图像
                    

                    请注意，对于每个图像，逆变器都起作用
                     图像需要按照在该图像的相应列表中给出的顺序应用。
                * 'original_images': 经过处理前的原始图像
                * 'original_labels': 经过处理前ground truth boxes.
                元组中输出的顺序是上面列表的顺序。 如果`returns`包含一个不可用的输出的关键字，那么在输出的元组中省略该输出并引发警告。
            keep_images_without_gt (bool, optional): 如果是False, 没有目标的图像将被移除
            degenerate_box_handling (str, optional): 如何处理不正常的boxes。'warn'警告或 'remove'移除

        Yields:
            下一批作为`returns`参数定义的tuple中的项目相同
        '''

        if self.dataset_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

        if self.labels is None:
            if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors', 'evaluation-neutral']]):
                warnings.warn("Since no labels were given, none of 'original_labels', 'processed_labels', 'evaluation-neutral', 'encoded_labels', and 'matched_anchors' " +
                              "are possible returns, but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif label_encoder is None:
            if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
                warnings.warn("Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif not isinstance(label_encoder, SSDInputEncoder):
            if 'matched_anchors' in returns:
                warnings.warn("`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))

        #############################################################################################
        #做一些准备工作，比如初始时打乱数据集。
        #############################################################################################

        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.eval_neutral is None):
                objects_to_shuffle.append(self.eval_neutral)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # 重写所有转换的标签格式以确保它们设置正确。
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        #############################################################################################
        # 生成小批量.
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_y = [], []
            if current >= self.dataset_size:
                current = 0
            #########################################################################################
            # 传递一次数据集后打乱
            #########################################################################################
                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    if not (self.eval_neutral is None):
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]
            #########################################################################################
            # 得到图像, (maybe) image IDs, (maybe) labels, etc. for this batch.
            # 1) 如果内存中已经加载图像，直接从中获取
            # 2) 否则，如果有HDF5数据集，从这里取出图像.
            # 3) 如果都没有,只能一张一张从磁盘加载图片了
            #########################################################################################
            batch_indices = self.dataset_indices[current:current+batch_size]
            if not (self.images is None):
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None
            elif not (self.hdf5_dataset is None):
                for i in batch_indices:
                    batch_X.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[current:current+batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            # 得到这一批的标签（如果有）.
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current+batch_size])
            else:
                batch_y = None

            if not (self.eval_neutral is None):
                batch_eval_neutral = self.eval_neutral[current:current+batch_size]
            else:
                batch_eval_neutral = None

            # 得到这一批的图像id (if there are any).
            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current+batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # 原始未改变的图像
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y) # 原始未改变的标签
 
            current += batch_size

            #########################################################################################
            # 数据增强.
            #########################################################################################

            batch_items_to_remove = [] # 如果我们要从batch中移除图像，存储它们的索引.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):

                if not (self.labels is None):
                    # 将图像的标签转成数组
                    batch_y[i] = np.array(batch_y[i])
                    # 如果这张图像没有ground truth boxes, 我们可能就不需要它了.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # 数据增强
                if transformations:
                    inverse_transforms = []
                    for transform in transformations:
                        if not (self.labels is None):
                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None: # 如果形变失败，没有生成任何图像
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:
                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # 检查这一批有没有不合理的boxes
                #########################################################################################

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                        if degenerate_box_handling == 'warn':
                            warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                          "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                          "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                          "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                          "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif degenerate_box_handling == 'remove':
                            batch_y[i] = box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                batch_items_to_remove.append(i)

            #########################################################################################
            # 移除这一批我们也许不想保留的项目
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # 这样做效率不高，但一般不需要经常这样做。
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # CAUTION: 注意：如果图像具有不同的大小或不同数量的通道，则将“batch_X”转换为数组将导致空批处理。 
            #                基于这点，所有图像必须具有相同的大小和相同数量的通道

            batch_X = np.array(batch_X)
            if (batch_X.size == 0):
                raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels. Note that after all transformations " +
                                           "(if any were given) have been applied to all images in the batch, all images " +
                                           "must be homogenous in size along all axes.")

            #########################################################################################
            # 如果我们有标签编码器，请编码我们的标签。
            #########################################################################################

            if not (label_encoder is None or self.labels is None):

                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None

            else:
                batch_y_encoded = None
                batch_matched_anchors = None



            ret = []
            if 'processed_images' in returns: ret.append(batch_X)
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)
            if 'matched_anchors' in returns: ret.append(batch_matched_anchors)
            if 'processed_labels' in returns: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
            if 'image_ids' in returns: ret.append(batch_image_ids)
            if 'evaluation-neutral' in returns: ret.append(batch_eval_neutral)
            if 'inverse_transform' in returns: ret.append(batch_inverse_transforms)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)

            yield ret

    def save_dataset(self,
                     filenames_path='filenames.pkl',
                     labels_path=None,
                     image_ids_path=None,
                     eval_neutral_path=None):
        '''
        将当前的 `filenames`, `labels`, 和 `image_ids` 列表写入具体的文件.
        对于有标注的大数据集非常有用，从xml文件中解析时间会很长。
        如果你将会重复使用同样的数据集，不需要每次都解析xml标记

        参数:
            filenames_path (str): 保存filenames的路径The path under which to save the filenames pickle.
            labels_path (str): 保存labels pickle的路径.
            image_ids_path (str, optional): 存储 image IDs pickle的路径.
            eval_neutral_path (str, optional): 存储 evaluation-neutrality annotations的路径
        '''
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        if not labels_path is None:
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        if not image_ids_path is None:
            with open(image_ids_path, 'wb') as f:
                pickle.dump(self.image_ids, f)
        if not eval_neutral_path is None:
            with open(eval_neutral_path, 'wb') as f:
                pickle.dump(self.eval_neutral, f)

    def get_dataset(self):
        '''
        Returns:
            4-元组 filenames, labels, image IDs,和evaluation-neutrality annotations.
            或者'None'
        '''
        return self.filenames, self.labels, self.image_ids, self.eval_neutral

    def get_dataset_size(self):
        '''
        Returns:
            数据集中图像数量
        '''
        return self.dataset_size
