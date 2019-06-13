#--coding: utf-8--
'''
The Keras-compatible loss function for the SSD model. Currently supports TensorFlow only.

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
import tensorflow as tf

class SSDLoss:
    '''
    SSD损失, 参见https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        参数:
            neg_pos_ratio (int, optional):参与损失计算的负样本（背景）和正样本（ground truth boxes）的比例。 
  				实际上ground tru778rth boxes中没有真正的背景，但是default boxes中包含标记为背景类的boxes.          
                由于y_true中背景boxes的数量一般远远超过正样本boxes的数量，在损失中平衡它们的影响是一件很重要的事情。
                根据论文默认设置为3。

            n_neg_min (int, optional):每一批（per batch)参与损失计算的negative ground truth boxes最小数量。
                这个参数，用来确保模型从batches从很少数量的负样本中学习，
                这个batches中包括非常少，或者一点也没有，正ground truth boxes.
                默认为0，如果使用，应该设置为一个值，代替训练使用的batch size的合理的比例。

            alpha (float, optional):计算总损失时，用来衡量位置损失所占权重，根据论文默认设置为1.0。
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        参数:
            y_true (nD tensor):(batch_size,#boxes,4)，最后一维包括`(xmin, xmax, ymin, ymax)`。
            y_pred (nD tensor): 和'y_true'有相同结构的一个Tensorflow的张量，包含预测的bounding box的坐标。

        返回:
            smooth L1损失，2维张量(batch, n_boxes_total)
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        参数:
            y_true (nD tensor): (batch_size, #boxes, #classes)
            y_pred (nD tensor):和'y_true'结构相同，包含预测的bounding box的类别
        返回:
            softmax对数损失,(batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        参数:
            y_true (array): 数组shape`(batch_size, #boxes, #classes + 12)`,
                
                #boxes是模型预测的每张图像的boxes的总数
                注意确认y_true的box索引值和对应的y-pred中box索引相同
                
               '#classes + 12'包括[类别one-hot编码,4个真实box的坐标偏移值，8个任意条目]
                在这里, 包含背景分类。

                这个函数没有用到最后一个维度的eight entries，因此它们的内容是什么都没有关系
                它们存在的意义仅仅是使`y_true`和`y_pred`有相同的shape,
                
                重要：损失函数忽略的boxes是全零的一个one-hot类别向量
                
            y_pred (Keras tensor): 模型预测，shape和y_true是相同的 `(batch_size, #boxes, #classes + 12)`.
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

        Returns:
            多任务损失
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0] # 输出类型: tf.int32
        n_boxes = tf.shape(y_pred)[1] # 输出类型: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.

        # 1: 计算每个box的类别和box预测的损失

        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])) # Output shape: (batch_size, n_boxes)

        # 2: 计算正样本和负样本的类别损失

        #创建正负样本的masks
        negatives = y_true[:,:,0] # Tensor of shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1)) # Tensor of shape (batch_size, n_boxes)

        # 计算整个batch中y_true中的正样本的数量
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes).
        # 现在mask所有负样本，将每个batch中正样本的损失相加（Keras 的损失函数每一个batch输出一个标量，
        # 而不是整个batch输出一个标量，这是我们为什么不summing across all axes
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 计算负样本的分类损失（如果有的话）

        #首先，计算所有负样本的分类损失
        neg_class_loss_all = classification_loss * negatives # shape为(batch_size, n_boxes)的张量
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # 负样本非零损失
        #`n_neg_losses`有什么意义？
        # For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, 
        # but for how many of those there actually is a positive (i.e. non-zero) loss. 

        # 下一步，将计算那些负样本参与了损失，我们不只想知道有多少负样本，还想知道其中有多少损失是正数（非零）。
        # 这很重要因为下面函数中的`tf.nn.top-k()`会选出k个损失最高的k个boxes，即使损失是0

        # 虽然所有的负样本分类损失都是零的事件不大可能发生，这种行为可能会导致`tf.nn.top-k()` 返回正样本的索引，
        # 导致负样本分类损失计算错误，使整个损失函数的计算出现错误
        
        # 因此我们需要确保`n_negative_keep`，它在`tf.nn.top-k（）`中扮演`k`参数的角色，
        # 最多是负样本的数量，其中存在正分类丢失。

        # 还要计算参与损失函数计算的负样本的数量
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        # 概率很小的情况(1) 没有一个负样本
        # (2) 所有负样本的分类损失都是零, 返回零作为负样本分类损失`neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])
        # 否则计算负样本损失.
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # 为了完成这个目标, 将`neg_class_loss_all` 转成1维...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
            # ...那么我们得到了那些具有最高损失的`n_negative_keep`框的索引...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False) # 不需要排序.
            # ...使用这些索引，我们将创建一个mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
            # ...使用这个mask只保留这些框的损失 ，并且mask所有其它分类损失
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

        # 3:计算正样本的位置损失， 不计算负样本的位置损失

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: 计算全部的损失.

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) 
        # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
