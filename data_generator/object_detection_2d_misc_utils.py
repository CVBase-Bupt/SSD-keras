'''
Miscellaneous data generator utilities.

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

def apply_inverse_transforms(y_pred_decoded, inverse_transforms):
    '''
    将变换后的图像的预测值，转换回原始图像对应的预测值。

    Arguments:
        y_pred_decoded (list or array):(num_predictions, 6)/(batch_size, num_predictions, 6)`. 
            最后一维 类别ID, 类别分数, 四个坐标 
        inverse_predictions (list): A nested list of length `batch_size` that contains
            for each batch item a list of functions that take one argument (one element
            of `y_pred_decoded` if it is a list or one slice along the first axis of
            `y_pred_decoded` if it is an array) and return an output of the same shape
            and data type.

    Returns:
        The transformed predictions, which have the same structure as `y_pred_decoded`.
    '''

    if isinstance(y_pred_decoded, list):

        y_pred_decoded_inv = []

        for i in range(len(y_pred_decoded)):
            y_pred_decoded_inv.append(np.copy(y_pred_decoded[i]))
            if y_pred_decoded_inv[i].size > 0: # If there are any predictions for this batch item.
                for inverter in inverse_transforms[i]:
                    if not (inverter is None):
                        y_pred_decoded_inv[i] = inverter(y_pred_decoded_inv[i])

    elif isinstance(y_pred_decoded, np.ndarray):

        y_pred_decoded_inv = np.copy(y_pred_decoded)

        for i in range(len(y_pred_decoded)):
            if y_pred_decoded_inv[i].size > 0: # If there are any predictions for this batch item.
                for inverter in inverse_transforms[i]:
                    if not (inverter is None):
                        y_pred_decoded_inv[i] = inverter(y_pred_decoded_inv[i])

    else:
        raise ValueError("`y_pred_decoded` must be either a list or a Numpy array.")

    return y_pred_decoded_inv
