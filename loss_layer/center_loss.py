# A Discriminative Feature Learning Approach for Deep Face Recognition
import os
from collections import defaultdict
import mxnet
import mxnet.ndarray as nd
import numpy as np

class CenterlossOutput(mx.operator.CustomOp):
    '''
    alpha: update learning rate for center loss grad
    scale: scale for grad output
    class_index: store map of center and class index
    '''
    def __init__(self, alpha, scale, class_index):
        self.alpha = alpha
        self.scale = scale
        self.class_index = class_index

    def forward(self, is_train, req, in_data, out_data, aux):
        data_input = in_data[0]
        batch_size = data_input.shape[0]
        label_input = in_data[1]
        center_input = in_data[2]

        label_index = self.class_index[label_input]
        batch_center = center_input[label_index]
        batch_diff = data_input - batch_center
        
        loss = nd.sum(nd.square(batch_diff)) / batch_size / 2
        self.assign(out_data[0], req[0], loss)
        self.assign(out_data[1], req[0], batch_diff)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        center_input = in_data[2]
        batch_diff = out_data[1]
        batch_size = batch_diff.shape[0]
        batch_scale = float(self.scale/batch_size)
        sum_grad = batch_diff[0].copy()
        self.assign(in_grad[0], req[0], batch_diff*batch_scale)

        labels = in_data[1].asnumpy()
        label_dict = defaultdict(list)

        for i,label in enumerate(labels):
            label_dict[int(label)].append(i)

        for label, index in label_dict.items():
            sum_grad = nd.sum(batch_diff[index], axis=0)
            center_grad = sum_grad / (1+len(sample_index))
            center_input[self.class_index[label]] += self.alpha * center_grad

@mx.operator.register('CenterlossOutput')
class CenterlossOutputProp(mx.operator.CustomOpProp):
    def __init__(self, alpha, scale, class_index):
        super(CenterlossOutputProp, self).__init__(need_top_grad=False)
        self.alpha = alpha
        self.scale = scale
        self.class_index = class_index

    def list_arguments(self):
        return ['data', 'labels', 'center']

    def list_outputs(self):
        return ['output', 'diff']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        center_shape = in_shape[2]
        output_shape = (input_shape[0],)
        diff_shape = (input_shape[0],)
        return [data_shape, label_shape, center_shape], [output_shape, diff_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CenterlossOutput(ctx, shapes, dtypes, self.alpha, self.scale, self.class_index)
