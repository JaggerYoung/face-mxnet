import os
import mxnet
import mxnet.ndarray as nd
import numpy as np

class CenterlossOutput(mx.operator.CustomOp):
    def __init__(self,):
        pass

    def forward(self, is_train, req, in_data, out_data, aux):
        pass

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register('CenterlossOutput')
class CenterlossOutputProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CenterlossOutputProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        pass

    def infer_type(self, in_type):
        pass

    def create_operator(self, ctx, shapes, dtypes):
        pass
