#FaceNet- A Unified Embedding for Face Recognition and Clustering
import os
import mxnet
import mxnet.ndarray as nd
import numpy as np

class TripletLoss(mx.operator.CumstomOp):
    def __init__(self,):
        pass

    def forward(self, is_train, req, in_data, out_data, aux):
        pass

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register('TripletLoss'):
class TripletLossProp(mx.operator.CustomOpProp):
    def __init__(self,):
        super(TripletLossProp, self).__init__(False)

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        pass

    def create_operator(self, ctx, shapes, dtypes):
        return TripletLoss(ctx, shapes, dtypes)
