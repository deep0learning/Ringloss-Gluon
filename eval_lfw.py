# MIT License
#
# Copyright (c) 2018 Haoxintong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Eval accuracy of LFW or other datasets"""
import sklearn
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.nn import SymbolBlock
from mxnet.gluon.data import DataLoader

from src.utils import transform_test
from src.data.dataset import get_recognition_dataset
from src.data.verification import FaceVerification

ctx = [mx.gpu()]

net = SymbolBlock.imports(symbol_file="models/mobile_facenet/model-symbol.json", input_names=["data"],
                          param_file="models/mobile_facenet/model-0000.params", ctx=ctx)
# net.hybridize()
net.summary(mx.nd.ones(shape=(1, 3, 112, 112), ctx=mx.gpu()))

targets = ['lfw']
val_sets = [get_recognition_dataset(name, transform=transform_test) for name in targets]
val_datas = [DataLoader(dataset, 128, num_workers=2) for dataset in val_sets]

metric = FaceVerification(nfolds=10)
results = []
for loader, name in zip(val_datas, targets):
    metric.reset()
    for i, batch in enumerate(loader):
        data0s = gluon.utils.split_and_load(batch[0][0], ctx, even_split=False)
        data1s = gluon.utils.split_and_load(batch[0][1], ctx, even_split=False)
        issame_list = gluon.utils.split_and_load(batch[1], ctx, even_split=False)

        embedding0s = [net(X) for X in data0s]
        embedding1s = [net(X) for X in data1s]

        embedding0s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding0s]
        embedding1s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding1s]

        for embedding0, embedding1, issame in zip(embedding0s, embedding1s, issame_list):
            metric.update(issame, embedding0, embedding1)

    tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
    print("{}: {:.6f}+-{:.6f}".format(name, accuracy, accuracy_std))
