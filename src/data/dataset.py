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
"""Face Recognition Dataset"""
import os
import pickle
import mxnet as mx
import numpy as np
from mxnet.gluon.data import Dataset
from mxnet.gluon.data import RecordFileDataset
from mxnet import image, recordio

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

__all__ = ["FRValDataset",
           "FRTrainRecordDataset",
           "get_recognition_dataset",
           "DaliDataset"
           ]


class FRTrainRecordDataset(RecordFileDataset):
    """A dataset wrapping over a rec serialized file provided by InsightFace Repo.

    Parameters
    ----------
    name : str. Name of val dataset.
    root : str. Path to face folder. Default is '$(HOME)/mxnet/datasets/face'
    transform : function, default None
        A user defined callback that transforms each sample. For example:
    ::
        transform=lambda data, label: (data.astype(np.float32)/255, label)
    """

    def __init__(self, name, root=os.path.expanduser('~/.mxnet/datasets/face'), flag=1, transform=None):
        super().__init__(os.path.join(root, name, "train.rec"))
        prop = open(os.path.join(root, name, "property"), "r").read().strip().split(',')
        self._flag = flag
        self._transform = transform

        assert len(prop) == 3
        self.num_classes = int(prop[0])
        self.image_size = [int(prop[1]), int(prop[2])]

    def __getitem__(self, idx):
        while True:
            record = super().__getitem__(idx)
            header, img = recordio.unpack(record)
            if _check_valid_image(img):
                decoded_img = image.imdecode(img, self._flag)
            else:
                idx = np.random.randint(low=0, high=len(self))
                continue
            if self._transform is not None:
                return self._transform(decoded_img, header.label)
            return decoded_img, header.label


def _check_valid_image(s):
    return False if len(s) == 0 else True


class FRValDataset(Dataset):
    """A dataset wrapping over a pickle serialized (.bin) file provided by InsightFace Repo.

    Parameters
    ----------
    name : str. Name of val dataset.
    root : str. Path to face folder. Default is '$(HOME)/mxnet/datasets/face'
    transform : callable, default None
        A function that takes data and transforms them:
    ::
        transform = lambda data: data.astype(np.float32)/255

    """

    def __init__(self, name, root=os.path.expanduser('~/.mxnet/datasets/face'), transform=None):
        super().__init__()
        self._transform = transform
        self.name = name
        with open(os.path.join(root, "{}.bin".format(name)), 'rb') as f:
            self.bins, self.issame_list = pickle.load(f, encoding='iso-8859-1')

        self._do_encode = not isinstance(self.bins[0], np.ndarray)

    def __getitem__(self, idx):
        img0 = self._decode(self.bins[2 * idx])
        img1 = self._decode(self.bins[2 * idx + 1])

        issame = 1 if self.issame_list[idx] else 0

        if self._transform is not None:
            img0 = self._transform(img0)
            img1 = self._transform(img1)

        return (img0, img1), issame

    def __len__(self):
        return len(self.issame_list)

    def _decode(self, im):
        if self._do_encode:
            im = im.encode("iso-8859-1")
        return mx.image.imdecode(im)


datasets = {"lfw": FRValDataset,
            "calfw": FRValDataset,
            "cplfw": FRValDataset,
            "cfp_fp": FRValDataset,
            "agedb_30": FRValDataset,
            "cfp_ff": FRValDataset,
            "vgg2_fp": FRValDataset,
            "emore": FRTrainRecordDataset,
            "vgg": FRTrainRecordDataset
            }


def get_recognition_dataset(name, **kwargs):
    return datasets[name](name, **kwargs)


class DaliDataset(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, name, num_gpu,
                 root=os.path.expanduser('~/.mxnet/datasets/face'), ):
        super().__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        idx_files = [os.path.join(root, name, "train.idx")]
        rec_files = [os.path.join(root, name, "train.rec")]
        prop = open(os.path.join(root, name, "property"), "r").read().strip().split(',')
        assert len(prop) == 3
        self.num_classes = int(prop[0])
        self.image_size = [int(prop[1]), int(prop[2])]

        self._input = ops.MXNetReader(path=rec_files, index_path=idx_files, random_shuffle=True,
                                      num_shards=num_gpu, tensor_init_bytes=self.image_size[0] * self.image_size[1] * 8)
        self._decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self._cmnp = ops.CropMirrorNormalize(device="gpu",
                                             output_dtype=types.FLOAT,
                                             output_layout=types.NCHW,
                                             crop=self.image_size,
                                             image_type=types.RGB,
                                             mean=[127.5, 127.5, 127.5],
                                             std=[127.5, 127.5, 127.5])
        self._contrast = ops.Contrast(device="gpu", )
        self._saturation = ops.Saturation(device="gpu", )
        self._brightness = ops.Brightness(device="gpu", )

        self._uniform = ops.Uniform(range=(0.7, 1.3))
        self._coin = ops.CoinFlip(probability=0.5)
        self.iter = 0

    def define_graph(self):
        inputs, labels = self._input(name="Reader")
        images = self._decode(inputs)

        images = self._contrast(images, contrast=self._uniform())
        images = self._saturation(images, saturation=self._uniform())
        images = self._brightness(images, brightness=self._uniform())

        output = self._cmnp(images, mirror=self._coin())
        return output, labels

    def iter_setup(self):
        pass
