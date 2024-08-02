# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_h5dataset.ipynb (unless otherwise specified).

__all__ = ['logger', 'H5DatasetIterator', 'H5DatasetLoader']

# Cell

import numpy as np
import h5py
# import tensorflow as tf

import tempfile

try:
    import torch
    from torch.utils.data import Dataset
    __has_torch = True
except ImportError:
    __has_torch = False


# import mmwave

import logging

logger = logging.getLogger()

# Cell

class H5DatasetIterator():
    """Iterates through aligned frames dataset"""
    def __init__(self, dset, streams):
        super().__init__()
        self._dset = dset
        self._idx = 0
        self.req_streams = streams

    def __iter__(self):
        return self

    def __next__(self):
        self._idx += 1
        try:
            return tuple(self._dset[s][self._idx] for s in self.req_streams)
        except IndexError:
            self._idx = 0
            raise StopIteration

    def __len__(self):
        return len(self._dset)

class H5DatasetLoader(object):
    """A thin wrapper around h5py to provide convenience functions for training"""
    def __init__(self, filenames, default_streams=None):
        super(H5DatasetLoader, self).__init__()
        self.filenames = filenames
        if isinstance(self.filenames, list):
            self._h5_tempfile = tempfile.NamedTemporaryFile()
            #self.h5_file = h5py.File(self._h5_tempfile, 'w', libver='latest')

            self._allfiles, _allstreams, _lengths = zip(*[H5DatasetLoader.load_single_h5(f) for f in self.filenames])

            total_len = sum(_lengths)

            #create virtual datasets of, assumes that all files have the streams of first file and shape of first file
            ll = (0,) + _lengths
            ll = np.cumsum(ll)
            for s in _allstreams[0]:
                shape = (total_len, ) + self._allfiles[0][s].shape[1:]
                layout = h5py.VirtualLayout(shape=shape, dtype=self._allfiles[0][s].dtype)

                for idx, f in enumerate(self._allfiles):
                    vsource = h5py.VirtualSource(f[s])
                    layout[ll[idx]:ll[idx+1]] = vsource

                with h5py.File(self._h5_tempfile.name, 'a', libver='latest') as f:
                    f.create_virtual_dataset(s, layout,)
            self._h5_tempfile.flush()
            self.h5_file = H5DatasetLoader.load_single_h5(self._h5_tempfile.name)[0]
        else:
            self.h5_file = H5DatasetLoader.load_single_h5(self.filenames)[0]
        self.streams_available = list(self.h5_file.keys())
        self.default_streams = default_streams

        if default_streams is not None:
            for s in default_streams:
                assert s in self.streams_available, f"{s} not found in available streams"

    @staticmethod
    def load_single_h5(filename):
        h5 = h5py.File(filename, 'r')
        streams_available = list(h5.keys())
        dataset_len = len(h5[streams_available[0]])
        return h5, streams_available, dataset_len

    def __len__(self):
        return len(self.h5_file[self.streams_available[0]])

    def __getitem__(self, stream):
        return self.h5_file[stream]

    def get_iterator(self, streams=None):
        """The default iterator includes all available streams in the order available on the h5 file"""
        if not streams:
            streams = self.default_streams.copy() if self.default_streams is not None else self.streams_available.copy()
        return H5DatasetIterator(self, streams)

    def __iter__(self):
        return self.get_iterator()

    @property
    def filename(self):
        return self.filenames

    def get_torch_dataset(self, streams=None):
        try:
            return RadicalDatasetTorch(self, streams)
        except NameError:
            raise RuntimeError('Torch is not available')


    def get_tf_dataset(self,
                       streams=None,
                       shuffle=False,
                       repeat=False,
                       batchsize=16,
                       preprocess_chain=None,
                       prefetch=2,
                       flatten_single=False,
                      ):
        logger.debug("Tensorflow Dataset creation")
        if streams is None:
            streams = ['radar', 'rgb', 'depth']

        out_shapes = tuple([
            tf.TensorShape(list(self.h5_file[s].shape[1:])) for s in streams
        ])
        out_types = tuple([self.h5_file[s].dtype for s in streams])

        def _gen():
            for i in range(len(self.h5_file[streams[0]])):
                yield tuple(self.h5_file[s][i] for s in streams)

        _dataset = tf.data.Dataset.from_generator(
            _gen,
            output_types = out_types,
            output_shapes = out_shapes,
        )

        if shuffle:
            logger.debug("  Outputs of dataset will be shuffled")
            _dataset = _dataset.shuffle(batchsize * 4)

        if repeat:
            logger.debug(f'  Dataset will be repeated {repeat} files')
            _dataset = _dataset.repeat(repeat)

        if preprocess_chain is not None:
            for op in preprocess_chain:
                _dataset = _dataset.map(op)

        if flatten_single:
            assert(len(streams) == 1)
            logger.debug("  Flattening shapes for single stream inference")
            logger.debug(_dataset)
            _dataset = _dataset.map(lambda x: x)
            logger.debug(_dataset)

        _dataset = _dataset.batch(batchsize)


        if prefetch:
            _dataset = _dataset.prefetch(prefetch)

        return _dataset

# Cell
if __has_torch:
    class RadicalDatasetTorch(Dataset):
        def __init__(self, src_dataset, streams=None, transforms=None):
            self.__src_dataset = src_dataset
            if streams is not None:
                self.__streams = streams
            else:
                self.__streams = ['radar', 'rgb', 'depth']

        def __len__(self):
            return len(self.__src_dataset)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            sample = {s:self.__src_dataset[s][idx, ...] for s in self.__streams}
            return sample