# tfrecord_iterator, tfrecord_loader, TFRecordDataset adapted from https://github.com/vahidk/tfrecord/

import os
import typing
import json
import functools
import io
import os
import struct
import typing

import numpy as np

import torch
import torchvision

import example_pb2


def tfrecord_iterator(data_path: str,
                      shard: typing.Optional[typing.Tuple[int, int]] = None
                      ) -> typing.Iterable[memoryview]:
    """Create an iterator over the tfrecord dataset.
    Since the tfrecords file stores each example as bytes, we can
    define an iterator over `datum_bytes_view`, which is a memoryview
    object referencing the bytes.
    Params:
    -------
    data_path: str
        TFRecord file path.
    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).
    Yields:
    -------
    datum_bytes_view: memoryview
        Object referencing the specified `datum_bytes` contained in the
        file (for a single record).
    """
    file = io.open(data_path, "rb")

    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)

    def read_records(start_offset=None, end_offset=None):
        nonlocal length_bytes, crc_bytes, datum_bytes

        if start_offset is not None:
            file.seek(start_offset)
        if end_offset is None:
            end_offset = os.path.getsize(data_path)
        while file.tell() < end_offset:
            if file.readinto(length_bytes) != 8:
                raise RuntimeError("Failed to read the record size.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the start token.")
            length, = struct.unpack("<Q", length_bytes)
            if length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:length]
            if file.readinto(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")
            yield datum_bytes_view

    yield from read_records()

    file.close()



def tfrecord_loader(data_path: str,
                    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                    shard: typing.Optional[typing.Tuple[int, int]] = None,
                    ) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.
    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.
    Params:
    -------
    data_path: str
        TFRecord file path.
    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.
    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).
    Yields:
    -------
    features: dict of {str, np.ndarray}
        Decoded bytes of the features into its respective data type (for
        an individual record).
    """

    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    record_iterator = tfrecord_iterator(data_path, shard)

    for record in record_iterator:
        example = example_pb2.Example()
        example.ParseFromString(record)

        all_keys = list(example.features.feature.keys())
        if description is None:
            description = dict.fromkeys(all_keys, None)
        elif isinstance(description, list):
            description = dict.fromkeys(description, None)

        features = {}
        for key, typename in description.items():
            if key not in all_keys:
                raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")
            # NOTE: We assume that each key in the example has only one field
            # (either "bytes_list", "float_list", or "int64_list")!
            field = example.features.feature[key].ListFields()[0]
            inferred_typename, value = field[0].name, field[1].value
            if typename is not None:
                tf_typename = typename_mapping[typename]
                if tf_typename != inferred_typename:
                    reversed_mapping = {v: k for k, v in typename_mapping.items()}
                    raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                                    f"(should be '{reversed_mapping[inferred_typename]}').")

            # Decode raw bytes into respective data types
            if inferred_typename == "bytes_list":
                value = np.frombuffer(value[0], dtype=np.uint8)
            elif inferred_typename == "float_list":
                value = np.array(value, dtype=np.float32)
            elif inferred_typename == "int64_list":
                value = np.array(value, dtype=np.int32)
            features[key] = value

        yield features

def shuffle_iterator(iterator: typing.Iterator,
                     queue_size: int) -> typing.Iterable[typing.Any]:
    """Shuffle elements contained in an iterator.
    Params:
    -------
    iterator: iterator
        The iterator.
    queue_size: int
        Length of buffer. Determines how many records are queued to
        sample from.
    Yields:
    -------
    item: Any
        Decoded bytes of the features into its respective data type (for
        an individual record) from an iterator.
    """
    buffer = []
    try:
        for _ in range(queue_size):
            buffer.append(next(iterator))
    except StopIteration:
        warnings.warn("Number of elements in the iterator is less than the "
                      f"queue size (N={queue_size}).")
    while buffer:
        index = np.random.randint(len(buffer))
        try:
            item = buffer[index]
            buffer[index] = next(iterator)
            yield item
        except StopIteration:
            yield buffer.pop(index)

class TFRecordDataset(torch.utils.data.IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s.
    Params:
    -------
    data_path: str
        The path to the tfrecords file.
    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.
    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.
    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.
    """

    def __init__(self,
                 data_path: str,
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 shuffle_queue_size: typing.Optional[int] = None,
                 transform: typing.Callable[[dict], typing.Any] = None
                 ) -> None:
        super(TFRecordDataset, self).__init__()
        self.data_path = data_path
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform or (lambda x: x)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = tfrecord_loader(
            self.data_path, self.description, shard)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it

class CLEVRWithMasks(torchvision.datasets.VisionDataset):
    def __init__(self, root, split_name, transform = torchvision.transforms.ToTensor(), loader = torchvision.datasets.folder.default_loader, filter = None):
        super().__init__(root, transform = transform)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return os.path.basename(path), sample
