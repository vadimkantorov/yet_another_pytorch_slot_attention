# Adapted from https://github.com/deepmind/multi_object_datasets/blob/master/clevr_with_masks.py

"""CLEVR (with masks) dataset reader."""

import tensorflow as tf


COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [240, 320]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.io.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.io.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.io.parse_single_example(example_proto, features)

  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.io.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


byte2word = dict(
    material = [None, 'rubber', 'metal'], 
    size     = [None, 'large', 'small'], 
    color    = [None, 'red', 'cyan', 'green', 'blue', 'brown', 'gray', 'purple', 'yellow'], 
    shape    = [None, 'sphere', 'cylinder', 'cube']
)

if __name__ == '__main__':
    import os
    import json
    import random
    import argparse
    import numpy as np
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-tfrecords', '-i', required = True)
    parser.add_argument('--output-dir', '-o', required = True)
    parser.add_argument('--split-name', default = 'train')
    parser.add_argument('--max-num-objects', type = int)
    parser.add_argument('--begin-index', type = int)
    parser.add_argument('--end-index', type = int)
    parser.add_argument('--keep-prob', type = float, default = 1.0)
    args = parser.parse_args()
    
    images_dir = os.path.join(args.output_dir, 'images', args.split_name)
    masks_dir = os.path.join(args.output_dir, 'masks', args.split_name)
    scenes_json = os.path.join(args.output_dir, 'scenes', f'CLEVR_{args.split_name}_scenes.json')
    
    for d in [images_dir, masks_dir, os.path.dirname(scenes_json)]:
        os.makedirs(d, exist_ok = True)

    scenes = []
    
    for i, example in enumerate(map(_decode, tf.compat.v1.io.tf_record_iterator(args.input_tfrecords, COMPRESSION_TYPE))):
        if (args.begin_index is not None and i < args.begin_index) or (args.end_index is not None and i >= args.end_index):
            continue

        image_file_name = f'CLEVR_{args.split_name}_{i:06d}.png'
        
        example = {k : v.numpy() for k, v in example.items()}
        example['padding'] = example['visibility'] == 0.0

        num_objects = (example['visibility'] == 1.0).sum()
        num_objects_with_padding = len(example['padding'])

        if args.max_num_objects is not None and num_objects > args.max_num_objects:
            continue

        if not (args.keep_prob == 1 or random.random() <= args.keep_prob):
            continue

        cv2.imwrite(os.path.join(images_dir, image_file_name), example.pop('image')[..., ::-1])
        np.save(os.path.join(masks_dir, image_file_name.replace('.png', '.npy')), (example.pop('mask') / 255).squeeze(-1))

        example = {k : v.tolist() for k, v in example.items()}
        s = dict(
            image_index = i,
            image_filename = image_file_name,
            split = args.split_name,
            objects = [{k : v[i] if k not in byte2word else byte2word[k][v[i]] for k, v in example.items()} for i in range(num_objects_with_padding)],
        )

        scenes.append(s)
        print(i)

    json.dump(dict(info = dict(split = args.split_name), scenes = scenes) , open(scenes_json, 'w'), indent = 2)
    
    print(args.output_dir)
    print('Number of files:', len(scenes))
