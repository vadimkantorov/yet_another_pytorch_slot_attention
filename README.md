Another reimplementation of [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055) by Locatello et al (Ojbect Discovery task).

Original TFv2 implementation is at https://github.com/google-research/google-research/tree/master/slot_attention/

Training loop code is adapted from https://github.com/evelinehong/slot-attention-pytorch/ by Yining Hong

ARI computation and CLEVRWithMasks code is adapted from https://github.com/deepmind/multi_object_datasets

```shell
# download original checkpoint (converted to PyTorch)
wget https://github.com/vadimkantorov/yet_another_pytorch_slot_attention/releases/download/data/slot-attention_object_discovery.pt

# infer on CPU with original checkpoint
python infer.py --device cpu --checkpoint-tensorflow slot-attention_object_discovery.pt

# download original CLEVR dataset (18 Gb)
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip

# and train the model on GPU from scratch
python train.py --device cuda

# download CLEVRWithMasks (10 Gb)
wget https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords
# install tensorflow with caution and unpack CLEVRWithMasks to file system
pip install tensorflow

# generate CLEVR6train, CLEVR6val, CLEVR10val
# python clevr_with_masks.py -i clevr_with_masks_train.tfrecords -o ./CLEVR_with_masks --split-name CLEVR6train --begin-index 0 --end-index 70000 --max-num-objects 6
python clevr_with_masks.py -i clevr_with_masks_train.tfrecords -o ./CLEVR_with_masks --split-name CLEVR6val --begin-index 70000 --end-index 100000 --max-num-objects 7 --keep-prob 0.1
python clevr_with_masks.py -i clevr_with_masks_train.tfrecords -o ./CLEVR_with_masks --split-name CLEVR10val --begin-index 70000 --end-index 100000 --max-num-objects 11 --keep-prob 0.1

python eval.py --dataset-root-dir ./CLEVR_with_masks --split-name CLEVR6val --checkpoint-tensorflow slot-attention_object_discovery.pt
```
