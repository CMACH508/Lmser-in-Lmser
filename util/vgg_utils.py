from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import numpy as np
import math
from PIL import Image


def resize(im, size):
    x, y = im.size

    y = int(max(y * size[0] / x, 1))
    x = int(size[0])
    if y > size[1]:
        x = int(max(x * size[1] / y, 1))
        y = int(size[1])
    size = x, y

    if im.size != size:
        im = im.resize(size, resample=Image.BILINEAR)

    return im


class DataLoader(object):
    """Class for loading data."""

    def __init__(self,
                 photo_dir,
                 photo_names,
                 labels,
                 in_size=(200, 250),
                 pad_size=(286, 286),
                 out_size=(256, 256),
                 batch_size=100,
                 is_train=True):
        self.photo_dir = photo_dir
        self.photo_names = photo_names
        self.labels = labels
        self.sample_len = len(self.photo_names)
        self.batch_size = batch_size  # batch size
        self.num_batches = math.ceil(self.sample_len / self.batch_size)
        self.in_size = in_size
        self.pad_size = pad_size
        self.out_size = out_size
        self.is_train = is_train

    def _process(self, batch):
        new_batch = []
        for i, sample in enumerate(batch):
            ori_size = sample.size
            assert sample.mode in ["RGB", "L"]
            if ori_size != self.in_size:
                sample = resize(sample, self.in_size)

            if sample.mode == "L":
                new_sample = np.full(self.pad_size, fill_value=255, dtype=np.float32)
            else:
                new_sample = np.full((self.pad_size[1], self.pad_size[0], 3), fill_value=255, dtype=np.float32)

            w_s = int((self.pad_size[0] - self.in_size[0]) / 2)
            w_e = int((self.pad_size[0] + self.in_size[0]) / 2)
            h_s = int((self.pad_size[1] - self.in_size[1]) / 2)
            h_e = int((self.pad_size[1] + self.in_size[1]) / 2)
            new_sample[h_s:h_e, w_s:w_e] = np.array(sample, dtype=np.float32)

            if self.is_train:
                h_offset = random.randint(0, self.pad_size[1] - self.out_size[1] - 1)
                w_offset = random.randint(0, self.pad_size[0] - self.out_size[0] - 1)
            else:
                h_offset = int((self.pad_size[1] - self.out_size[1]) / 2)
                w_offset = int((self.pad_size[0] - self.out_size[0]) / 2)
            new_sample = new_sample[h_offset:h_offset + self.out_size[1], w_offset:w_offset + self.out_size[0]]
            new_batch.append(new_sample)

        return new_batch

    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        batch_photo_paths = list()
        batch_label = list()
        for idx in indices:
            photo_path = os.path.join(self.photo_dir, self.photo_names[idx])
            batch_photo_paths.append(photo_path)
            batch_label.append(self.labels[idx])

        batch_photo_im = self.load_images(batch_photo_paths, "L")
        batch_photo = self._process(batch_photo_im)
        return batch_photo, batch_label

    def random_batch(self):
        """Return a randomised portion of the training data."""
        idx = np.random.permutation(range(0, self.sample_len))[0:self.batch_size]
        return self._get_batch_from_indices(idx)

    def get_batch(self, idx):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        if end_idx > self.sample_len - 1:
            end_idx = self.sample_len - 1
        indices = range(start_idx, end_idx)
        return self._get_batch_from_indices(indices)

    def load_images(self, photo_paths, mode):
        assert mode in ["RGB", "L"]
        img_batch = []
        for img_idx in range(len(photo_paths)):
            image_path = photo_paths[img_idx]
            image = Image.open(image_path).convert(mode=mode)
            img_batch.append(image)

        return img_batch
