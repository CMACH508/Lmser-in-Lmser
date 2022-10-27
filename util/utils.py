from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
import math
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


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


def draw_image(batch, size, o_dir):
    num = len(batch)
    fig = plt.figure()
    for j in range(num):
        im = batch[j]
        fig.add_subplot(1, num, j + 1)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        im = Image.fromarray(np.uint8(im)).convert(mode="RGB")
        im = im.resize((size[1], size[0]), Image.BILINEAR)
        plt.imshow(im)
        plt.show()
    local_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    fig.savefig(o_dir + '/pic' + local_time + '.png', format='png', transparent=True)
    plt.close()


class DataLoader(object):
    """Class for loading data."""

    def __init__(self,
                 sketch_paths,
                 photo_paths,
                 in_size=(200, 250),
                 pad_size=(286, 286),
                 out_size=(256, 256),
                 batch_size=100,
                 is_train=True):
        self.sketch_paths = sketch_paths
        self.photo_paths = photo_paths
        self.sample_len = len(self.sketch_paths)
        self.batch_size = batch_size  # batch size
        self.num_batches = int(self.sample_len / self.batch_size)
        self.in_size = in_size
        self.pad_size = pad_size
        self.out_size = out_size
        self.is_train = is_train

    def _process(self, batch, h_offset, w_offset):
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

            # if self.is_train:
            #     h_offset = random.randint(0, self.pad_size[1] - self.out_size[1] - 1)
            #     w_offset = random.randint(0, self.pad_size[0] - self.out_size[0] - 1)
            # else:
            #     h_offset = int((self.pad_size[1] - self.out_size[1]) / 2)
            #     w_offset = int((self.pad_size[0] - self.out_size[0]) / 2)
            new_sample = new_sample[h_offset:h_offset + self.out_size[1], w_offset:w_offset + self.out_size[0]]
            new_batch.append(new_sample)

        return new_batch

    def _get_batch_from_indices(self, indices, paired):
        """Given a list of indices, return the potentially augmented batch."""
        batch_sketch_paths = list()
        batch_negative_paths = list()
        batch_photo_paths = list()
        for idx in indices:
            batch_sketch_paths.append(self.sketch_paths[idx])
            while True:
                n_idx = random.randint(0, self.sample_len - 1)
                if n_idx != idx:
                    break
            batch_negative_paths.append(self.sketch_paths[n_idx])
            if paired:
                photo_idx = idx
            else:
                while True:
                    photo_idx = random.randint(0, self.sample_len - 1)
                    if photo_idx != idx:
                        break
            batch_photo_paths.append(self.photo_paths[photo_idx])

        batch_sketch_im = self.load_images(batch_sketch_paths, 'L')
        batch_negative_im = self.load_images(batch_negative_paths, 'L')
        batch_photo_im = self.load_images(batch_photo_paths, "RGB")
        if self.is_train:
            h_offset = random.randint(0, self.pad_size[1] - self.out_size[1] - 1)
            w_offset = random.randint(0, self.pad_size[0] - self.out_size[0] - 1)
        else:
            h_offset = int((self.pad_size[1] - self.out_size[1]) / 2)
            w_offset = int((self.pad_size[0] - self.out_size[0]) / 2)
        batch_sketch = self._process(batch_sketch_im, h_offset, w_offset)
        batch_negative = self._process(batch_negative_im, h_offset, w_offset)
        batch_photo = self._process(batch_photo_im, h_offset, w_offset)
        return batch_sketch, batch_sketch_paths, batch_negative, batch_negative_paths, \
               batch_photo, batch_photo_paths

    def random_batch(self, paired=True):
        """Return a randomised portion of the training data."""
        idx = np.random.permutation(range(0, len(self.sketch_paths)))[0:self.batch_size]
        return self._get_batch_from_indices(idx, paired=paired)

    def get_batch(self, idx, paired=True):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        indices = range(start_idx, start_idx + self.batch_size)
        return self._get_batch_from_indices(indices, paired=paired)

    def load_images(self, photo_paths, mode):
        assert len(photo_paths) == self.batch_size
        assert mode in ["RGB", "L"]
        img_batch = []
        for img_idx in range(len(photo_paths)):
            image_path = photo_paths[img_idx]
            image = Image.open(image_path).convert(mode=mode)
            img_batch.append(image)

        return img_batch
