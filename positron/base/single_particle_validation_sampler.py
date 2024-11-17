#!/usr/bin/env python3

"""
"""
import math

import torch


class SingleParticleValidationSampler:
    def __init__(
            self,
            num_samples,
            valid_fraction,
            valid_batch_size,
            train_batch_size
    ) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, "
                             f"but got num_samples={num_samples}")

        if not isinstance(valid_fraction, float) or isinstance(valid_fraction, bool) or \
                valid_fraction < 0 or valid_fraction >= 1:
            raise ValueError(f"valid_samples should be a float in [0,1), "
                             f"but got valid_fraction={valid_fraction}")

        if not isinstance(train_batch_size, int) or isinstance(train_batch_size, bool) or \
                train_batch_size <= 0:
            raise ValueError(f"train_batch_size should be a positive integer value, "
                             f"but got train_batch_size={train_batch_size}")

        if not isinstance(valid_batch_size, int) or isinstance(valid_batch_size, bool) or \
                valid_batch_size < 0.:
            raise ValueError(f"valid_batch_size should be a positive integer value, "
                             f"but got valid_batch_size={valid_batch_size}")

        if valid_batch_size == 0 or valid_fraction == 0:
            valid_batch_size = 0
            valid_fraction = 0

        self.valid_fraction = valid_fraction
        valid_num_samples = int(math.ceil(valid_fraction * num_samples))

        self.num_samples = num_samples
        self.valid_num_samples = valid_num_samples
        self.train_num_samples = num_samples - valid_num_samples

        all_indices = torch.randperm(self.num_samples)
        self.train_indices = all_indices[valid_num_samples:]
        self.valid_indices = all_indices[:valid_num_samples]

        self.train_mask = torch.ones(self.num_samples, dtype=bool)
        self.train_mask[self.valid_indices] = False

        self.train_batch_size = max(1, train_batch_size)
        self.valid_batch_size = max(1, valid_batch_size)

        self.train_mode = True

        self.this_batch_size = -1

    def __iter__(self):
        if self.train_mode:
            self.train_indices = self.train_indices[torch.randperm(len(self.train_indices))]
            self.valid_indices = self.valid_indices[torch.randperm(len(self.valid_indices))]
            j = 0  # Current count taken from the validation indices

            for batch_idx in range(self.__len__()):
                if j + self.valid_batch_size < self.valid_num_samples:
                    valid = self.valid_indices[j:j + self.valid_batch_size]
                    j += self.valid_batch_size
                else:
                    valid = torch.cat([
                        self.valid_indices[j:],
                        self.valid_indices[:j + self.valid_batch_size - self.valid_num_samples],
                    ], 0)
                    j = self.valid_batch_size - self.valid_num_samples - 1

                i = batch_idx * self.train_batch_size  # Current count taken from the training indices
                train = self.train_indices[i:min(i + self.train_batch_size, self.train_num_samples)]

                batch = torch.cat([valid, train], 0)
                yield batch
        else:
            indices = torch.randperm(self.num_samples)
            for batch_idx in range(self.__len__()):
                i = batch_idx * self.train_batch_size
                batch = indices[i: min(i + self.train_batch_size, self.num_samples)]
                yield batch

    def __len__(self):
        if self.train_mode:
            return math.ceil(self.train_num_samples / float(self.train_batch_size))
        else:
            return math.ceil(self.num_samples / float(self.train_batch_size))

    def get_current_train_mask(self, indices):
        return self.train_mask[indices]

    def eval(self):
        self.train_mode = False

    def train(self):
        self.train_mode = True

    def __str__(self):
        return (f"Number of images (train): {self.train_num_samples}\n"
                f"Number of images (valid): {self.valid_num_samples}")

