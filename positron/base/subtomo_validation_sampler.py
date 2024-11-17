#!/usr/bin/env python3

"""
"""
import math
import torch


class SubtomoValidationSampler:
    def __init__(
            self,
            group_indices,
            valid_fraction,
            valid_batch_size,
            train_batch_size
    ) -> None:
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

        if not torch.is_tensor(group_indices):
            group_indices = torch.from_numpy(group_indices)

        unique, unique_inverse = torch.unique(group_indices, return_inverse=True)
        
        self.num_samples = len(group_indices)
        self.num_groups = len(unique)
        self.group_size = torch.zeros(self.num_groups, dtype=int)
        self.mean_group_size = 0.

        if valid_batch_size == 0 or valid_fraction == 0:
            valid_batch_size = 0
            valid_fraction = 0
        
        self.valid_fraction = valid_fraction
        valid_num_groups = int(math.ceil(valid_fraction * self.num_groups))

        self.valid_num_groups = valid_num_groups
        self.train_num_groups = self.num_groups - valid_num_groups

        all_indices = torch.randperm(self.num_groups)
        self.train_indices = all_indices[valid_num_groups:]
        self.valid_indices = all_indices[:valid_num_groups]

        self.groups = [[] for _ in range(len(unique))]
        for i in range(self.num_samples):
            self.groups[unique_inverse[i]].append(i)

        self.train_mask = torch.ones(self.num_samples, dtype=bool)

        for i in range(self.num_groups):
            self.groups[i] = torch.IntTensor(self.groups[i])
            self.group_size[i] = len(self.groups[i])
            self.mean_group_size += len(self.groups[i])
            if i in self.valid_indices:
                self.train_mask[self.groups[i]] = False

        self.mean_group_size /= self.num_groups

        self.train_batch_size = max(1, round(train_batch_size / self.mean_group_size))
        self.valid_batch_size = max(1, round(valid_batch_size / self.mean_group_size))

        self.train_mode = True

        self.this_batch_validation_size = -1
        self.this_batch_size = -1
    
    def groups_to_samples(self, indices):
        sample_indices = self.groups[indices[0]]
        for i in torch.arange(1, len(indices)):
            sample_indices = torch.cat([sample_indices, self.groups[indices[i]]], 0)
        return sample_indices

    def __iter__(self):
        if self.train_mode:
            self.train_indices = self.train_indices[torch.randperm(len(self.train_indices))]
            self.valid_indices = self.valid_indices[torch.randperm(len(self.valid_indices))]
            j = 0  # Current count taken from the validation indices

            for batch_idx in range(self.__len__()):
                if j + self.valid_batch_size < self.valid_num_groups:
                    valid = self.valid_indices[j:j + self.valid_batch_size]
                    j += self.valid_batch_size
                else:
                    valid = torch.cat([
                        self.valid_indices[j:],
                        self.valid_indices[:j + self.valid_batch_size - self.valid_num_groups],
                    ], 0)
                    j = self.valid_batch_size - self.valid_num_groups - 1

                i = batch_idx * self.train_batch_size  # Current count taken from the training indices
                train = self.train_indices[i:min(i + self.train_batch_size, self.train_num_groups)]

                self.this_batch_validation_size = torch.sum(self.group_size[valid])

                groups = torch.cat([valid, train], 0)
                samples = self.groups_to_samples(groups)

                self.this_batch_size = len(samples)

                yield samples
        else:
            indices = torch.randperm(self.num_groups)
            self.this_batch_validation_size = 0
            for batch_idx in range(self.__len__()):
                i = batch_idx * self.train_batch_size
                groups = indices[i:min(i + self.train_batch_size, self.num_groups)]
                samples = self.groups_to_samples(groups)

                self.this_batch_size = len(samples)

                yield samples

    def __len__(self):
        if self.train_mode:
            return math.ceil(self.train_num_groups / float(self.train_batch_size))
        else:
            return math.ceil(self.num_groups / float(self.train_batch_size))

    def get_current_train_mask(self, indices):
        return self.train_mask[indices]

    def eval(self):
        self.train_mode = False

    def train(self):
        self.train_mode = True

    def __str__(self):
        return (f"Number of images (train): {self.train_num_groups}\n"
                f"Number of images (valid): {self.valid_num_groups}")
