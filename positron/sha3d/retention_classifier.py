#!/usr/bin/env python3
from typing import Dict

import numpy as np
import torch
from voxelium import ModelContainer


class RetentionClassifier(torch.nn.Module):
    def __init__(
            self,
            logit_size:int = None,
            classifier_description: dict = None
    ):
        super().__init__()

        self.classifier_description = {
            'type': 'ResidMLP',
            'input_dim': logit_size,
            'resid_dim': 8,
            'resid_count': 3,
            'output_dim': 1,
            'activation': 'elu',
            'normalize': 'batch'
        } if classifier_description is None else classifier_description
        self.classifier = ModelContainer(self.classifier_description)
        self.estimator = ModelContainer(self.classifier_description)

        params = [
            {
                "params": self.classifier.parameters(),
                "lr": 1e-2,
                "weight_decay": 1e-2
            },
            {
                "params": self.estimator.parameters(),
                "lr": 1e-2,
                "weight_decay": 1e-2
            }
        ]

        self.opt = torch.optim.AdamW(params)

        self.last_accuracy_ratio = 1.

        self.summary = None

    def forward(self, logit, labels, make_summary=True):
        self.opt.zero_grad()
        self.train()

        classification_logit = self.classifier(logit.detach()).squeeze(1)
        estimator_logit = self.estimator(logit.detach()).squeeze(1)

        # Calculate this batch's accuracy
        batch_accuracy_train = (classification_logit[labels] > 0).float()
        batch_accuracy_valid = (classification_logit[~labels] < 0).float()

        batch_accuracy = torch.zeros_like(classification_logit)
        batch_accuracy[labels] = batch_accuracy_train
        batch_accuracy[~labels] = batch_accuracy_valid

        pos_count = torch.count_nonzero(labels)
        pos_weight = (labels.shape[-1] - pos_count) / (pos_count + 1e-12)
        pos_weight *= self.last_accuracy_ratio

        accuracy_ratio = batch_accuracy_valid.mean() / (batch_accuracy_train.mean() + 1e-12)
        self.last_accuracy_ratio = np.clip(float(accuracy_ratio), 1./1.2, 1.2)

        if pos_weight <= 0:
            raise RuntimeError(f"Bad examples counts (pos_weight={pos_weight})")

        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            classification_logit, labels.float(), pos_weight=pos_weight)

        estimator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            estimator_logit, batch_accuracy)

        loss = classification_loss + estimator_loss
        loss.backward()
        self.opt.step()

        if make_summary:
            self.summary = {
                'batch_accuracy': float(batch_accuracy.mean()),
                'batch_accuracy_train': float(batch_accuracy_train.mean()),
                'batch_accuracy_valid': float(batch_accuracy_valid.mean()),
                'accuracy_ratio': float(self.last_accuracy_ratio),
                'batch retention': float(batch_accuracy.mean() * 2 - 1),
                'estimated retention': float(estimator_logit.sigmoid().mean() * 2 - 1)
            }

    def get_retention(self, logit):
        return self.estimator(logit).sigmoid() * 2 - 1

    def get_summary(self, title="Retention"):
        summary = {}
        for key in self.summary:
            summary[f"{title}/{key}"] = self.summary[key]
        return summary

    def get_state_dict(self) -> Dict:
        return {
            "type": "RetentionClassifier",
            "version": "0.0.1",

            "classifier_description": self.classifier_description,
            "classifier": self.classifier.state_dict(),
            "estimator": self.estimator.state_dict(),
            "opt": self.opt.state_dict(),

            "last_accuracy_ratio": self.last_accuracy_ratio,
        }

    @staticmethod
    def load_from_state_dict(state_dict, skip_optimizers=False):
        if "type" not in state_dict or state_dict["type"] != "RetentionClassifier":
            raise TypeError("Input is not an 'RetentionClassifier' instance.")

        if "version" not in state_dict:
            raise RuntimeError("RetentionClassifier instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            container = RetentionClassifier(
                classifier_description=state_dict["classifier_description"]
            )
            container.classifier.load_state_dict(state_dict["classifier"])
            container.estimator.load_state_dict(state_dict["estimator"])

            container.last_accuracy_ratio = state_dict["last_accuracy_ratio"]

            if not skip_optimizers:
                container.opt.load_state_dict(state_dict["opt"])

            return container
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
