"""Anycost editing module."""

import logging
import os
import typing as t
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from ..constants import SUPPORTED_AUS
from ..semanticdiffusion import SemanticDiffusion
from ..stateclass import Q
from ..utils import image_grid
from .predictor import AUPredictor

logger = logging.getLogger("rsp")


class AnycostPredictor:
    """Code is adopted from: AnyCostGAN (https://github.com/mit-han-lab/anycost-gan)."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        estimator: t.Optional[AUPredictor] = None,
    ):
        """Initialize the AnycostPredictor class.

        Args:
            device (str): Device to use for computation.
            estimator (AUPredictor): Attribute predictor.
        """
        self.device = device
        if estimator is None:
            self.estimator = AUPredictor(device=str(device))
        else:
            self.estimator = estimator

    def get_attr(self, img: torch.Tensor):
        """Get attribute scores for the generated image.

        Args:
            img: Input image tensor.

        Returns:
            torch.Tensor: Attribute scores for the generated image.
        """
        attr_preds = self.estimator.model.detect(img, data_type="tensor")

        # Extract AU predictions
        au_scores = attr_preds.aus
        tensor_au_scores = torch.tensor(au_scores.values.T, dtype=torch.float32)
        print(tensor_au_scores.shape)

        return tensor_au_scores


class AnycostDirections:
    """A class for generating attribute directions."""

    def __init__(
        self,
        sd: SemanticDiffusion,
        out_folder: Path = Path("results/anycost/"),
        etas=None,
        num_examples: int = 100,
        idx_size: int = 10000,
    ):
        """Initialize the AnycostDirections class."""
        self.out_folder = out_folder
        if not self.out_folder.exists():
            self.out_folder.mkdir(parents=True)

        self.num_examples = num_examples
        self.sd = sd
        self.device = sd.device
        self.attr_list = SUPPORTED_AUS

        self.attr_idx = {a: i for i, a in enumerate(self.attr_list)}
        self.idx_size = idx_size

        self.test_labels = NotImplemented
        self.conf_labels = NotImplemented
        self.ap = AnycostPredictor(device=self.device)
        self.etas = etas
        self.idx_path = Path(
            self.out_folder,
            f"{self.sd.model_label}-etas{self.etas}-idxsize{self.idx_size}.safetensors",
        )
        self.attrs = self.get_attrs(etas=self.etas)
        self.ns = {}

    def compute_test_directions(self):
        """Compute the directions for the test labels."""
        for label in self.test_labels:
            self.calc_direction(label)

    def compute_conf_directions(self):
        """Compute the directions for the confident?? labels."""
        for label in self.conf_labels:
            self.calc_direction(label)

    def get_attrs(self, force_rerun=False, etas=None):
        if os.path.exists(self.idx_path) and not force_rerun:
            logger.info("Anycost attributes index loaded from", self.idx_path)
            return load_file(self.idx_path)["attr"]

        logger.info("Calculating attribute index to", self.idx_path)
        results = {"attr": torch.zeros((self.idx_size, len(self.attr_list), 1))}

        for i in tqdm(range(self.idx_size)):
            q = self.sd.sample(seed=i, etas=etas)

            results["attr"][i] = self.ap.get_attr(q.x0.float())

        save_file(results, self.idx_path)
        logger.info("Anycost attributes index saved to", self.idx_path)
        return results["attr"]

    def get_attr_values(self, label):
        attr_idx = self.attr_list.index(label)
        value = self.attrs[:, attr_idx]
        return value

    def get_idx_for_attr(self, label):
        attr_values = self.get_attr_values(label)
        sort_idx = torch.argsort(attr_values)
        neg_idx = sort_idx[: self.num_examples]
        pos_idx = sort_idx[-self.num_examples :]
        return pos_idx, neg_idx

    def _calc_direction(self, label):
        logger.info("[INFO] Calculating", label)

        pos_idx, neg_idx = self.get_idx_for_attr(label)
        num_samples = len(pos_idx)
        q = self.sd.sample(etas=self.etas)
        n = Q()
        if self.sd.vqvae is not None:
            n.wT = torch.zeros_like(q.wT)
        else:
            n.xT = torch.zeros_like(q.xT)
        n.x0 = torch.zeros_like(q.x0)
        n.hs = torch.zeros_like(q.hs)
        n.delta_hs = torch.zeros_like(q.hs)
        if self.etas is not None:
            n.zs = torch.zeros_like(q.zs)

        for seed_pos, seed_neg in tqdm(zip(pos_idx, neg_idx)):
            q_pos = self.sd.sample(seed=seed_pos, etas=self.etas)
            q_neg = self.sd.sample(seed=seed_neg, etas=self.etas)
            if self.sd.vqvae is not None:
                n.wT += (q_pos.wT - q_neg.wT) / num_samples
            else:
                n.xT += (q_pos.xT - q_neg.xT) / num_samples
            if self.etas is not None:
                n.zs += (q_pos.zs - q_neg.zs) / num_samples

            n.x0 += (q_pos.x0 - q_neg.x0) / num_samples
            n.delta_hs += (q_pos.hs - q_neg.hs) / num_samples
        return n

    def calc_direction(self, label: str, force_rerun: bool = False):
        dir_path = self.idx_path.with_suffix(
            f"-label{label}-numexamples{self.num_examples}.safetensors"
        )

        if label in self.ns.keys() and not force_rerun:
            return self.ns[label]
        elif os.path.exists(dir_path) and not force_rerun:
            logger.info(f"Loading {dir_path}")
            n = Q().from_state_dict(load_file(dir_path))
            self.ns[label] = n
            return n

        n = self._calc_direction(label)

        self.ns[label] = n
        save_file(n.to_state_dict(), dir_path)
        logger.info(f"Saved to {dir_path}")
        return n

    def plot_test_directions(self, q):
        imgs = [
            self.sd.interpolate_direction(
                q, self.calc_direction(label), space="hspace", t1=-1, t2=1, numsteps=5
            )
            for label in self.test_labels
        ]
        return image_grid(imgs, titles=self.test_labels, rows=len(imgs), cols=1)

    def get_direction(self, label, clabels=None):
        """Get conditional direction."""
        n = self.calc_direction(label).delta_hs.clone()
        n_perp = n.clone()
        if not clabels is None:
            for clabel in clabels:
                nc = self.calc_direction(clabel).delta_hs.clone()
                n = n - (n * nc).sum() / (nc * nc).sum() * nc

            for clabel in clabels:
                nc = self.calc_direction(clabel).delta_hs.clone()
        return Q(delta_hs=n)

    def get_cond_dir(self, label, clabels):
        """Following https://github.com/genforce/interfacegan/blob/8da3fc0fe2a1d4c88dc5f9bee65e8077093ad2bb/utils/manipulator.py#L190."""
        primal = self.get_direction(label).delta_hs
        if clabels is None or clabels == []:
            return Q(delta_hs=primal)

        primal_shape = primal.shape
        primal = primal.flatten()

        N = torch.cat(
            [self.get_direction(l).delta_hs.flatten().unsqueeze(0) for l in clabels]
        )
        A = N @ N.T
        B = N @ primal

        x = torch.linalg.solve(A, B)

        new = primal - x @ N
        # new = primal - N.T @  torch.linalg.inv(A) @ B (fails on machines with low memory)
        new = new.reshape(primal_shape)

        return Q(delta_hs=new)
