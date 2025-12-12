"""Anycost editing module."""

import json
import logging
import typing as t
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from ..constants import RESULTS_DIR, SUPPORTED_AUS
from ..log_utils import log
from ..semanticdiffusion import SemanticDiffusion
from ..stateclass import Q
from ..utils import image_grid
from .predictor import AUPredictor


class AnycostPredictor:
    """Code is adopted from: AnyCostGAN (https://github.com/mit-han-lab/anycost-gan)."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        estimator: t.Optional[AUPredictor] = None,
        **kwargs,
    ):
        """Initialize the AnycostPredictor class.

        Args:
            device (str): Device to use for computation.
            estimator (AUPredictor): Attribute predictor.
            **kwargs: Additional keyword arguments, passed to the estimator.
        """
        self.device = device
        if estimator is None:
            self.estimator = AUPredictor(device=str(device), **kwargs)
        else:
            self.estimator = estimator

    def __call__(self, imgs: torch.Tensor, batch_size: int | None = None):
        """Get attribute scores for the generated image or images.

        Args:
            imgs:
                Input image tensors.
            batch_size:
                Batch size for prediction.

        Returns:
            torch.Tensor: Attribute scores for all of the input images.
        """
        # Unnormalize the images from [-1, 1] to [0, 255] for the predictor
        # by first mapping to [0, 2], then to [0, 255] and clamping
        # finally move to CPU since predictor will move it to GPU itself.
        imgs = imgs.add(1).mul(127.5).clamp(0, 255).cpu()

        attr_preds = self.estimator.predict(images=imgs, batch_size=batch_size)

        return attr_preds


class AnycostDirections:
    """A class for generating attribute directions."""

    def __init__(
        self,
        sd: SemanticDiffusion,
        out_folder: Path = Path(RESULTS_DIR, "anycost"),
        etas=None,
        batch_size: int = 1,
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
        self.ap = AnycostPredictor(device=self.device, progress_bar=False)
        self.etas = etas
        self.idx_path = Path(
            self.out_folder,
            "attributes.safetensors",
        )
        self.batch_size = batch_size
        attr_results = self.get_attrs(etas=self.etas)
        self.attrs = attr_results["attr"]
        self.attr_idxs = attr_results["idx"]
        self.ns = {}
        self.cond_dirs = {}
        self.config = self._get_config()
        self.force_overwrite = False

    def _get_config(self):
        config = {
            "model": self.sd.model_id,
            "timesteps": self.sd.num_inference_steps,
            "h_space": self.sd.h_space,
            "etas": self.etas,
            "idx_size": self.idx_size,
            "num_examples": self.num_examples,
        }
        config_path = self.out_folder / "config.json"
        if config_path.exists():
            with config_path.open("r") as config_file:
                existing_config = json.load(config_file)

            # check if they differ
            diff_str = ""
            for key, new in config.items():
                old = existing_config.get(key)
                if old != new:
                    diff_str += f"key {key}: old: {old} -> new: {new}\n"

            if diff_str and not self.force_overwrite:
                raise ValueError(
                    f"Config differs with existing config:\n{diff_str}. "
                    "If you want to overwrite, set force_overwrite=True"
                )
            elif self.force_overwrite:
                with config_path.open("w") as config_file:
                    json.dump(config, config_file)
            else:
                return existing_config
        else:
            with config_path.open("w") as config_file:
                json.dump(config, config_file)

    def compute_test_directions(self):
        """Compute the directions for the test labels."""
        for label in self.test_labels:
            self.calc_direction(label)

    def compute_conf_directions(self):
        """Compute the directions for the confident?? labels."""
        for label in self.conf_labels:
            self.calc_direction(label)

    def get_attrs(self, force_rerun=False, etas=None) -> dict[str, torch.Tensor]:
        """Get the attributes for the labels.

        Args:
            force_rerun: Whether to force the recalculation of the attributes.
            etas (optional): The eta values to use for the calculation.

        Returns:
            dict: The attributes for the labels.
        """
        if self.idx_path.exists() and not force_rerun:
            log(
                f"Anycost attributes index loaded from {self.idx_path}",
                level=logging.INFO,
            )
            return load_file(self.idx_path)

        log(f"Calculating attribute index to {self.idx_path}", level=logging.INFO)
        results = {
            "attr": torch.zeros((self.idx_size, len(self.attr_list))),
        }

        if self.batch_size > 1:
            for i in tqdm(range(0, self.idx_size, self.batch_size)):
                # Sample a batch of images
                batch_end = min(i + self.batch_size, self.idx_size)
                seeds = list(range(i, batch_end))
                qs = [self.sd.sample(seed=seed, etas=etas) for seed in seeds]
                # Concatenate the images from the batch and get their attributes
                imgs = torch.cat([q.x0.float() for q in qs], dim=0)
                batch_attrs = self.ap(imgs, batch_size=self.batch_size)
                results["attr"][i:batch_end] = batch_attrs
        else:
            for i in tqdm(range(self.idx_size)):
                q = self.sd.sample(seed=i, etas=etas)

                results["attr"][i : i + 1] = self.ap(q.x0.float(), batch_size=1)

        # get valid indices, where the results are not -1
        valid_indices = torch.where((results["attr"] != -1).all(dim=1))[0]
        results["idx"] = valid_indices

        num_invalid = self.idx_size - valid_indices.numel()
        if num_invalid > 0:
            log(
                f"Skipped {num_invalid} images where AU detection failed!",
                level=logging.WARNING,
            )

        save_file(results, self.idx_path)
        log(f"Anycost attributes index saved to {self.idx_path}", level=logging.INFO)
        return results

    def get_attr_values(self, label):
        """Get the attribute values for a given label.

        Args:
            label: The label of the attribute.

        Returns:
            torch.Tensor: The attribute values for the given label.
        """
        attr_idx = self.attr_list.index(label)
        value = self.attrs[:, attr_idx]
        return value

    def get_idx_for_attr(self, label):
        """Get the indices for a given attribute.

        Args:
            label: The label of the attribute.

        Returns:
            tuple: The positive and negative indices for the given attribute.
        """
        attr_values = self.get_attr_values(label)
        # handle -1 by using the self.atrr_idx which contains only the valid indices
        valid_attr_values = attr_values[self.attr_idxs]
        # sort the valid attribute values
        sort_idx = torch.argsort(valid_attr_values)
        # map back to original indices
        neg_idx = self.attr_idxs[sort_idx[: self.num_examples]]
        pos_idx = self.attr_idxs[sort_idx[-self.num_examples :]]
        return pos_idx, neg_idx

    def _calc_direction(self, label):
        log(f"Calculating {label}", level=logging.INFO)

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

        convergence_test = True  # Setting this here to avoid saving
        # as default, but can be used if you
        # want to track convergence

        if convergence_test:
            convergence_dict = {
                # num_samples x each is n.hs size
                "steps_delta_hs": torch.zeros(num_samples, *n.hs.shape)
            }  # Unnused if not changed in for loop

        for step_i, (seed_pos, seed_neg) in enumerate(
            tqdm(zip(pos_idx, neg_idx), total=self.num_examples)
        ):
            q_pos = self.sd.sample(seed=seed_pos, etas=self.etas)
            q_neg = self.sd.sample(seed=seed_neg, etas=self.etas)
            if self.sd.vqvae is not None:
                n.wT += (q_pos.wT - q_neg.wT) / num_samples
            else:
                n.xT += (q_pos.xT - q_neg.xT) / num_samples
            if self.etas is not None:
                n.zs += (q_pos.zs - q_neg.zs) / num_samples

            n.x0 += (q_pos.x0 - q_neg.x0) / num_samples
            n.delta_hs += q_pos.hs - q_neg.hs

            if convergence_test:
                convergence_dict["steps_delta_hs"][step_i] = (  # pyright: ignore [reportPossiblyUnboundVariable]
                    n.delta_hs.detach().cpu().clone() / step_i
                )

        n.delta_hs /= num_samples

        if convergence_test:
            convergence_path = self.out_folder / "convergence" / f"{label}.safetensors"
            if not convergence_path.parent.exists():
                convergence_path.parent.mkdir(parents=True)
            save_file(convergence_dict, convergence_path)
            log(
                f"Saved convergence dict {label} to {convergence_path}",
                level=logging.INFO,
            )

        return n

    def calc_direction(self, label: str, force_rerun: bool = False) -> Q:
        """Calculate the direction for a given label.

        Args:
            label: The label of the direction.
            force_rerun: Whether to force a re-run of the calculation.

        Returns:
            Q: The calculated direction.
        """
        dir_path = self.out_folder / "directions" / f"{label}.safetensors"
        if not dir_path.parent.exists():
            dir_path.parent.mkdir(parents=True)

        if label in self.ns.keys() and not force_rerun:
            return self.ns[label]
        elif dir_path.exists() and not force_rerun:
            log(f"Loading {dir_path}", level=logging.INFO)
            n = Q().from_state_dict(load_file(dir_path))
            self.ns[label] = n
            return n

        n = self._calc_direction(label)

        self.ns[label] = n
        save_file(n.to_state_dict(), dir_path)
        log(f"Saved to {dir_path}", level=logging.INFO)
        return n

    def plot_test_directions(self, q: Q) -> Image.Image:
        """Plot test directions."""
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
        if clabels is not None:
            for clabel in clabels:
                nc = self.calc_direction(clabel).delta_hs.clone()
                n = n - (n * nc).sum() / (nc * nc).sum() * nc

        return Q(delta_hs=n)

    def get_cond_dir(self, label, clabels):
        """Following https://github.com/genforce/interfacegan/blob/8da3fc0fe2a1d4c88dc5f9bee65e8077093ad2bb/utils/manipulator.py#L190."""
        cond_path = self.out_folder / "disentangled" / f"{label}.safetensors"
        if not cond_path.parent.exists():
            cond_path.parent.mkdir(parents=True)
        if label in self.cond_dirs.keys():
            return self.cond_dirs[label]
        elif cond_path.exists() and label not in self.cond_dirs.keys():
            log(f"Loading condition from {cond_path}", logging.INFO)
            cond = Q().from_state_dict(load_file(cond_path))
            self.cond_dirs[label] = cond
            return cond

        primal = self.get_direction(label).delta_hs.to(self.device)
        if clabels is None or clabels == []:
            return Q(delta_hs=primal)

        primal_shape = primal.shape
        primal = primal.flatten()

        N = torch.cat(
            [
                self.get_direction(label)
                .delta_hs.to(self.device)
                .flatten()
                .unsqueeze(0)
                for label in clabels
            ]
        )
        A = N @ N.T
        B = N @ primal

        x = torch.linalg.solve(A, B)

        new = primal - x @ N
        # new = primal - N.T @  torch.linalg.inv(A) @ B (fails on machines with low mem)
        new = new.reshape(primal_shape)

        new_Q = Q(delta_hs=new)
        save_file(new_Q.to_state_dict(), cond_path)

        return new_Q
