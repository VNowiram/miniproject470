# gnn_se_trainer.py
"""
GNN-based AC State Estimator — Training Pipeline
=================================================
Author  : (project)
Version : 1.0

This file is **fully separated** from the model definition (gnn_se_model.py).
It owns everything needed to train, validate, and save the GNN:

    1. DataGenerator  — physics-driven synthetic training samples
    2. GNNSEDataset   — PyTorch Dataset wrapping generated data
    3. collate_fn     — batch assembly helper for DataLoader
    4. Trainer        — training loop, LR scheduling, checkpointing

Relationship to the existing WLS pipeline
------------------------------------------
DataGenerator uses the same per-unit measurement equations as the WLS solver
(P_inject, Q_inject, V_mag, P_flow, Q_flow) to produce ground-truth labels.
No real meters are required — operating points are sampled from a physics-valid
distribution, then synthetic noisy measurements are derived from them.

Quick start
-----------
    from gnn_se_trainer import DataGenerator, Trainer
    from gnn_se_model   import GNNStateEstimator, GraphBuilder

    # 1.  Describe the network (same branch params as your System object)
    branch_params = [
        #  (branch_id, fbus_0based, tbus_0based, g_s,   b_s,   b_sh)
        (1, 0, 1,  0.5390, -3.7736, 0.0),
        (2, 0, 2,  0.5390, -3.7736, 0.0),
        (3, 1, 2,  0.5390, -3.7736, 0.0),
    ]

    # 2.  Generate training data
    gen   = DataGenerator(branch_params, slack_idx=0, n_buses=3)
    data  = gen.generate(n_samples=5000, noise_sd=0.01, seed=42)

    # 3.  Build model + train
    model = GNNStateEstimator(hidden_dim=64, n_mp_layers=4, slack_idx=0)
    trainer = Trainer(model, data, branch_params, slack_idx=0)
    trainer.train(epochs=200, batch_size=64, lr=1e-3)
    trainer.save("gnn_se_checkpoint.pt")

Dependencies
------------
    torch >= 1.13
    numpy
    (optional) tqdm  — progress bars during training
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from gnn_se_model import (
    GNNStateEstimator,
    GraphBuilder,
    GNNSEInput,
    GNNSELabel,
    SELoss,
    N_BUSES,
    N_EDGES,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    _DEFAULT_EDGE_INDEX,
)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS HELPERS   (same equations as MeasurementFunc in dynamic_parameter.py)
# ══════════════════════════════════════════════════════════════════════════════

def _Pi(i: int, G: np.ndarray, B: np.ndarray, theta: np.ndarray, V: np.ndarray) -> float:
    """Active power injection: P_i = V_i Σ_j V_j (G_ij cosθ_ij + B_ij sinθ_ij)"""
    dth = theta[i] - theta
    return float(V[i] * np.dot(V, G[i] * np.cos(dth) + B[i] * np.sin(dth)))


def _Qi(i: int, G: np.ndarray, B: np.ndarray, theta: np.ndarray, V: np.ndarray) -> float:
    """Reactive power injection: Q_i = V_i Σ_j V_j (G_ij sinθ_ij − B_ij cosθ_ij)"""
    dth = theta[i] - theta
    return float(V[i] * np.dot(V, G[i] * np.sin(dth) - B[i] * np.cos(dth)))


def _Pij(fi: int, ti: int, g_s: float, b_s: float,
         theta: np.ndarray, V: np.ndarray) -> float:
    """Active power flow from bus fi to ti."""
    dth = theta[fi] - theta[ti]
    return float(V[fi] ** 2 * g_s - V[fi] * V[ti] * (g_s * np.cos(dth) + b_s * np.sin(dth)))


def _Qij(fi: int, ti: int, g_s: float, b_s: float, b_sh: float,
         theta: np.ndarray, V: np.ndarray) -> float:
    """Reactive power flow from bus fi to ti."""
    dth = theta[fi] - theta[ti]
    return float(
        -V[fi] ** 2 * (b_s + b_sh)
        - V[fi] * V[ti] * (g_s * np.sin(dth) - b_s * np.cos(dth))
    )


def _build_YBus(
    n_buses       : int,
    branch_params : List[Tuple[int, int, int, float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build G and B matrices from branch parameters."""
    Y = np.zeros((n_buses, n_buses), dtype=complex)
    for _, fi, ti, g_s, b_s, b_sh in branch_params:
        y_s  = complex(g_s, b_s)
        b_sh_c = complex(0, b_sh)
        Y[fi, fi] += y_s + b_sh_c
        Y[ti, ti] += y_s + b_sh_c
        Y[fi, ti] -= y_s
        Y[ti, fi] -= y_s
    return Y.real.copy(), Y.imag.copy()


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE   (one training pair: graph_input + label)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GNNSESample:
    """One training pair returned by DataGenerator."""
    gnn_input : GNNSEInput
    label     : GNNSELabel


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class DataGenerator:
    """
    Physics-driven synthetic data generator for GNN State Estimation training.

    Strategy
    ---------
    1. Sample a random operating state  (θ, V)  within realistic p.u. ranges.
    2. Compute all possible measurements h(x) from that true state using the
       same AC power-flow equations as the WLS solver.
    3. Add Gaussian noise with standard deviation `noise_sd` to every
       measurement value.
    4. Randomly drop measurements according to `drop_prob` to simulate
       partial observability (at least 1 per node is kept to remain
       observable).
    5. Pack the noisy, partially-available measurements into a GNNSEInput
       and the true  (θ, V) into a GNNSELabel.

    The generated dataset is physics-consistent by construction.  No
    power-flow solver is needed because we start from the state, not the
    injections.

    Parameters
    ----------
    branch_params : list of (branch_id, fbus_0, tbus_0, g_s, b_s, b_sh)
    slack_idx     : 0-based reference bus index
    n_buses       : number of buses
    theta_range   : (min, max) angle range in degrees for non-slack buses
    V_range       : (min, max) voltage magnitude range in p.u.
    """

    def __init__(
        self,
        branch_params : List[Tuple[int, int, int, float, float, float]],
        slack_idx     : int   = 0,
        n_buses       : int   = 3,
        theta_range   : Tuple[float, float] = (-15.0, 15.0),   # degrees
        V_range       : Tuple[float, float] = (0.90,  1.10),   # p.u.
    ) -> None:
        self.branch_params = branch_params
        self.slack_idx     = slack_idx
        self.n_buses       = n_buses
        self.theta_range   = theta_range
        self.V_range       = V_range
        self.n_lines       = len(branch_params)

        self.G, self.B = _build_YBus(n_buses, branch_params)
        self.builder   = GraphBuilder(n_buses, branch_params, slack_idx)

    def generate(
        self,
        n_samples   : int   = 5000,
        noise_sd    : float = 0.01,    # std of Gaussian measurement noise (p.u.)
        drop_prob   : float = 0.10,    # probability of randomly dropping a measurement
        include_flows : bool = True,   # whether to include P/Q flow measurements
        seed        : Optional[int]  = None,
    ) -> List[GNNSESample]:
        """
        Generate `n_samples` labelled training pairs.

        Parameters
        ----------
        n_samples     : number of samples to generate
        noise_sd      : standard deviation of additive Gaussian noise (p.u.)
        drop_prob     : probability of dropping any single measurement
        include_flows : if True, also include P/Q flow measurements on edges
        seed          : random seed for reproducibility

        Returns
        -------
        List[GNNSESample]   length n_samples
        """
        rng     = np.random.default_rng(seed)
        samples : List[GNNSESample] = []

        th_lo, th_hi = np.radians(self.theta_range[0]), np.radians(self.theta_range[1])
        v_lo,  v_hi  = self.V_range

        for _ in range(n_samples):
            # ── Sample true state ─────────────────────────────────────────
            theta             = rng.uniform(th_lo, th_hi, size=self.n_buses)
            theta[self.slack_idx] = 0.0           # slack angle fixed
            V                 = rng.uniform(v_lo, v_hi, size=self.n_buses)

            # ── Compute true measurements ─────────────────────────────────
            node_feat = np.zeros((self.n_buses, NODE_FEAT_DIM), dtype=np.float32)
            edge_feat = np.zeros((2 * self.n_lines, EDGE_FEAT_DIM), dtype=np.float32)

            # Fill fixed physics on edges
            for k, (_, fi, ti, g_s, b_s, b_sh) in enumerate(self.branch_params):
                edge_feat[2*k,     4:7] = [g_s, b_s, b_sh]
                edge_feat[2*k + 1, 4:7] = [g_s, b_s, b_sh]

            # Bus measurements (P_inject, Q_inject, V_mag)
            for i in range(self.n_buses):
                p_true = _Pi(i, self.G, self.B, theta, V) + rng.normal(0, noise_sd)
                q_true = _Qi(i, self.G, self.B, theta, V) + rng.normal(0, noise_sd)
                v_true = V[i] + rng.normal(0, noise_sd)

                # Random drop (but always keep at least V_mag for each bus)
                keep_p = rng.random() >= drop_prob
                keep_q = rng.random() >= drop_prob
                keep_v = True   # always keep voltage magnitude

                if keep_p:
                    node_feat[i, 0] = p_true
                    node_feat[i, 3] = 1.0   # mask_P
                if keep_q:
                    node_feat[i, 1] = q_true
                    node_feat[i, 4] = 1.0   # mask_Q
                if keep_v:
                    node_feat[i, 2] = max(v_true, 0.01)   # clamp for safety
                    node_feat[i, 5] = 1.0   # mask_V

            # Branch flow measurements (P_flow, Q_flow) — optional
            if include_flows:
                for k, (_, fi, ti, g_s, b_s, b_sh) in enumerate(self.branch_params):
                    pf_true = _Pij(fi, ti, g_s, b_s, theta, V) + rng.normal(0, noise_sd)
                    qf_true = _Qij(fi, ti, g_s, b_s, b_sh, theta, V) + rng.normal(0, noise_sd)

                    keep_pf = rng.random() >= drop_prob
                    keep_qf = rng.random() >= drop_prob

                    if keep_pf:
                        edge_feat[2*k, 0] = pf_true
                        edge_feat[2*k, 2] = 1.0   # mask_Pf
                    if keep_qf:
                        edge_feat[2*k, 1] = qf_true
                        edge_feat[2*k, 3] = 1.0   # mask_Qf

            gnn_input = GNNSEInput(
                node_features = node_feat,
                edge_features = edge_feat,
                edge_index    = self.builder.edge_index.copy(),
                slack_idx     = self.slack_idx,
            )
            label = GNNSELabel(
                V_pu      = V.astype(np.float32),
                theta_rad = theta.astype(np.float32),
            )
            samples.append(GNNSESample(gnn_input=gnn_input, label=label))

        return samples

    def from_ybus(self, ybus) -> "DataGenerator":
        """
        Alternate constructor — rebuild from a YBusMatrix object so you can
        reuse the exact same network parameters as your System.

        Usage:
            gen = DataGenerator.__new__(DataGenerator)
            gen = DataGenerator.from_ybus(grid.ybus, slack_idx=0)
        """
        params = []
        for br in ybus.branches:
            g_s, b_s, b_sh = ybus.branch_params(br.id)
            fi = ybus.bus_index(br.fbus)
            ti = ybus.bus_index(br.tbus)
            params.append((br.id, fi, ti, g_s, b_s, b_sh))

        return DataGenerator(
            branch_params = params,
            slack_idx     = self.slack_idx,
            n_buses       = ybus.n,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class GNNSEDataset(Dataset):
    """
    PyTorch Dataset wrapping a List[GNNSESample].

    Each item returned is:
        node_feat : FloatTensor (n_buses, NODE_FEAT_DIM)
        edge_feat : FloatTensor (n_edges, EDGE_FEAT_DIM)
        x_true    : FloatTensor (n_buses, 2)  — [[θ_i, V_i], ...]

    edge_index is stored as a class-level constant (fixed topology).
    """

    edge_index : torch.Tensor = _DEFAULT_EDGE_INDEX

    def __init__(self, samples: List[GNNSESample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        node_feat = torch.tensor(s.gnn_input.node_features, dtype=torch.float32)
        edge_feat = torch.tensor(s.gnn_input.edge_features, dtype=torch.float32)

        # Label: stack [θ, V] → (n_buses, 2)
        theta = torch.tensor(s.label.theta_rad, dtype=torch.float32)
        V_pu  = torch.tensor(s.label.V_pu,      dtype=torch.float32)
        x_true = torch.stack([theta, V_pu], dim=-1)   # (n_buses, 2)

        return node_feat, edge_feat, x_true


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stacks a list of (node_feat, edge_feat, x_true) into batched tensors.

    Returns
    -------
    node_feat  : (B, n_buses, NODE_FEAT_DIM)
    edge_feat  : (B, n_edges, EDGE_FEAT_DIM)
    edge_index : (2, n_edges)  — shared, not batched
    x_true     : (B, n_buses, 2)
    """
    nf_list, ef_list, xt_list = zip(*batch)
    return (
        torch.stack(nf_list),
        torch.stack(ef_list),
        GNNSEDataset.edge_index,
        torch.stack(xt_list),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainHistory:
    """Logged metrics per epoch."""
    train_loss : List[float] = field(default_factory=list)
    val_loss   : List[float] = field(default_factory=list)
    lr         : List[float] = field(default_factory=list)


class Trainer:
    """
    Training loop for GNNStateEstimator.

    Features
    --------
    - Train / validation split (configurable ratio)
    - ReduceLROnPlateau learning-rate scheduling
    - Early stopping based on validation loss patience
    - Automatic device selection (CUDA if available, else CPU)
    - Checkpoint saving / loading

    Parameters
    ----------
    model         : GNNStateEstimator
    samples       : List[GNNSESample] from DataGenerator.generate()
    branch_params : list of (branch_id, fi, ti, g_s, b_s, b_sh) — for reference
    slack_idx     : 0-based slack bus index
    lambda_theta  : weight of angle loss in SELoss
    lambda_V      : weight of voltage magnitude loss in SELoss
    val_ratio     : fraction of samples used for validation (default 0.15)
    device        : torch.device  (auto-selected if None)
    """

    def __init__(
        self,
        model         : GNNStateEstimator,
        samples       : List[GNNSESample],
        branch_params : List[Tuple[int, int, int, float, float, float]],
        slack_idx     : int   = 0,
        lambda_theta  : float = 1.0,
        lambda_V      : float = 1.0,
        val_ratio     : float = 0.15,
        device        : Optional[torch.device] = None,
    ) -> None:
        self.model         = model
        self.branch_params = branch_params
        self.slack_idx     = slack_idx
        self.val_ratio     = val_ratio
        self.history       = TrainHistory()

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        # Loss
        self.criterion = SELoss(lambda_theta=lambda_theta, lambda_V=lambda_V,
                                slack_idx=slack_idx)

        # Datasets
        dataset = GNNSEDataset(samples)
        n_val   = max(1, int(len(dataset) * val_ratio))
        n_train = len(dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(0)
        )

        print(f"  Trainer ready  |  device={self.device}  "
              f"|  train={n_train}  val={n_val}  "
              f"|  params={model.n_parameters():,}")

    # ── Main training entry point ────────────────────────────────────────────

    def train(
        self,
        epochs      : int   = 200,
        batch_size  : int   = 64,
        lr          : float = 1e-3,
        patience    : int   = 20,
        lr_factor   : float = 0.5,
        lr_patience : int   = 10,
        verbose     : bool  = True,
    ) -> TrainHistory:
        """
        Run the full training loop.

        Parameters
        ----------
        epochs      : maximum number of training epochs
        batch_size  : mini-batch size
        lr          : initial learning rate for Adam
        patience    : early-stopping patience (epochs without val improvement)
        lr_factor   : factor by which LR is reduced on plateau
        lr_patience : epochs without improvement before reducing LR
        verbose     : print epoch summaries

        Returns
        -------
        TrainHistory — dict of lists: train_loss, val_loss, lr
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode      = "min",
            factor    = lr_factor,
            patience  = lr_patience,
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size  = batch_size,
            shuffle     = True,
            collate_fn  = collate_fn,
            num_workers = 0,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size  = batch_size,
            shuffle     = False,
            collate_fn  = collate_fn,
            num_workers = 0,
        )

        edge_index = GNNSEDataset.edge_index.to(self.device)
        best_val   = float("inf")
        no_improve = 0
        t0         = time.time()

        if verbose:
            self._print_header()

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, edge_index)
            val_loss   = self._val_epoch(val_loader, edge_index)
            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(val_loss)
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.lr.append(current_lr)

            # Early stopping
            if val_loss < best_val - 1e-8:
                best_val   = val_loss
                no_improve = 0
                self._save_best()
            else:
                no_improve += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                elapsed = time.time() - t0
                print(f"  {epoch:>5}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
                      f"{current_lr:>10.2e}  {elapsed:>8.1f}s")

            if no_improve >= patience:
                if verbose:
                    print(f"\n  Early stop at epoch {epoch}  (best val={best_val:.6f})")
                break

        if verbose:
            print(f"\n  Training complete.  Best val loss = {best_val:.6f}")
            print(f"  Total time : {time.time() - t0:.1f} s")

        # Restore best weights
        self._load_best()
        return self.history

    # ── Epoch helpers ────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader     : DataLoader,
        optimizer  : torch.optim.Optimizer,
        edge_index : torch.Tensor,
    ) -> float:
        self.model.train()
        total, count = 0.0, 0

        for nf, ef, ei, xt in loader:
            nf, ef, xt = nf.to(self.device), ef.to(self.device), xt.to(self.device)

            optimizer.zero_grad()
            x_hat = self.model(nf, ef, edge_index)
            loss  = self.criterion(x_hat, xt)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total += loss.item() * nf.size(0)
            count += nf.size(0)

        return total / count if count else 0.0

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, edge_index: torch.Tensor) -> float:
        self.model.eval()
        total, count = 0.0, 0

        for nf, ef, ei, xt in loader:
            nf, ef, xt = nf.to(self.device), ef.to(self.device), xt.to(self.device)
            x_hat = self.model(nf, ef, edge_index)
            loss  = self.criterion(x_hat, xt)
            total += loss.item() * nf.size(0)
            count += nf.size(0)

        return total / count if count else 0.0

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _best_ckpt_path(self) -> str:
        return "_gnn_se_best.pt"

    def _save_best(self) -> None:
        torch.save(self.model.state_dict(), self._best_ckpt_path())

    def _load_best(self) -> None:
        path = self._best_ckpt_path()
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path: str) -> None:
        """
        Save model weights + metadata to `path`.

        Metadata includes: hidden_dim, n_mp_layers, slack_idx, branch_params.
        Load back with Trainer.load().
        """
        torch.save({
            "state_dict"   : self.model.state_dict(),
            "hidden_dim"   : self.model.hidden_dim,
            "n_mp_layers"  : self.model.n_mp_layers,
            "slack_idx"    : self.model.slack_idx,
            "branch_params": self.branch_params,
            "train_loss"   : self.history.train_loss,
            "val_loss"     : self.history.val_loss,
        }, path)
        print(f"  Saved checkpoint → {path}")

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None) -> GNNStateEstimator:
        """
        Load and return a GNNStateEstimator from a checkpoint saved by Trainer.save().

        Parameters
        ----------
        path   : file path to .pt checkpoint
        device : target device (auto-selected if None)

        Returns
        -------
        GNNStateEstimator  with weights restored, set to eval mode.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt  = torch.load(path, map_location=device)
        model = GNNStateEstimator(
            hidden_dim  = ckpt["hidden_dim"],
            n_mp_layers = ckpt["n_mp_layers"],
            slack_idx   = ckpt["slack_idx"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        print(f"  Loaded checkpoint ← {path}")
        return model

    # ── Evaluation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        samples     : Optional[List[GNNSESample]] = None,
        batch_size  : int = 128,
    ) -> Dict[str, float]:
        """
        Evaluate mean absolute errors for |V| and θ on a given sample list.
        If `samples` is None, the held-out validation set is used.

        Returns
        -------
        dict with keys:
            mae_V_pu      (p.u.)
            mae_theta_deg (degrees)
            mae_theta_rad (radians)
        """
        if samples is None:
            loader = DataLoader(
                self.val_dataset,
                batch_size = batch_size,
                shuffle    = False,
                collate_fn = collate_fn,
            )
        else:
            loader = DataLoader(
                GNNSEDataset(samples),
                batch_size = batch_size,
                shuffle    = False,
                collate_fn = collate_fn,
            )

        edge_index = GNNSEDataset.edge_index.to(self.device)
        self.model.eval()

        all_dV, all_dTh = [], []

        for nf, ef, ei, xt in loader:
            nf, ef, xt = nf.to(self.device), ef.to(self.device), xt.to(self.device)
            x_hat = self.model(nf, ef, edge_index)

            dTh = (x_hat[:, :, 0] - xt[:, :, 0]).abs()   # (B, n_buses)
            dV  = (x_hat[:, :, 1] - xt[:, :, 1]).abs()

            # Exclude slack from angle error
            mask = [i for i in range(N_BUSES) if i != self.slack_idx]
            dTh  = dTh[:, mask]

            all_dV.append(dV.cpu())
            all_dTh.append(dTh.cpu())

        mae_V  = torch.cat(all_dV).mean().item()
        mae_th = torch.cat(all_dTh).mean().item()

        return {
            "mae_V_pu"      : mae_V,
            "mae_theta_rad" : mae_th,
            "mae_theta_deg" : math.degrees(mae_th),
        }

    # ── Formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def _print_header() -> None:
        sep = "  " + "─" * 62
        print("\n  " + "═" * 62)
        print("  GNN State Estimator — Training")
        print("  " + "═" * 62)
        print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  "
              f"{'LR':>10}  {'Elapsed':>8}")
        print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED TRAINING DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    3-bus / 3-line training demo.

    Branch parameters match the runtime.py example:
        rs = 0.035,  xs = 0.25  →  z_s = 0.035 + 0.25j
        y_s = 1/z_s ≈ 0.5390 − 3.7736j   →  g_s = 0.5390, b_s = -3.7736
    """

    import warnings
    warnings.filterwarnings("ignore")

    # ── Network definition ─────────────────────────────────────────────────
    # (branch_id, fbus_0based, tbus_0based, g_s,    b_s,     b_sh)
    z_s   = complex(0.035, 0.25)
    y_s   = 1.0 / z_s
    g_s   = y_s.real
    b_s   = y_s.imag
    b_sh  = 0.0

    branch_params = [
        (1,  0, 1,  g_s,  b_s,  b_sh),   # Bus1 − Bus2
        (2,  0, 2,  g_s,  b_s,  b_sh),   # Bus1 − Bus3
        (3,  1, 2,  g_s,  b_s,  b_sh),   # Bus2 − Bus3
    ]

    print("\n" + "=" * 64)
    print("  GNN State Estimator — 3-Bus Demo")
    print("=" * 64)
    print(f"  Branch params (g_s={g_s:.4f}, b_s={b_s:.4f}, b_sh={b_sh})")

    # ── Generate data ─────────────────────────────────────────────────────
    gen = DataGenerator(
        branch_params = branch_params,
        slack_idx     = 0,
        n_buses       = 3,
        theta_range   = (-15.0, 15.0),
        V_range       = (0.90, 1.10),
    )
    print("\n  Generating 6 000 training samples …")
    samples = gen.generate(
        n_samples     = 6000,
        noise_sd      = 0.01,
        drop_prob     = 0.10,
        include_flows = True,
        seed          = 42,
    )
    print(f"  Done.  Sample 0 node_feat shape : "
          f"{samples[0].gnn_input.node_features.shape}")
    print(f"         Sample 0 edge_feat shape : "
          f"{samples[0].gnn_input.edge_features.shape}")
    print(f"         Sample 0 label V_pu      : "
          f"{samples[0].label.V_pu}")
    print(f"         Sample 0 label theta_rad : "
          f"{samples[0].label.theta_rad}")

    # ── Build model ────────────────────────────────────────────────────────
    model = GNNStateEstimator(
        hidden_dim  = 64,
        n_mp_layers = 4,
        slack_idx   = 0,
    )
    print(f"\n{model}")

    # ── Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model         = model,
        samples       = samples,
        branch_params = branch_params,
        slack_idx     = 0,
        lambda_theta  = 1.0,
        lambda_V      = 1.0,
        val_ratio     = 0.15,
    )

    history = trainer.train(
        epochs      = 150,
        batch_size  = 64,
        lr          = 1e-3,
        patience    = 25,
        lr_factor   = 0.5,
        lr_patience = 10,
        verbose     = True,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print("\n" + "=" * 64)
    print("  Evaluation on validation set")
    print("=" * 64)
    print(f"  MAE  |V|   : {metrics['mae_V_pu']:.6f}  p.u.")
    print(f"  MAE   θ    : {metrics['mae_theta_deg']:.6f}  degrees")
    print("=" * 64)

    # ── Save ───────────────────────────────────────────────────────────────
    trainer.save("gnn_se_checkpoint.pt")

    # ── Quick inference demo ───────────────────────────────────────────────
    test_sample = samples[-1]
    result = model.predict(test_sample.gnn_input)
    print("\n  Sample prediction vs ground truth")
    print(f"  {'Bus':>4}  {'|V| pred':>10}  {'|V| true':>10}  "
          f"{'θ pred (°)':>12}  {'θ true (°)':>12}")
    for i in range(3):
        print(f"  {i+1:>4}  "
              f"{result.V_pu[i]:>10.5f}  "
              f"{test_sample.label.V_pu[i]:>10.5f}  "
              f"{result.theta_deg[i]:>12.4f}  "
              f"{np.degrees(test_sample.label.theta_rad[i]):>12.4f}")