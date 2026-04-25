# gnn_se_model.py
"""
GNN-based AC State Estimator — Model Definition
================================================
Author  : (project)
Version : 1.0

Inputs → Solver → Outputs
--------------------------

┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT SUMMARY (all values in per-unit unless noted)                    │
│                                                                         │
│  Node features  (per bus i, shape [n_buses, 6]):                        │
│    [0] P_inject_i   Active power injection   (p.u.)  mtype='pinject'   │
│    [1] Q_inject_i   Reactive power injection (p.u.)  mtype='qinject'   │
│    [2] V_mag_i      Voltage magnitude        (p.u.)  mtype='vmag'      │
│    [3] mask_P       1 if P_inject measured, else 0                      │
│    [4] mask_Q       1 if Q_inject measured, else 0                      │
│    [5] mask_V       1 if V_mag    measured, else 0                      │
│                                                                         │
│  Edge features  (per directed edge i→j, shape [n_edges, 7]):            │
│    [0] P_flow_ij    Active power flow from i (p.u.)  mtype='pflow'     │
│    [1] Q_flow_ij    Reactive power flow from i(p.u.) mtype='qflow'     │
│    [2] mask_Pf      1 if P_flow measured, else 0                        │
│    [3] mask_Qf      1 if Q_flow measured, else 0                        │
│    [4] g_s          Series conductance  (p.u.)  from YBus               │
│    [5] b_s          Series susceptance  (p.u.)  from YBus               │
│    [6] b_sh         Half-line charging susceptance (p.u.) from YBus     │
│                                                                         │
│  Graph topology  (fixed for this system):                               │
│    edge_index : (2, n_edges)  — directed edges both ways per line       │
│    slack_mask : (n_buses,)    — 1 at the reference/slack bus index      │
│                                                                         │
│  OUTPUTS (per bus, shape [n_buses]):                                    │
│    Voltage magnitude  |V_i|  (p.u.)                                     │
│    Voltage angle       θ_i   (radians) — θ_slack forced to 0           │
└─────────────────────────────────────────────────────────────────────────┘

Architecture — Edge-Conditioned Message Passing GNN
----------------------------------------------------
1. Node encoder  : MLP  (6  → hidden_dim)
2. Edge encoder  : MLP  (7  → hidden_dim)
3. K × Message-Passing layers
       message_k  = MLP_msg( [h_i || h_j || e_ij] )    edge-conditioned
       agg_i      = Σ_{j∈N(i)} message_k               sum aggregation
       h_i'       = MLP_upd( [h_i || agg_i] ) + h_i    residual update
4. Output decoder: MLP  (hidden_dim → 2)  per node
       out[:,0]   = θ_i   (radians)
       out[:,1]   = |V_i| (p.u.)
   Post-process: zero out θ at slack bus index.

Dependencies
------------
    torch >= 1.13   (pure PyTorch — no PyG required)
    numpy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (3-bus, 3-line fixed topology)
# ══════════════════════════════════════════════════════════════════════════════

N_BUSES  : int = 3
N_LINES  : int = 3
N_EDGES  : int = N_LINES * 2    # directed (each line → both directions)

NODE_FEAT_DIM : int = 6         # [P, Q, V, mask_P, mask_Q, mask_V]
EDGE_FEAT_DIM : int = 7         # [Pf, Qf, mask_Pf, mask_Qf, g_s, b_s, b_sh]
OUTPUT_DIM    : int = 2         # [θ_i (rad), |V_i| (p.u.)]

# Default topology:  Bus1-Bus2 (line 1), Bus1-Bus3 (line 2), Bus2-Bus3 (line 3)
# Directed edges: each undirected line becomes (i→j) and (j→i)
# Index mapping: bus id 1→0, 2→1, 3→2
_DEFAULT_EDGE_INDEX: torch.Tensor = torch.tensor([
    [0, 1, 0, 2, 1, 2],   # source
    [1, 0, 2, 0, 2, 1],   # destination
], dtype=torch.long)

# Which edge index (in the 6-edge list above) corresponds to line_id {1,2,3}
# from-direction: line1→edge0, line2→edge2, line3→edge4
# to-direction:   line1→edge1, line2→edge3, line3→edge5
_LINE_TO_EDGE_FROM : Dict[int, int] = {1: 0, 2: 2, 3: 4}
_LINE_TO_EDGE_TO   : Dict[int, int] = {1: 1, 2: 3, 3: 5}


# ══════════════════════════════════════════════════════════════════════════════
# DATA CONTAINERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GNNSEInput:
    """
    One sample (graph) ready to feed into GNNStateEstimator.

    All numeric values are **per-unit**.  Unavailable measurements are set to
    0.0 and their corresponding mask entry is 0.

    Attributes
    ----------
    node_features : ndarray (n_buses, NODE_FEAT_DIM)
        [P_inject, Q_inject, V_mag, mask_P, mask_Q, mask_V]
    edge_features : ndarray (n_edges, EDGE_FEAT_DIM)
        [P_flow, Q_flow, mask_Pf, mask_Qf, g_s, b_s, b_sh]
    edge_index    : ndarray (2, n_edges)  long — src / dst bus indices
    slack_idx     : int    — 0-based index of the slack/reference bus
    """
    node_features : np.ndarray            # (n_buses, 6)
    edge_features : np.ndarray            # (n_edges, 7)
    edge_index    : np.ndarray            # (2, n_edges)  int
    slack_idx     : int = 0


@dataclass
class GNNSEOutput:
    """
    Predicted state returned by GNNStateEstimator.forward() or .predict().

    Attributes
    ----------
    V_pu      : ndarray (n_buses,)  — voltage magnitude per bus [p.u.]
    theta_rad : ndarray (n_buses,)  — voltage angle per bus [radians]
    theta_deg : ndarray (n_buses,)  — voltage angle per bus [degrees]
    """
    V_pu      : np.ndarray
    theta_rad : np.ndarray
    theta_deg : np.ndarray


@dataclass
class GNNSELabel:
    """
    Ground-truth state label used during training.

    Attributes
    ----------
    V_pu      : ndarray (n_buses,)  — true voltage magnitudes [p.u.]
    theta_rad : ndarray (n_buses,)  — true voltage angles [radians]
    """
    V_pu      : np.ndarray
    theta_rad : np.ndarray


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def _mlp(in_dim: int, out_dim: int, hidden: int = 64, layers: int = 2) -> nn.Sequential:
    """Utility — build a small feed-forward MLP with LayerNorm + ReLU."""
    sizes  = [in_dim] + [hidden] * layers + [out_dim]
    blocks: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        blocks.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            blocks.append(nn.LayerNorm(sizes[i + 1]))
            blocks.append(nn.ReLU())
    return nn.Sequential(*blocks)


class MessagePassingLayer(nn.Module):
    """
    One round of edge-conditioned message passing.

    Step 1 — Compute message for edge (i → j):
        m_ij = MLP_msg( [h_i ‖ h_j ‖ e_ij] )

    Step 2 — Aggregate incoming messages at node j:
        agg_j = Σ_{i∈N(j)} m_ij       (sum)

    Step 3 — Update node embedding with residual:
        h_j' = MLP_upd( [h_j ‖ agg_j] ) + h_j

    Edge embeddings are **not** updated (they carry fixed physical parameters
    plus measurement values — updating them could corrupt the physics signal).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message MLP: [h_i || h_j || e_ij] → hidden_dim
        self.msg_mlp = _mlp(
            in_dim = hidden_dim * 2 + hidden_dim,   # src + dst + edge
            out_dim = hidden_dim,
            hidden  = hidden_dim,
            layers  = 2,
        )

        # Update MLP: [h_j || agg_j] → hidden_dim
        self.upd_mlp = _mlp(
            in_dim  = hidden_dim * 2,
            out_dim = hidden_dim,
            hidden  = hidden_dim,
            layers  = 2,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h          : torch.Tensor,    # (n_buses, hidden_dim)
        e          : torch.Tensor,    # (n_edges, hidden_dim)
        edge_index : torch.Tensor,    # (2, n_edges)  long
    ) -> torch.Tensor:
        """
        Returns updated node embeddings  h' : (n_buses, hidden_dim).
        """
        n_buses = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        # ── Step 1: messages ─────────────────────────────────────────────
        h_src  = h[src]   # (n_edges, hidden_dim)
        h_dst  = h[dst]   # (n_edges, hidden_dim)
        msg_in = torch.cat([h_src, h_dst, e], dim=-1)   # (n_edges, 3*hidden)
        msgs   = self.msg_mlp(msg_in)                   # (n_edges, hidden)

        # ── Step 2: aggregate (scatter-sum) ─────────────────────────────
        agg = torch.zeros(n_buses, self.hidden_dim, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msgs), msgs)

        # ── Step 3: update with residual ─────────────────────────────────
        upd_in = torch.cat([h, agg], dim=-1)            # (n_buses, 2*hidden)
        h_new  = self.upd_mlp(upd_in) + h               # residual
        return self.norm(h_new)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GNNStateEstimator(nn.Module):
    """
    Graph Neural Network AC State Estimator.

    Maps a set of power system measurements (P/Q injections, P/Q flows,
    voltage magnitudes) to the full AC state (voltage magnitudes + angles)
    for every bus.

    Parameters
    ----------
    hidden_dim   : width of all hidden layers (default 64)
    n_mp_layers  : number of message-passing rounds (default 4)
    slack_idx    : 0-based index of the reference/slack bus (default 0)
                   The predicted angle at this bus is zeroed out in
                   post-processing so the model never needs to learn it.

    Forward pass
    ------------
    Input  tensors (all float32):
        node_feat  : (batch, n_buses, NODE_FEAT_DIM)
        edge_feat  : (batch, n_edges, EDGE_FEAT_DIM)
        edge_index : (2, n_edges)  long  — shared across batch

    Output tensor (float32):
        x_hat : (batch, n_buses, OUTPUT_DIM)
                  [:,  :, 0]  = θ_i   (radians)
                  [:, :, 1]   = |V_i| (p.u.)
    """

    def __init__(
        self,
        hidden_dim  : int = 64,
        n_mp_layers : int = 4,
        slack_idx   : int = 0,
    ) -> None:
        super().__init__()

        self.hidden_dim  = hidden_dim
        self.n_mp_layers = n_mp_layers
        self.slack_idx   = slack_idx

        # ── Encoders ─────────────────────────────────────────────────────
        self.node_encoder = _mlp(NODE_FEAT_DIM, hidden_dim, hidden=hidden_dim, layers=2)
        self.edge_encoder = _mlp(EDGE_FEAT_DIM, hidden_dim, hidden=hidden_dim, layers=2)

        # ── Message-passing stack ─────────────────────────────────────────
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(n_mp_layers)
        ])

        # ── Output decoder  ───────────────────────────────────────────────
        # hidden → [θ_i, |V_i|]  per node
        self.decoder = _mlp(hidden_dim, OUTPUT_DIM, hidden=hidden_dim // 2, layers=2)

    # ── Forward (batched) ────────────────────────────────────────────────────

    def forward(
        self,
        node_feat  : torch.Tensor,   # (B, n_buses, 6)
        edge_feat  : torch.Tensor,   # (B, n_edges, 7)
        edge_index : torch.Tensor,   # (2, n_edges)  long
    ) -> torch.Tensor:               # (B, n_buses, 2)
        """
        Batched forward pass.

        Returns x_hat of shape (B, n_buses, 2):
            x_hat[:, :, 0] = θ_i  (radians)   — slack bus zeroed
            x_hat[:, :, 1] = |V_i| (p.u.)
        """
        B = node_feat.size(0)

        # ── Encode ───────────────────────────────────────────────────────
        # Process each graph in the batch independently (graph is same topology)
        h = self.node_encoder(node_feat)    # (B, n_buses, hidden)
        e = self.edge_encoder(edge_feat)    # (B, n_edges, hidden)

        # ── Message passing  (loop over batch dimension) ──────────────────
        # Each sample is an independent graph — same edge_index for all
        h_out = torch.stack([
            self._mp_single(h[b], e[b], edge_index)
            for b in range(B)
        ])                                  # (B, n_buses, hidden)

        # ── Decode ───────────────────────────────────────────────────────
        x_hat = self.decoder(h_out)         # (B, n_buses, 2)

        # ── Post-process: zero slack angle ────────────────────────────────
        x_hat = x_hat.clone()
        x_hat[:, self.slack_idx, 0] = 0.0  # θ_slack ≡ 0

        return x_hat

    def _mp_single(
        self,
        h          : torch.Tensor,   # (n_buses, hidden)
        e          : torch.Tensor,   # (n_edges, hidden)
        edge_index : torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.mp_layers:
            h = layer(h, e, edge_index)
        return h

    # ── Convenience: numpy predict ───────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        gnn_input  : GNNSEInput,
        device     : torch.device = torch.device("cpu"),
    ) -> GNNSEOutput:
        """
        Predict the full AC state from a single GNNSEInput object.

        Returns a GNNSEOutput with numpy arrays (|V|, θ_rad, θ_deg).
        """
        self.eval()

        nf = torch.tensor(gnn_input.node_features, dtype=torch.float32, device=device).unsqueeze(0)
        ef = torch.tensor(gnn_input.edge_features,  dtype=torch.float32, device=device).unsqueeze(0)
        ei = torch.tensor(gnn_input.edge_index,      dtype=torch.long,   device=device)

        x_hat = self.forward(nf, ef, ei)   # (1, n_buses, 2)
        x_hat = x_hat.squeeze(0).cpu().numpy()

        theta_rad = x_hat[:, 0]
        V_pu      = np.abs(x_hat[:, 1])   # magnitude must be positive

        return GNNSEOutput(
            V_pu      = V_pu,
            theta_rad = theta_rad,
            theta_deg = np.degrees(theta_rad),
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"GNNStateEstimator(\n"
            f"  hidden_dim   = {self.hidden_dim}\n"
            f"  n_mp_layers  = {self.n_mp_layers}\n"
            f"  slack_idx    = {self.slack_idx}\n"
            f"  n_parameters = {self.n_parameters():,}\n"
            f")"
        )


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER   (Measurements → GNNSEInput)
# ══════════════════════════════════════════════════════════════════════════════

class GraphBuilder:
    """
    Converts raw per-unit measurement values into a GNNSEInput graph object
    that the model can directly consume.

    This is the bridge between the existing System/Measurement pipeline and
    the GNN.  It accepts the same mtype strings used throughout the project:
        'pinject', 'qinject', 'vmag', 'pflow', 'qflow'

    Parameters
    ----------
    n_buses     : number of buses (default 3)
    branch_params : list of (branch_id, fbus_idx, tbus_idx, g_s, b_s, b_sh)
                    one entry per transmission line (0-based bus indices)
    slack_idx   : 0-based index of the slack bus (default 0)

    Usage
    -----
        builder = GraphBuilder(
            n_buses = 3,
            branch_params = [
                (1, 0, 1, g1, b1, bsh1),
                (2, 0, 2, g2, b2, bsh2),
                (3, 1, 2, g3, b3, bsh3),
            ],
            slack_idx = 0,
        )

        gnn_input = builder.build(measurements_list)   # list of Measurement
    """

    def __init__(
        self,
        n_buses       : int,
        branch_params : List[Tuple[int, int, int, float, float, float]],
        slack_idx     : int = 0,
    ) -> None:
        self.n_buses      = n_buses
        self.branch_params = branch_params   # [(br_id, fi, ti, g_s, b_s, b_sh)]
        self.slack_idx    = slack_idx

        n_lines = len(branch_params)

        # Build edge_index (directed, both ways per line)
        src_list, dst_list = [], []
        for _, fi, ti, *_ in branch_params:
            src_list += [fi, ti]
            dst_list += [ti, fi]
        self.edge_index = np.array([src_list, dst_list], dtype=np.int64)  # (2, 2*n_lines)

        # Pre-fill edge physical features (g_s, b_s, b_sh) — same for both
        # directions of a line; measurement slots filled later
        n_edges = n_lines * 2
        self._edge_phys = np.zeros((n_edges, 3), dtype=np.float32)   # [g_s, b_s, b_sh]
        for k, (_, fi, ti, g_s, b_s, b_sh) in enumerate(branch_params):
            # edge 2k   = fi → ti (from direction)
            # edge 2k+1 = ti → fi (to direction)
            self._edge_phys[2 * k]     = [g_s, b_s, b_sh]
            self._edge_phys[2 * k + 1] = [g_s, b_s, b_sh]

        # Map branch_id → edge row indices
        self._br_id_to_edge: Dict[int, Tuple[int, int]] = {
            br_id: (2 * k, 2 * k + 1)
            for k, (br_id, *_) in enumerate(branch_params)
        }

    def build(self, measurements: list) -> GNNSEInput:
        """
        Build a GNNSEInput from a list of Measurement objects (already in p.u.).

        Parameters
        ----------
        measurements : List[Measurement]   — each has .mtype, .pos_id,
                                             .mvalue_pu, .mside (for flows)

        Returns
        -------
        GNNSEInput ready for GNNStateEstimator.forward()
        """
        n_edges = self.edge_index.shape[1]

        # ── Node features: [P_inj, Q_inj, V_mag, m_P, m_Q, m_V] ─────────
        node_feat = np.zeros((self.n_buses, NODE_FEAT_DIM), dtype=np.float32)

        # ── Edge features: [P_flow, Q_flow, m_Pf, m_Qf, g_s, b_s, b_sh] ─
        edge_feat = np.zeros((n_edges, EDGE_FEAT_DIM), dtype=np.float32)
        edge_feat[:, 4:7] = self._edge_phys    # fill fixed physics

        for m in measurements:
            mt    = (m.mtype or "").lower()
            val   = m.mvalue_pu if m.mvalue_pu is not None else 0.0
            b_idx = m.pos_id - 1   # convert 1-based bus id to 0-based

            if mt == "pinject":
                node_feat[b_idx, 0] = val
                node_feat[b_idx, 3] = 1.0   # mask_P
            elif mt == "qinject":
                node_feat[b_idx, 1] = val
                node_feat[b_idx, 4] = 1.0   # mask_Q
            elif mt == "vmag":
                node_feat[b_idx, 2] = val
                node_feat[b_idx, 5] = 1.0   # mask_V
            elif mt in ("pflow", "qflow"):
                br_id = m.pos_id
                if br_id not in self._br_id_to_edge:
                    continue
                e_from, e_to = self._br_id_to_edge[br_id]
                side = (m.mside or "from").lower()
                e_idx = e_from if side == "from" else e_to
                if mt == "pflow":
                    edge_feat[e_idx, 0] = val
                    edge_feat[e_idx, 2] = 1.0   # mask_Pf
                else:
                    edge_feat[e_idx, 1] = val
                    edge_feat[e_idx, 3] = 1.0   # mask_Qf

        return GNNSEInput(
            node_features = node_feat,
            edge_features = edge_feat,
            edge_index    = self.edge_index,
            slack_idx     = self.slack_idx,
        )

    @classmethod
    def from_ybus(cls, ybus, slack_idx: int = 0) -> "GraphBuilder":
        """
        Convenience constructor — build a GraphBuilder directly from a
        YBusMatrix object (from math_model.static_parameter).

        Parameters
        ----------
        ybus      : YBusMatrix
        slack_idx : 0-based slack bus index in the YBus ordering

        Example
        -------
            builder = GraphBuilder.from_ybus(grid.ybus, slack_idx=0)
        """
        params = []
        for br in ybus.branches:
            g_s, b_s, b_sh = ybus.branch_params(br.id)
            fi = ybus.bus_index(br.fbus)
            ti = ybus.bus_index(br.tbus)
            params.append((br.id, fi, ti, g_s, b_s, b_sh))

        return cls(
            n_buses       = ybus.n,
            branch_params = params,
            slack_idx     = slack_idx,
        )


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class SELoss(nn.Module):
    """
    Weighted MSE loss for state estimation.

    Computes separate loss terms for voltage angles (θ) and magnitudes (|V|),
    weighted by lambda_theta and lambda_V respectively.  The slack-bus angle
    is excluded from the angle loss since it is always 0.

    Loss = lambda_theta * MSE(θ_pred, θ_true)   [non-slack buses only]
         + lambda_V     * MSE(V_pred, V_true)   [all buses]
    """

    def __init__(
        self,
        lambda_theta : float = 1.0,
        lambda_V     : float = 1.0,
        slack_idx    : int   = 0,
    ) -> None:
        super().__init__()
        self.lambda_theta = lambda_theta
        self.lambda_V     = lambda_V
        self.slack_idx    = slack_idx

    def forward(
        self,
        x_hat  : torch.Tensor,   # (B, n_buses, 2)  predicted  [θ, V]
        x_true : torch.Tensor,   # (B, n_buses, 2)  ground truth
    ) -> torch.Tensor:
        """
        Returns scalar loss value.
        """
        # Indices of non-slack buses
        all_idx      = list(range(x_hat.size(1)))
        non_slack    = [i for i in all_idx if i != self.slack_idx]

        theta_pred   = x_hat[:, non_slack, 0]
        theta_true   = x_true[:, non_slack, 0]
        V_pred       = x_hat[:, :, 1]
        V_true       = x_true[:, :, 1]

        loss_theta   = nn.functional.mse_loss(theta_pred, theta_true)
        loss_V       = nn.functional.mse_loss(V_pred,     V_true)

        return self.lambda_theta * loss_theta + self.lambda_V * loss_V


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  GNNStateEstimator — architecture check")
    print("=" * 60)

    model = GNNStateEstimator(hidden_dim=64, n_mp_layers=4, slack_idx=0)
    print(model)

    # Dummy batch: 8 samples
    B = 8
    nf = torch.randn(B, N_BUSES,  NODE_FEAT_DIM)
    ef = torch.randn(B, N_EDGES,  EDGE_FEAT_DIM)
    ei = _DEFAULT_EDGE_INDEX

    out = model(nf, ef, ei)
    print(f"\n  Input  node_feat : {nf.shape}")
    print(f"  Input  edge_feat : {ef.shape}")
    print(f"  Output x_hat     : {out.shape}  →  [batch, bus, (θ, |V|)]")

    print(f"\n  Slack angle (bus {model.slack_idx}) all-zero : "
          f"{torch.all(out[:, model.slack_idx, 0] == 0).item()}")

    loss_fn = SELoss(lambda_theta=1.0, lambda_V=1.0, slack_idx=0)
    dummy_label = torch.randn(B, N_BUSES, OUTPUT_DIM)
    loss = loss_fn(out, dummy_label)
    print(f"\n  Dummy loss value : {loss.item():.6f}")
    print("=" * 60)