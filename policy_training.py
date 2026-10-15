# Combined flat-direction RL training script

# Generated from the two user-provided scripts.

# Objective: find sparse operator subsets with the smallest SM-like chi2 at fixed ||C||.

import json

import numpy as np

import flavio



# ============================================================

# 1. CONFIGURATION & EXPERIMENTAL CONSTRAINTS



# ============================================================

obs_correspondence = {

    'GammaZ': 'GammaZ',

    'sigma_had': 'sigma_had',

    'R_e': 'R_e',

    'R_mu': 'R_mu',

    'R_tau': 'R_tau',

    'AFB(Z->ee)': 'AFB(Z->ee)',

    'AFB(Z->mumu)': 'AFB(Z->mumu)',

    'AFB(Z->tautau)': 'AFB(Z->tautau)',

    'A(Z->ee)': 'A(Z->ee)',

    'A(Z->mumu)': 'A(Z->mumu)',

    'A(Z->tautau)': 'A(Z->tautau)',

    'GammaW': 'GammaW',

    'BR(W->enu)': 'BR(W->enu)',

    'BR(W->munu)': 'BR(W->munu)',

    'BR(W->taunu)': 'BR(W->taunu)',

    'R(W->cX)': 'R(W->cX)',

    'Rmue(W->lnu)': 'Rmue(W->lnu)',

    'Rtaue(W->lnu)': 'Rtaue(W->lnu)',

    'Rtaumu(W->lnu)': 'Rtaumu(W->lnu)',

    'A(Z->ss)': 'A(Z->ss)',

    'm_W': 'm_W',          # Delta F=0

    'AFB(Z->bb)': 'AFB(Z->bb)',

    'BR(B+->Knunu)': 'BR(B+->Knunu)',  # CC

    'Rtaul(B->Dlnu)': 'Rtaul(B->Dlnu)',

    'epsp/eps': 'epsp/eps', # Delta F=1

    # Binned observables

    ("<P5p>(B0->K*mumu)", 4, 6): 'P5p46', # Delta F=1

    ("<dBR/dq2>(Bs->phimumu)", 1, 6): 'Bsphimumu16',

    'BR(K+->pinunu)': 'BR(K+->pinunu)', # Delta F=1

    'BR(KL->pinunu)': 'BR(KL->pinunu)',

    'DeltaM_s': 'DeltaM_s', # Delta F=1

    'eps_K': 'eps_K', # constraint

    'BR(Bs->mumu)': 'BR(Bs->mumu)', # constraint

}

constraints_ewp = keys = list(obs_correspondence.values())

constraints_ewp_exp = list(obs_correspondence.keys())

SIGMA_TARGET = 1.0

EXPONENT_CLAMP = 50.0

# Path to pre-computed SMEFT linear expansion coefficients

COEFFICIENTS_PATH = (

#    "/home/kumarj/links/scratch/tmp_real/ewp_linear_coefficients_quad_fixed.json"

    "/home/kumarj/links/scratch/tmp_real/ewp_linear_coefficients_2.json"

)



# ============================================================

# 3. LOAD LINEAR SMEFT COEFFICIENTS



# ============================================================

with open(COEFFICIENTS_PATH, "r", encoding="utf-8") as f:

    ewp_data = json.load(f)

ewp_coefficients = ewp_data["observables"]

ewp_coefficients_meta = ewp_data["metadata"]



# ============================================================

# 4. KINEMATIC & PREDICTION HELPERS



# ============================================================



def predict_ewp(obs: str, wc_dict: dict, ewp_data: dict) -> float:

    """

    Computes linearized SMEFT prediction for a single observable, per the

    linear-only JSON schema:

        O/O_SM = 1 + sum_i coefficient_i * (NORMALIZE * C_i)

    Parameters

    ----------

    obs : str

        Observable name, must be a key in ewp_data["observables"].

    wc_dict : dict

        Wilson coefficients {operator_name: value}, in the *raw*

        (un-normalized) convention -- normalization is applied here.

    ewp_data : dict

        The full parsed JSON (contains "metadata" and "observables").

    """

    observables = ewp_data.get("observables", {})

    if obs not in observables:

        raise KeyError(f"Observable '{obs}' not found in loaded JSON data.")

    data = observables[obs]

    metadata = ewp_data.get("metadata", {})

    normalize = metadata.get("normalize", 1.0)

    sm = data.get("sm_prediction", 0.0)

    coeffs = data.get("coefficients", {})

    # Scale each relevant Wilson coefficient by NORMALIZE once

    scaled_wc = {

        wc: val #* normalize

        for wc, val in wc_dict.items()

        if wc in coeffs

    }

    linear_term = sum(

        coeffs[wc] * scaled_wc[wc]

        for wc in scaled_wc

    )

    delta = linear_term

    return sm * (1 + delta)



def predict_observables(wc_dict: dict) -> dict:

    """Computes predictions for all observables in constraints_ewp."""

    return {obs: predict_ewp(obs, wc_dict, ewp_data) for obs in constraints_ewp}



# ============================================================

# EXPERIMENTAL CONSTRAINTS & CHI2 HELPERS



# ============================================================

obs_correspondence_reverse = {

    value: key

    for key, value in obs_correspondence.items()

}



# ============================================================

# FLAT-DIRECTION OBJECTIVE



# ============================================================

# RL chooses a sparse operator subset S. For each selected subset, solve

#

#     minimize chi2_SM_like(C_S)

#     subject to ||C_S|| = FIXED_WC_NORM.

#

# All subsets are therefore compared at exactly the same NP coefficient

# radius. Because the observable model is linear in the Wilson coefficients,

# chi2_SM_like is a quadratic form and the constrained minimum is obtained

# exactly from the smallest eigenvector of the selected Fisher/Gram matrix.

# RL is used only for the combinatorial operator-subset search.

FIXED_WC_NORM = 1.0

FLAT_REWARD_EPS = 1e-12

SPARSITY_PENALTY = 0.05

# Later operators are accepted only if they reduce fixed-norm SM-like chi2
# by at least this many decades. 0.05 is about a 12% reduction.
MIN_LOG10_CHI2_IMPROVEMENT = 0.05
REJECTED_STEP_PENALTY = 0.20

SM_PREDICTIONS = predict_observables({})

FLAT_SIGMAS = {}

FLAT_OBSERVABLES = []

for _obs_name in constraints_ewp:

    _theory_sigma = float(

        ewp_coefficients[_obs_name].get("sm_uncertainty", 0.0)

    )

    if not np.isfinite(_theory_sigma) or _theory_sigma <= 0.0:

        print(

            f"[flat-direction] Excluding {_obs_name}: "

            f"theory uncertainty = {_theory_sigma}"

        )

        continue

    FLAT_SIGMAS[_obs_name] = _theory_sigma

    FLAT_OBSERVABLES.append(_obs_name)



if not FLAT_OBSERVABLES:

    raise ValueError("No observables have positive theory uncertainty.")



print(

    f"[flat-direction] Using {len(FLAT_OBSERVABLES)}/"

    f"{len(constraints_ewp)} observables with positive theory uncertainty"

)





# RL observable space is defined only after filtering invalid theory uncertainties.

OBS_NAMES = list(FLAT_OBSERVABLES)

N_OBS = len(OBS_NAMES)

FLAT_BASELINE_PULLS = np.zeros(N_OBS, dtype=np.float32)



def compute_sm_like_pulls(wc_dict: dict) -> list[float]:

    """Return (O_NP - O_SM) / sigma_theory for every observable."""

    predictions = predict_observables(wc_dict)

    pulls = []

    for obs_name in FLAT_OBSERVABLES:

        delta = float(predictions[obs_name]) - float(SM_PREDICTIONS[obs_name])

        pulls.append(float(delta / FLAT_SIGMAS[obs_name]))

    return pulls



def compute_sm_like_chi2(wc_dict: dict) -> float:

    """chi2_SM-like = sum_i [(O_i(C)-O_i(SM))/sigma_theory_i]^2."""

    pulls = np.asarray(compute_sm_like_pulls(wc_dict), dtype=np.float64)

    return float(np.dot(pulls, pulls))



def _build_pull_sensitivity(wc_names: list[str]) -> np.ndarray:

    """

    Build B with delta_pull = B @ C for the selected operators.

    For the linear model

        O = O_SM * (1 + sum_j a_j C_j),

    so

        B_ij = O_SM_i * a_ij / sigma_theory_i.

    """

    B = np.zeros((len(FLAT_OBSERVABLES), len(wc_names)), dtype=np.float64)

    for i, obs_name in enumerate(FLAT_OBSERVABLES):

        data = ewp_coefficients[obs_name]

        sm = float(data.get("sm_prediction", SM_PREDICTIONS[obs_name]))

        coeffs = data.get("coefficients", {})

        sigma = FLAT_SIGMAS[obs_name]

        for j, wc_name in enumerate(wc_names):

            B[i, j] = sm * float(coeffs.get(wc_name, 0.0)) / sigma

    return B



def _minimise_sm_deviation_at_fixed_norm(

    wc_names: list[str],

    fixed_norm: float = FIXED_WC_NORM,

) -> tuple[float, dict]:

    """

    Minimize chi2_SM-like subject to ||C|| = fixed_norm.

    With linear predictions, chi2 = C^T (B^T B) C. At fixed norm,

    the exact minimum is fixed_norm^2 times the smallest eigenvalue of B^T B.

    """

    n = len(wc_names)

    if n == 0:

        return 0.0, {}

    if fixed_norm <= 0.0:

        raise ValueError("fixed_norm must be positive")

    B = _build_pull_sensitivity(wc_names)

    fisher = B.T @ B

    eigenvalues, eigenvectors = np.linalg.eigh(fisher)

    idx = int(np.argmin(eigenvalues))

    direction = np.asarray(eigenvectors[:, idx], dtype=np.float64)

    direction_norm = float(np.linalg.norm(direction))

    if not np.isfinite(direction_norm) or direction_norm <= 0.0:

        raise RuntimeError(f"Invalid minimum-eigenvalue direction for {wc_names}")

    coeffs = fixed_norm * direction / direction_norm

    best_wc = {wc_names[i]: float(coeffs[i]) for i in range(n)}

    min_chi2 = compute_sm_like_chi2(best_wc)

    if not np.isfinite(min_chi2):

        raise RuntimeError(f"Non-finite fixed-norm chi2 for {wc_names}")

    return float(min_chi2), best_wc



def flat_direction_fn(fired_ops: list) -> tuple:

    """

    Environment used by RL. For the selected operators, return the direction

    with ||C|| = FIXED_WC_NORM that is least visible relative to the SM.

    Returns

    -------

    OBS_NAMES, sm_like_pulls, min_sm_like_chi2, predictions, best_wc

    """

    if len(fired_ops) == 0:

        best_wc = {}

        min_chi2 = 0.0

    else:

        min_chi2, best_wc = _minimise_sm_deviation_at_fixed_norm(fired_ops)

    prediction_dict = predict_observables(best_wc)

    predictions = [float(prediction_dict[o]) for o in FLAT_OBSERVABLES]

    pulls = compute_sm_like_pulls(best_wc)

    sm_like_chi2 = float(np.dot(np.asarray(pulls), np.asarray(pulls)))

    if not np.all(np.isfinite(pulls)):

        raise ValueError(f"Non-finite flat-direction pulls: {pulls}")

    if len(fired_ops) > 0 and not np.isclose(

        sum(float(v) ** 2 for v in best_wc.values()),

        FIXED_WC_NORM ** 2,

        rtol=1e-7,

        atol=1e-9,

    ):

        raise RuntimeError("Fixed-norm optimizer returned a WC vector with wrong norm")

    return OBS_NAMES, pulls, sm_like_chi2, predictions, best_wc



# ============================================================

# RL TRAINING



# ============================================================

import sys

from pathlib import Path

import copy

import random

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import contextlib

sys.path.insert(

    0,

    "/home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly")

from smeft_new.state import SMEFTState, step

from smeft_new.helpers import print_policy_diagnostics

#from smeft_new.sensitivity_matrix import M_ij

# ------------------------------------------------------------

# LOAD OPERATOR VOCABULARY

# ------------------------------------------------------------

import argparse

import warnings

import flavio

import wilson

from wilson import wcxf

warnings.filterwarnings("ignore")



def _build_catalogue_from_wcxf() -> dict[str, list[str]]:

    """

    Query the wcxf package for every WC in the SMEFT Warsaw basis and

    organise them by their wcxf 'sector' label.

    Returns

    -------

    dict  {sector_name: [wc_name, ...]}  - ordered by sector, then WC name

    """

    try:

        basis_obj = wcxf.Basis["SMEFT", "Warsaw"]

        catalogue: dict[str, list[str]] = {}

        for sector_name, sector_data in basis_obj.sectors.items():

            if sector_name == "dB=de=dmu=dtau=0":

                wcs = sorted(sector_data.keys())

                if wcs:

                    catalogue[sector_name] = wcs

        return catalogue

    except Exception as exc:

        warnings.warn(

            f"Could not load Warsaw basis from wcxf ({exc}). "

            "Falling back to a minimal built-in catalogue."

        )

        return _fallback_catalogue()

OPERATOR_CATALOGUE: dict[str, list[str]] = _build_catalogue_from_wcxf()

all_operator_vocab: list[str] = [

    wc for ops in OPERATOR_CATALOGUE.values() for wc in ops

]



def _operator_has_observable_sensitivity(wc_name: str, tol: float = 0.0) -> bool:

    """Exclude operators that are identically invisible to the chosen dataset."""

    for obs_name in FLAT_OBSERVABLES:

        coeff = float(

            ewp_coefficients[obs_name].get("coefficients", {}).get(wc_name, 0.0)

        )

        if abs(coeff) > tol:

            return True

    return False

operator_vocab: list[str] = [

    wc for wc in all_operator_vocab if _operator_has_observable_sensitivity(wc)

]

print(

    f"Loaded {len(operator_vocab)} observable-sensitive operators "

    f"out of {len(all_operator_vocab)} Warsaw-basis operators"

)

print(operator_vocab[:100])

# Adding a copy method to SMEFTState class

# This assumes SMEFTState is a dataclass or similar simple class



def _smeftstate_copy(self):

    new_state = SMEFTState(

        pulls=np.copy(self.pulls),

        chi2=self.chi2,

        obs_names=self.obs_names,

    )

    new_state.active_ops = list(getattr(self, "active_ops", []))

    new_state.wc_values = copy.deepcopy(getattr(self, "wc_values", {}))

    new_state.step = int(getattr(self, "step", 0))

    return new_state

SMEFTState.copy = _smeftstate_copy



def _smeftstate_to_dict(self):

    return {

        "pulls": np.asarray(self.pulls).tolist(),

        "chi2": float(self.chi2),

        "obs_names": self.obs_names,

        "active_ops": list(getattr(self, "active_ops", [])),

        "wc_values": {

            k: float(v) for k, v in getattr(self, "wc_values", {}).items()

        },

        "step": int(getattr(self, "step", 0)),

    }

SMEFTState.to_dict = _smeftstate_to_dict

SEED = 42

print(f"Using seed = {SEED}")

random.seed(SEED)

np.random.seed(SEED)

torch.cuda.manual_seed_all(SEED)

n_ops = len(operator_vocab)

n_obs = len(OBS_NAMES)

M_ij = np.zeros(

    (n_ops,n_obs),

    dtype=np.float32

)

print("M_ij shape =", M_ij.shape)

print("operator_vocab =", len(operator_vocab))

print("flat-direction observables =", len(OBS_NAMES))



# ============================================================

# SMEFT ACTOR-CRITIC

#

# IMPORTANT STATE DEFINITION



# ============================================================

#

# The Markov state is explicitly

#

#   s_t = (

#       pulls_t,

#       chi2_t,

#       active_operator_mask_t,

#       fired_operator_mask_t,

#       step_t

#   )

#

# BOTH actor and critic receive this SAME state.

#

# Actor:

#

#   pi_theta(a | s_t)

#

# Critic:

#

#   V_phi(s_t)

#

# The sampled return-to-go

#

#   G_t = r_t + gamma r_{t+1} + ...

#

# is used as the Monte-Carlo estimate of

#

#   Q^pi(s_t, a_t)

#

# and

#

#   A_t = G_t - V_phi(s_t)

#



# ============================================================

import os

import json

import random

import numpy as np

import scipy.stats

import torch

import torch.nn as nn

import torch.nn.functional as F



# ============================================================

# DEVICE



# ============================================================

DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)



# ============================================================

# MODEL



# ============================================================



class SMEFTActorCritic(nn.Module):

    """

    Actor-Critic model with an explicitly shared state definition.

    STATE:

        s_t = (

            pulls,

            chi2,

            active operator mask,

            fired/evaluated operator mask,

            step

        )

    Both actor and critic are functions of exactly this state.

    Actor:

        pi(a | s)

    Critic:

        V(s)

    """

    def __init__(

        self,

        op_vocab,

        obs_names,

        sensitivity_matrix,

        chi2_reference,

        max_steps=10,

        embed_dim=32,

        n_heads=4,

        alpha_init=1.0,

        pull_gain=2.5,

    ):

        super().__init__()

        self.op_vocab = op_vocab

        self.obs_names = obs_names

        self.num_ops = len(op_vocab)

        self.num_obs = len(obs_names)

        self.embed_dim = embed_dim

        self.max_steps = max_steps

        # ----------------------------------------------------

        # Operator -> integer index

        # ----------------------------------------------------

        self.op_to_idx = {

            op: i

            for i, op in enumerate(op_vocab)

        }

        # ----------------------------------------------------

        # Sensitivity matrix

        # ----------------------------------------------------

        sm_shape = tuple(

            sensitivity_matrix.shape

        )

        expected = (

            self.num_obs,

            self.num_ops

        )

        expected_T = (

            self.num_ops,

            self.num_obs

        )

        if sm_shape == expected:

            M = sensitivity_matrix.T

        elif sm_shape == expected_T:

            M = sensitivity_matrix

        else:

            raise ValueError(

                f"sensitivity_matrix shape {sm_shape} does not match "

                f"(num_obs, num_ops)={expected} or "

                f"(num_ops, num_obs)={expected_T}"

            )

        self.register_buffer(

            "M",

            torch.as_tensor(

                M,

                dtype=torch.float32

            )

        )

        # ----------------------------------------------------

        # Reference chi2

        #

        # We normalize chi2 so the network does not receive

        # a potentially large raw number.

        # ----------------------------------------------------

        self.chi2_reference = float(

            max(abs(chi2_reference), 1.0) #FIXME

        )

        # ====================================================

        # STATE ENCODING

        # ====================================================

        # Operator identity

        self.op_embed = nn.Embedding(

            self.num_ops,

            embed_dim

        )

        # Observable pulls

        self.pull_proj = nn.Linear(

            1,

            embed_dim

        )

        # Active operator status:

        #

        #   0 = not active

        #   1 = active

        #

        self.active_embed = nn.Embedding(

            2,

            embed_dim

        )

        # Fired/evaluated status:

        #

        #   0 = not fired

        #   1 = fired

        #

        self.fired_embed = nn.Embedding(

            2,

            embed_dim

        )

        # Cross-attention between operators and observable pulls

        self.cross_attn = nn.MultiheadAttention(

            embed_dim=embed_dim,

            num_heads=n_heads,

            batch_first=True

        )

        # ----------------------------------------------------

        # Global state features

        #

        # chi2 + step

        # ----------------------------------------------------

        self.global_state_encoder = nn.Sequential(

            nn.Linear(

                2,

                embed_dim

            ),

            nn.Tanh(),

            nn.Linear(

                embed_dim,

                embed_dim

            ),

        )

        # ====================================================

        # ACTOR

        # ====================================================

        # Each operator receives:

        #

        #   operator identity

        #   pull/attention information

        #   active status

        #   fired status

        #   global state

        #

        # ALL of these come from the same s_t.

        self.actor_head = nn.Sequential(

            nn.Linear(

                embed_dim * 5,

                embed_dim

            ),

            nn.Tanh(),

            nn.Linear(

                embed_dim,

                1

            ),

        )

        # ====================================================

        # CRITIC

        # ====================================================

        # Critic receives a pooled representation of the SAME

        # operator-level state representation used by actor,

        # together with the global state.

        self.critic_head = nn.Sequential(

            nn.Linear(

                embed_dim * 6,

                embed_dim

            ),

            nn.ReLU(),

            nn.Linear(

                embed_dim,

                embed_dim // 2

            ),

            nn.ReLU(),

            nn.Linear(

                embed_dim // 2,

                1

            ),

        )

        # Trainable scaling parameters

        self.alpha = nn.Parameter(

            torch.tensor(

                alpha_init,

                dtype=torch.float32

            )

        )

        self.beta = nn.Parameter(

            torch.tensor(

                pull_gain,

                dtype=torch.float32

            )

        )

    # ========================================================

    # STATE ENCODER

    # ========================================================

    def encode_state(

        self,

        state,

        fired_ops=None,

    ):

        """

        Construct ONE explicit representation of s_t.

        This function is the central guarantee that actor and

        critic receive the SAME state.

        Returns

        -------

        operator_state : (num_ops, embed_dim * 5)

            Per-operator representation containing:

                operator identity

                pull/attention representation

                active status

                fired status

        global_state : (1, embed_dim)

            Global state containing:

                chi2

                step

        """

        device = self.M.device

        # ----------------------------------------------------

        # Pulls

        # ----------------------------------------------------

        pulls_data = (

            state.pulls

            if hasattr(state, "pulls")

            else state["pulls"]

        )

        if not isinstance(

            pulls_data,

            torch.Tensor

        ):

            pulls_tensor = torch.tensor(

                pulls_data,

                dtype=torch.float32,

                device=device

            )

        else:

            pulls_tensor = pulls_data.to(

                device=device,

                dtype=torch.float32

            )

        if pulls_tensor.ndim != 1:

            pulls_tensor = pulls_tensor.flatten()

        if pulls_tensor.numel() != self.num_obs:

            raise ValueError(

                f"Expected {self.num_obs} pulls, "

                f"got {pulls_tensor.numel()}"

            )

        # ----------------------------------------------------

        # Current chi2

        # ----------------------------------------------------

        chi2 = float(

            getattr(

                state,

                "chi2",

                state.get("chi2", 0.0)

                if isinstance(state, dict)

                else 0.0

            )

        )

        # q = min chi2 at fixed ||C|| can span many orders of magnitude.

        # Feed a bounded log-scale feature to the network.

        chi2_log10 = np.log10(max(chi2, FLAT_REWARD_EPS))

        chi2_normalized = float(np.clip(chi2_log10 / 12.0, -1.0, 1.0))

        # ----------------------------------------------------

        # Current step

        # ----------------------------------------------------

        step_value = float(

            getattr(

                state,

                "step",

                state.get("step", 0)

                if isinstance(state, dict)

                else 0

            )

        )

        step_fraction = (

            step_value

            / float(

                max(self.max_steps, 1)

            )

        )

        # ----------------------------------------------------

        # Operator identity

        # ----------------------------------------------------

        op_ids = torch.arange(

            self.num_ops,

            device=device

        )

        op_repr = self.op_embed(

            op_ids

        )

        # ----------------------------------------------------

        # Pull representation

        # ----------------------------------------------------

        pull_tokens = self.pull_proj(

            pulls_tensor.unsqueeze(-1)

        )

        # MultiheadAttention expects:

        #

        # query = operator tokens

        # key/value = observable pull tokens

        #

        # Shape:

        #

        #   (1, num_ops, embed_dim)

        #   (1, num_obs, embed_dim)

        # ----------------------------------------------------

        op_query = op_repr.unsqueeze(0)

        pull_tokens = pull_tokens.unsqueeze(0)

        attn_out, attn_weights = self.cross_attn(

            query=op_query,

            key=pull_tokens,

            value=pull_tokens

        )

        attn_out = attn_out.squeeze(0)

        # ----------------------------------------------------

        # ACTIVE OPERATOR MASK

        # ----------------------------------------------------

        active_mask = torch.zeros(

            self.num_ops,

            dtype=torch.long,

            device=device

        )

        active_ops = getattr(

            state,

            "active_ops",

            []

        )

        for op in active_ops:

            if op in self.op_to_idx:

                active_mask[

                    self.op_to_idx[op]

                ] = 1

        # ----------------------------------------------------

        # FIRED / EVALUATED MASK

        # ----------------------------------------------------

        fired_mask = torch.zeros(

            self.num_ops,

            dtype=torch.long,

            device=device

        )

        if fired_ops is not None:

            for idx in fired_ops:

                idx = int(idx)

                if 0 <= idx < self.num_ops:

                    fired_mask[idx] = 1

        # ----------------------------------------------------

        # Embed masks

        # ----------------------------------------------------

        active_repr = self.active_embed(

            active_mask

        )

        fired_repr = self.fired_embed(

            fired_mask

        )

        # ----------------------------------------------------

        # GLOBAL STATE

        #

        # EXACT SAME chi2 and step information is available

        # to both actor and critic.

        # ----------------------------------------------------

        global_scalar_state = torch.tensor(

            [

                chi2_normalized,

                step_fraction,

            ],

            dtype=torch.float32,

            device=device

        ).unsqueeze(0)

        global_repr = self.global_state_encoder(

            global_scalar_state

        )

        # ----------------------------------------------------

        # Broadcast global state to every operator

        # ----------------------------------------------------

        global_per_operator = (

            global_repr

            .expand(

                self.num_ops,

                -1

            )

        )

        # ----------------------------------------------------

        # ONE SHARED STATE REPRESENTATION

        # ----------------------------------------------------

        operator_state = torch.cat(

            [

                op_repr,

                attn_out,

                active_repr,

                fired_repr,

                global_per_operator,

            ],

            dim=-1

        )

        return (

            operator_state,

            global_repr,

            attn_weights,

            pulls_tensor,

            active_mask,

            fired_mask,

        )

    # ========================================================

    # FORWARD

    # ========================================================

    def forward(

        self,

        state,

        fired_ops=None,

    ):

        """

        Returns:

            logits

            V(s)

            attention

            diagnostics

        Both actor and critic are computed from the SAME

        state encoding produced by encode_state().

        """

        (

            operator_state,

            global_repr,

            attn_weights,

            pulls_tensor,

            active_mask,

            fired_mask,

        ) = self.encode_state(

            state,

            fired_ops=fired_ops

        )

        # ====================================================

        # ACTOR

        # ====================================================

        neural_logits = (

            self.actor_head(

                operator_state

            )

            .squeeze(-1)

            * self.alpha

        )

        # Optional direct physics term

        #

        # We keep this available but disabled, as in your

        # current implementation.

        pulls_column = pulls_tensor.unsqueeze(-1)

        direct_physics_logits = (

            torch.matmul(

                self.M,

                pulls_column

            )

            .squeeze(-1)

            * self.beta

        )

        # Current policy

        final_logits = neural_logits

        # ----------------------------------------------------

        # Mask operators that have already been fired

        # ----------------------------------------------------

        final_logits = final_logits.masked_fill(

            fired_mask.bool(),

            -1e9

        )

        # ====================================================

        # CRITIC

        # ====================================================

        # Pool the EXACT SAME operator_state that the actor

        # received.

        #

        # Therefore:

        #

        # actor = f(s)

        # critic = g(s)

        #

        # with the same s.

        # ----------------------------------------------------

        pooled_operator_state = (

            operator_state.mean(

                dim=0,

                keepdim=True

            )

        )

        critic_input = torch.cat(

            [

                pooled_operator_state,

                global_repr,

            ],

            dim=-1

        )

        # Defensive check: operator_state contains 5 * embed_dim features,

        # and global_repr contributes another embed_dim.

        expected_critic_dim = self.embed_dim * 6

        if critic_input.shape[-1] != expected_critic_dim:

            raise RuntimeError(

                f"Critic input dimension mismatch: got "

                f"{critic_input.shape[-1]}, expected {expected_critic_dim}. "

                f"pooled_operator_state={tuple(pooled_operator_state.shape)}, "

                f"global_repr={tuple(global_repr.shape)}"

            )

        state_value = self.critic_head(

            critic_input

        ).squeeze()

        diagnostics = {

            "base_logits":

                neural_logits.detach(),

            "direct_logits":

                direct_physics_logits.detach(),

            "pulls":

                pulls_tensor.detach()

                .cpu()

                .numpy(),

            "active_mask":

                active_mask.detach()

                .cpu()

                .numpy(),

            "fired_mask":

                fired_mask.detach()

                .cpu()

                .numpy(),

        }

        return (

            final_logits,

            state_value,

            attn_weights.squeeze(0),

            diagnostics,

        )



# ============================================================

# STEP-LEVEL REWARD



# ============================================================



def compute_step_rewards(

    history,

    sparsity_penalty=SPARSITY_PENALTY,

    rejected_penalty=REJECTED_STEP_PENALTY,

    eps=FLAT_REWARD_EPS,

):

    """Reward accepted flatness improvements; penalize rejected proposals."""

    rewards = []

    for item in history:

        if not bool(item.get("accepted", True)):

            rewards.append(float(-rejected_penalty))

            continue

        new_chi2 = max(float(item["trial_sm_chi2"]), 0.0)

        old_chi2 = item.get("old_sm_chi2")

        if old_chi2 is None:

            reward = -np.log10(new_chi2 + eps) - sparsity_penalty

        else:

            old_chi2 = max(float(old_chi2), 0.0)

            reward = (

                np.log10(old_chi2 + eps)

                - np.log10(new_chi2 + eps)

                - sparsity_penalty

            )

        rewards.append(float(reward))

    return rewards



# ============================================================

# RETURN-TO-GO



# ============================================================



def compute_returns_to_go(step_rewards, gamma=0.99):

    returns = [0.0 for _ in step_rewards]

    running = 0.0

    for t in reversed(range(len(step_rewards))):

        running = step_rewards[t] + gamma * running

        returns[t] = running

    return returns



# ============================================================

# TRAJECTORY REWARD



# ============================================================



def compute_rollout_reward(

    final_state,

    trajectory_length=None,

    sparsity_penalty=SPARSITY_PENALTY,

    eps=FLAT_REWARD_EPS,

    **_ignored,

):

    """Logging/best-trajectory score: -log10(q_final) - lambda*N_ops."""

    final_chi2 = max(float(final_state.chi2), 0.0)

    n_ops = int(trajectory_length or 0)

    reward = -np.log10(final_chi2 + eps) - sparsity_penalty * n_ops

    return float(reward), float(final_chi2)



# ============================================================

# RUNNING RETURN NORMALIZER



# ============================================================



class RunningReturnNormalizer:

    """

    Running normalization of Monte-Carlo returns.

    The critic therefore learns a normalized value:

        V_phi(s) ~ normalized G_t

    rather than a value on an ever-growing raw reward scale.

    """

    def __init__(

        self,

        momentum=0.01,

        eps=1e-6,

    ):

        self.momentum = momentum

        self.eps = eps

        self.mean = 0.0

        self.var = 1.0

        self.initialized = False

    def update(

        self,

        returns_t

    ):

        batch_mean = (

            returns_t

            .mean()

            .item()

        )

        batch_var = (

            returns_t

            .var(

                unbiased=False

            )

            .item()

        )

        if not self.initialized:

            self.mean = batch_mean

            self.var = max(

                batch_var,

                self.eps

            )

            self.initialized = True

        else:

            self.mean = (

                (1.0 - self.momentum)

                * self.mean

                + self.momentum

                * batch_mean

            )

            self.var = (

                (1.0 - self.momentum)

                * self.var

                + self.momentum

                * batch_var

            )

            self.var = max(

                self.var,

                self.eps

            )

    def normalize(

        self,

        returns_t

    ):

        std = (

            self.var ** 0.5

        ) + self.eps

        return (

            returns_t

            - self.mean

        ) / std

    def update_and_normalize(

        self,

        returns_t

    ):

        self.update(

            returns_t

        )

        return self.normalize(

            returns_t

        )



# ============================================================

# ROLLOUT



# ============================================================



def rollout(

    policy,

    pull_fn,

    obs_names,

    baseline_pulls,

    baseline_chi2,

    max_steps=10,

    deterministic=False,

    verbose=True,

    trajectory_idx=1,

    total_trajectories=30,

):

    # Every proposal is evaluated once. The first operator establishes the
    # one-operator reference. Later operators are accepted only when they
    # improve fixed-norm SM-like chi2 by MIN_LOG10_CHI2_IMPROVEMENT.

    current_state = SMEFTState(

        pulls=np.asarray(baseline_pulls, dtype=np.float32),

        chi2=float(baseline_chi2),

        obs_names=obs_names,

    )

    current_state.active_ops = []

    current_state.wc_values = {}

    current_state.step = 0

    all_evaluated_op_indices = set()

    accepted_op_indices = []

    log_probs = []

    entropies = []

    attentions = []

    diagnostics = []

    state_values = []

    state_trajectory = [current_state.copy()]

    history = []

    if verbose:

        print(

            f"\n--- Trajectory [{trajectory_idx:02d} / "

            f"{total_trajectories:02d}] ---"

        )

    for step_idx in range(max_steps):

        state_before = current_state.copy()

        logits, state_val, attn_weights, diag = policy(

            current_state,

            fired_ops=all_evaluated_op_indices,

        )

        dist = torch.distributions.Categorical(logits=logits)

        action = torch.argmax(logits) if deterministic else dist.sample()

        action_idx = int(action.item())

        log_probs.append(dist.log_prob(action))

        entropies.append(dist.entropy())

        attentions.append(attn_weights.detach())

        diagnostics.append(diag)

        state_values.append(state_val)

        all_evaluated_op_indices.add(action_idx)

        proposed_op = policy.op_vocab[action_idx]

        trial_ops = list(current_state.active_ops) + [proposed_op]

        old_sm_chi2 = None if len(current_state.active_ops) == 0 else float(current_state.chi2)

        _, trial_pulls, trial_chi2, trial_predictions, trial_wc = pull_fn(trial_ops)

        trial_chi2 = float(trial_chi2)

        if old_sm_chi2 is None:

            log_improvement = None

            accepted = True

        else:

            log_improvement = float(

                np.log10(max(old_sm_chi2, 0.0) + FLAT_REWARD_EPS)

                - np.log10(max(trial_chi2, 0.0) + FLAT_REWARD_EPS)

            )

            accepted = bool(

                np.isfinite(log_improvement)

                and log_improvement >= MIN_LOG10_CHI2_IMPROVEMENT

            )

        if accepted:

            accepted_op_indices.append(action_idx)

            current_state = SMEFTState(

                pulls=np.asarray(trial_pulls, dtype=np.float32),

                chi2=trial_chi2,

                obs_names=obs_names,

            )

            current_state.active_ops = list(trial_ops)

            current_state.wc_values = dict(trial_wc)

            current_state.step = step_idx + 1

        else:

            current_state = state_before.copy()

            current_state.step = step_idx + 1

        state_trajectory.append(current_state.copy())

        if verbose:

            improvement_text = "first-op" if log_improvement is None else f"dlog10chi2={log_improvement:+8.4f}"

            status_text = "ACCEPT" if accepted else "REJECT"

            print(

                f"STEP {step_idx:02d}: {status_text:6s} {proposed_op:20s} "

                f"trial_chi2={trial_chi2:12.6g} "

                f"{improvement_text} "

                f"WC={{{', '.join(f'{k}:{v:+.3g}' for k, v in trial_wc.items())}}}"

            )

        history.append({

            "step": step_idx,

            "proposed_operator": proposed_op,

            "accepted": bool(accepted),

            "acceptance_threshold_log10": float(MIN_LOG10_CHI2_IMPROVEMENT),

            "state_before": state_before.to_dict(),

            "state_after": current_state.to_dict(),

            "old_sm_chi2": old_sm_chi2,

            "trial_sm_chi2": trial_chi2,

            "log10_chi2_improvement": log_improvement,

            "trial_wc": {k: float(v) for k, v in trial_wc.items()},

            "trial_wc_norm_sq": float(sum(float(v) ** 2 for v in trial_wc.values())),

            "trial_predictions": [float(x) for x in trial_predictions],

        })

    if verbose:

        final_ops_str = " -> ".join(current_state.active_ops) if current_state.active_ops else "None"

        print(f"FINAL ACCEPTED SEQUENCE [{len(current_state.active_ops)} ops]: {final_ops_str}")

        print(

            f"FINAL RESULTS: min chi2 at ||C||={FIXED_WC_NORM:g} = "

            f"{current_state.chi2:.8g} | WC = {current_state.wc_values}"

        )

    return {

        "final_state": current_state,

        "state_trajectory": state_trajectory,

        "ops": accepted_op_indices,

        "evaluated_ops": sorted(all_evaluated_op_indices),

        "op_names": current_state.active_ops,

        "log_probs": torch.stack(log_probs),

        "entropies": torch.stack(entropies),

        "state_values": torch.stack(state_values),

        "attention": attentions,

        "diagnostics": diagnostics,

        "history": history,

    }



# ============================================================

# TRAINING PARAMETERS



# ============================================================

MAX_STEPS = 3  # start small for validation; increase after SVD checks

ROLLOUTS_PER_BATCH = 25

N_BATCHES = 300

# First operator is accepted; later operators must pass the minimum log-chi2 improvement threshold.

GRAD_CLIP = 0.5

GAMMA = 0.99

CRITIC_LOSS_COEF = 0.5

ADV_NORMALIZE = True

RETURN_NORM_MOMENTUM = 0.01



# ============================================================

# MODEL INITIALIZATION



# ============================================================

M_tensor = torch.tensor(

    M_ij,

    dtype=torch.float32

)

M_mean = M_tensor.mean()

M_std = (

    M_tensor.std()

    + 1e-6

)

M_normalized = (

    M_tensor - M_mean

) / M_std

model = SMEFTActorCritic(

    op_vocab=operator_vocab,

    obs_names=OBS_NAMES,

    sensitivity_matrix=

        M_normalized,

    chi2_reference=

        1.0,

    max_steps=

        MAX_STEPS,

    embed_dim=32,

    n_heads=4,

    alpha_init=1.0,

    pull_gain=2.5,

).to(DEVICE)

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-3,

    weight_decay=1e-4,

)

return_normalizer = (

    RunningReturnNormalizer(

        momentum=

            RETURN_NORM_MOMENTUM

    )

)



# ============================================================

# OUTPUT



# ============================================================

OUTPUT_DIR = (

    "/home/kumarj/links/scratch/blind/"

)

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)

METRICS_LOG_PATH = os.path.join(

    OUTPUT_DIR,

    "metrics_log.txt"

)

THEORY_ARCHIVE_PATH = os.path.join(

    OUTPUT_DIR,

    "theory_archive.json"

)

REWARD_DISTS_PATH = os.path.join(

    OUTPUT_DIR,

    "reward_distributions.json"

)

TOP_COMBINATIONS_PATH = os.path.join(

    OUTPUT_DIR,

    "top_100_operator_combinations.txt"

)

TOP_COMBINATIONS_JSON_PATH = os.path.join(

    OUTPUT_DIR,

    "top_100_operator_combinations.json"

)



# ============================================================

# METRICS



# ============================================================

with open(

    METRICS_LOG_PATH,

    "w"

) as f:

    f.write(

        "batch\t"

        "mean_traj_reward\t"

        "max_traj_reward\t"

        "mean_flat_chi2\t"

        "actor_loss\t"

        "critic_loss\t"

        "entropy_loss\t"

        "total_loss\t"

        "alpha\t"

        "beta\t"

        "mean_advantage\t"

        "raw_advantage_mean\t"

        "raw_advantage_std\t"

        "mean_abs_raw_advantage\t"

        "max_abs_raw_advantage\t"

        "explained_variance\t"

        "return_mean\t"

        "return_std\t"

        "grad_norm\n"

    )

if not os.path.exists(

    THEORY_ARCHIVE_PATH

):

    with open(

        THEORY_ARCHIVE_PATH,

        "w"

    ) as f:

        json.dump(

            [],

            f,

            indent=2

        )

if not os.path.exists(

    REWARD_DISTS_PATH

):

    with open(

        REWARD_DISTS_PATH,

        "w"

    ) as f:

        json.dump(

            {},

            f,

            indent=2

        )

theory_archive = []

reward_distributions = {}

best_reward = -float("inf")

best_ops = None



# Best occurrence of each UNIQUE operator combination.

# The key is sorted, so different selection orders count as the same combination.

best_combinations = {}

TOP_N_COMBINATIONS = 100



def update_best_combination(

    operator_names,

    traj_reward,

    flat_chi2,

    wc_values,

    batch_idx,

    rollout_idx,

):

    key = tuple(sorted(operator_names))

    if not key:

        return



    record = {

        "operators": list(key),

        "n_operators": len(key),

        "traj_reward": float(traj_reward),

        "sm_like_chi2": float(flat_chi2),

        "fixed_norm": float(FIXED_WC_NORM),

        "wc_values": {k: float(v) for k, v in wc_values.items()},

        "batch": int(batch_idx),

        "rollout": int(rollout_idx),

    }



    previous = best_combinations.get(key)

    if previous is None or record["traj_reward"] > previous["traj_reward"]:

        best_combinations[key] = record





def get_top_combinations(n=TOP_N_COMBINATIONS):

    records = list(best_combinations.values())

    records.sort(

        key=lambda r: (

            -r["traj_reward"],

            r["sm_like_chi2"],

            r["n_operators"],

            tuple(r["operators"]),

        )

    )

    return records[:n]





def write_top_combinations():

    top_records = get_top_combinations()



    with open(TOP_COMBINATIONS_JSON_PATH, "w") as f:

        json.dump(top_records, f, indent=2)



    with open(TOP_COMBINATIONS_PATH, "w") as f:

        f.write(

            "rank\ttraj_reward\tsm_like_chi2\tn_operators\t"

            "operators\twc_values\tbatch\trollout\n"

        )

        for rank, record in enumerate(top_records, start=1):

            operators_str = ", ".join(record["operators"])

            wc_str = ", ".join(

                f"{op}={record['wc_values'].get(op, 0.0):+.8g}"

                for op in record["operators"]

            )

            f.write(

                f"{rank}\t"

                f"{record['traj_reward']:.10g}\t"

                f"{record['sm_like_chi2']:.10g}\t"

                f"{record['n_operators']}\t"

                f"{operators_str}\t"

                f"{wc_str}\t"

                f"{record['batch']}\t"

                f"{record['rollout']}\n"

            )



# ============================================================

# TRAINING LOOP



# ============================================================

for batch_idx in range(N_BATCHES):

    model.train()

    progress = (batch_idx / float(N_BATCHES))

    entropy_coef = (0.01 * (1.0 - progress) + 0.0001 * progress)

    batch_traj_rewards = []

    batch_flat_chi2 = []

    batch_ops = []

    # --------------------------------------------------------

    # Step-level quantities

    # --------------------------------------------------------

    all_log_probs = []

    all_entropies = []

    all_state_values = []

    all_returns = []

    # --------------------------------------------------------

    # Diagnostics

    # --------------------------------------------------------

    batch_attn_correlations = []

    batch_base_norms = []

    batch_direct_norms = []

    print(f"\n==================== "

          f"BATCH {batch_idx:03d} "

          f"====================")

    # ========================================================

    # ROLLOUTS

    # ========================================================

    for rollout_idx in range(

        ROLLOUTS_PER_BATCH

    ):

        result = rollout(

            policy=model,

            pull_fn=flat_direction_fn,

            obs_names=OBS_NAMES,

            baseline_pulls=

                FLAT_BASELINE_PULLS,

            baseline_chi2=

                0.0,

            max_steps=

                MAX_STEPS,

            deterministic=False,

            verbose=True,

            trajectory_idx=

                rollout_idx + 1,

            total_trajectories=

                ROLLOUTS_PER_BATCH,

        )

        final_state = result["final_state"]

        hist = result["history"]

        traj_ops = result["ops"]

        traj_len = len(traj_ops)

        # ====================================================

        # TRAJECTORY REWARD

        # ====================================================

        (traj_reward,final_flat_chi2) = compute_rollout_reward(

            final_state=final_state,

            trajectory_length=traj_len,

            sparsity_penalty=SPARSITY_PENALTY,

        )

        # ====================================================

        # STEP REWARDS

        # ====================================================

        step_rewards = (

            compute_step_rewards(

                hist,

                sparsity_penalty=SPARSITY_PENALTY,

                ))

        # ====================================================

        # RETURN-TO-GO

        # ====================================================

        returns_to_go = (

            compute_returns_to_go(

                step_rewards,

                gamma=GAMMA))

        returns_t = torch.tensor(

            returns_to_go,

            dtype=torch.float32,

            device=DEVICE)

        # ====================================================

        # STORE ACTOR / CRITIC DATA

        # ====================================================

        all_log_probs.append(result["log_probs"])

        all_entropies.append(result["entropies"])

        all_state_values.append(result["state_values"])

        all_returns.append(returns_t)

        # ====================================================

        # ARCHIVE

        # ====================================================

        """

        theory_archive.append({"batch":

                batch_idx,

            "rollout": rollout_idx,

            "traj_reward": float(traj_reward),

            "step_rewards": [float(r) for r in step_rewards],

            "returns_to_go":[float(r) for r in returns_to_go],

            "fixed_norm": float(FIXED_WC_NORM),

            "sm_like_chi2": float(final_state.chi2),

            "wc_values": {k: float(v) for k, v in getattr(final_state, "wc_values", {}).items()},

            "operators": result["op_names"],

            "pulls": np.asarray(final_state.pulls).tolist(),

            "history":hist,})

        with open(THEORY_ARCHIVE_PATH, "w") as f:

            json.dump(theory_archive, f, indent=2)

        """

        # ====================================================

        # BATCH LOGGING

        # ====================================================

        batch_traj_rewards.append(float(traj_reward))

        batch_flat_chi2.append(float(final_flat_chi2))

        batch_ops.append(traj_ops)

        theory_archive.append({

            "batch": batch_idx,

            "rollout": rollout_idx,

            "traj_reward": float(traj_reward),

            "step_rewards": [float(r) for r in step_rewards],

            "returns_to_go": [float(r) for r in returns_to_go],

            "fixed_norm": float(FIXED_WC_NORM),

            "sm_like_chi2": float(final_state.chi2),

            "wc_values": {

                k: float(v) for k, v in getattr(final_state, "wc_values", {}).items()

            },

            "operators": result["op_names"],

            "pulls": np.asarray(final_state.pulls).tolist(),

            "history": hist,

        })

        update_best_combination(

            operator_names=result["op_names"],

            traj_reward=traj_reward,

            flat_chi2=final_state.chi2,

            wc_values=getattr(final_state, "wc_values", {}),

            batch_idx=batch_idx,

            rollout_idx=rollout_idx,

        )

        """

        # ====================================================

        # ATTENTION DIAGNOSTICS

        # ====================================================

        for (attn_matrix,diag) in zip(result["attention"],result["diagnostics"]):

            abs_pulls = np.abs(diag["pulls"])

            if np.sum(abs_pulls) > 0:

                step_corrs = []

                attn_np = (attn_matrix.cpu().numpy())

                for op_idx in range(attn_np.shape[0]):

                    row_attn = (attn_np[op_idx])

                    corr, _ = (scipy.stats.spearmanr(row_attn,abs_pulls))

                    if not np.isnan(corr):

                        step_corrs.append(corr)

                if step_corrs:

                    batch_attn_correlations.append(np.mean(step_corrs))

            batch_base_norms.append(torch.norm(diag["base_logits"]).cpu().item())

            batch_direct_norms.append(torch.norm(diag["direct_logits"]).cpu().item())

        """

        # ====================================================

        # BEST TRAJECTORY

        # ====================================================

        if (traj_reward > best_reward):

            best_reward = (traj_reward)

            best_ops = list(traj_ops)

    # ========================================================

    # REWARD DISTRIBUTION

    # ========================================================

    reward_distributions[f"batch_{batch_idx}"] = batch_traj_rewards

    # ========================================================

    # FLATTEN ALL STEPS

    # ========================================================

    flat_log_probs = torch.cat(all_log_probs)

    flat_entropies = torch.cat(all_entropies)

    flat_values = torch.cat(all_state_values)

    flat_returns = torch.cat(all_returns)

    # ========================================================

    # NORMALIZED RETURNS

    # ========================================================

    flat_returns_norm = (return_normalizer.update_and_normalize(flat_returns))

    # ========================================================

    # RAW ADVANTAGE

    #

    # G_t is the Monte-Carlo Q estimate.

    #

    # V(s_t) is the critic.

    #

    # Therefore:

    #

    # A_t = G_t - V(s_t)

    #

    # ========================================================

    raw_advantages = (flat_returns_norm- flat_values.detach())

    raw_advantage_mean = (raw_advantages.mean())

    raw_advantage_std = (raw_advantages.std())

    mean_abs_raw_advantage = (raw_advantages.abs().mean())

    max_abs_raw_advantage = (raw_advantages.abs().max())

    # ========================================================

    # EXPLAINED VARIANCE

    # ========================================================

    target_var = (flat_returns_norm.detach().var())

    if target_var.item() > 1e-8:

        explained_variance = (

            1.0

            - (

                flat_returns_norm.detach()

                - flat_values.detach()

            )

            .var()

            / (

                target_var

                + 1e-8

            )

        )

    else:

        explained_variance = torch.tensor(

            0.0,

            device=DEVICE

        )

    # ========================================================

    # ADVANTAGE NORMALIZATION

    # ========================================================

    advantages = raw_advantages

    if (

        ADV_NORMALIZE

        and advantages.numel() > 1

        and advantages.std().item() > 1e-6

    ):

        advantages = (

            (

                advantages

                - advantages.mean()

            )

            / (

                advantages.std()

                + 1e-8

            )

        )

    # ========================================================

    # ACTOR LOSS

    # ========================================================

    actor_loss = -(

        advantages.detach()

        * flat_log_probs

    ).mean()

    # ========================================================

    # CRITIC LOSS

    # ========================================================

    critic_loss = F.mse_loss(

        flat_values,

        flat_returns_norm.detach()

    )

    # ========================================================

    # ENTROPY

    # ========================================================

    entropy_loss = (

        -entropy_coef

        * flat_entropies.mean()

    )

    # ========================================================

    # TOTAL LOSS

    # ========================================================

    total_loss = (

        actor_loss

        + CRITIC_LOSS_COEF

        * critic_loss

        + entropy_loss

    )

    # ========================================================

    # BACKPROP

    # ========================================================

    optimizer.zero_grad(

        set_to_none=True

    )

    total_loss.backward()

    # ========================================================

    # GRADIENT CLIPPING

    # ========================================================

    raw_grad_norm = (

        torch.nn.utils

        .clip_grad_norm_(

            model.parameters(),

            GRAD_CLIP

        )

        .item()

    )

    optimizer.step()

    # ========================================================

    # BATCH METRICS

    # ========================================================

    batch_mean_traj_reward = float(

        np.mean(

            batch_traj_rewards

        )

    )

    batch_max_traj_reward = float(

        np.max(

            batch_traj_rewards

        )

    )

    batch_mean_flat_chi2 = float(

        np.mean(

            batch_flat_chi2

        )

    )

    return_std = (

        return_normalizer.var

        ** 0.5

    )

    # ========================================================

    # SAVE METRICS

    # ========================================================

    with open(

        METRICS_LOG_PATH,

        "a"

    ) as f:

        f.write(

            f"{batch_idx}\t"

            f"{batch_mean_traj_reward:.6f}\t"

            f"{batch_max_traj_reward:.6f}\t"

            f"{batch_mean_flat_chi2:.6f}\t"

            f"{actor_loss.item():.6f}\t"

            f"{critic_loss.item():.6f}\t"

            f"{entropy_loss.item():.6f}\t"

            f"{total_loss.item():.6f}\t"

            f"{model.alpha.item():.6f}\t"

            f"{model.beta.item():.6f}\t"

            f"{advantages.mean().item():.6f}\t"

            f"{raw_advantage_mean.item():.6f}\t"

            f"{raw_advantage_std.item():.6f}\t"

            f"{mean_abs_raw_advantage.item():.6f}\t"

            f"{max_abs_raw_advantage.item():.6f}\t"

            f"{explained_variance.item():.6f}\t"

            f"{return_normalizer.mean:.6f}\t"

            f"{return_std:.6f}\t"

            f"{raw_grad_norm:.6f}\n"

        )

    # ========================================================

    # PRINT SUMMARY

    # ========================================================

    print(

        f"\n>>> SUMMARY "

        f"BATCH {batch_idx:03d} <<<"

    )

    print(

        f"Actor: "

        f"{actor_loss.item():.4f} | "

        f"Critic: "

        f"{critic_loss.item():.4f} | "

        f"Entropy: "

        f"{entropy_loss.item():.4f} | "

        f"Mean fixed-norm chi2: "

        f"{batch_mean_flat_chi2:.2f}"

    )

    print(

        f"Return mean: "

        f"{return_normalizer.mean:.4f} | "

        f"Return std: "

        f"{return_std:.4f} | "

        f"Mean |Raw Adv|: "

        f"{mean_abs_raw_advantage.item():.4f} | "

        f"Raw Adv Std: "

        f"{raw_advantage_std.item():.4f} | "

        f"Explained Var: "

        f"{explained_variance.item():.4f}"

    )

    print(

        f"Gradient norm: "

        f"{raw_grad_norm:.4f}"

    )



    # Keep the top-100 files current even if a long training run stops early.

    write_top_combinations()



with open(THEORY_ARCHIVE_PATH, "w") as f:

    json.dump(theory_archive, f, indent=2)



with open(REWARD_DISTS_PATH, "w") as f:

    json.dump(reward_distributions, f, indent=2)



# Final top-100 write and concise console summary.

write_top_combinations()

top_records = get_top_combinations()

print("\n================ TOP OPERATOR COMBINATIONS ================")

for rank, record in enumerate(top_records, start=1):



    wc_values = record["wc_values"]



    ops_with_wc = " + ".join(

        f"{op}({float(wc_values[op]):+.6f})"

        for op in record["operators"]

    )



    print(

        f"{rank:3d}. reward={record['traj_reward']:.6f} | "

        f"chi2_SM={record['sm_like_chi2']:.6e} | "

        f"ops={ops_with_wc}"

    )

print(f"\nSaved top combinations to: {TOP_COMBINATIONS_PATH}")

print(f"Saved JSON version to: {TOP_COMBINATIONS_JSON_PATH}")
