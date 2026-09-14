from mpi4py import MPI

import flavio
import numdifftools as nd
import numpy as np
import json

from pathlib import Path
from wilson import Wilson

import argparse
import warnings
import wilson
from wilson import wcxf


# ============================================================
# MPI
# ============================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#OUTPUT_FILE = "/home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly/smeft_new/ewp_linear_coefficients.json"

OUTPUT_FILE="/home/kumarj/links/scratch/tmp_real/ewp_linear_coefficients_2.json"

MZ = 1000#91.1876

EFT = "SMEFT"
BASIS = "Warsaw"

# We calculate derivatives with respect to:
#
#     x_i = 1e6 * C_i
#
# where C_i is the Wilson coefficient.
#
NORMALIZE = 1e6

# ============================================================
# EWP OBSERVABLES
# ============================================================

constraints_ewp = [
    "GammaZ",
    "sigma_had",
    "R_e",
    "R_mu",
    "R_tau",

    "AFB(Z->ee)",
    "AFB(Z->mumu)",
    "AFB(Z->tautau)",

    "A(Z->ee)",
    "A(Z->mumu)",
    "A(Z->tautau)",

    "R_b",
    "R_c",

    "AFB(Z->cc)",

    "A(Z->bb)",
    "A(Z->cc)",

    "GammaW",

    "BR(W->enu)",
    "BR(W->munu)",
    "BR(W->taunu)",

    "R(W->cX)",
    "Rmue(W->lnu)",
    "Rtaue(W->lnu)",
    "Rtaumu(W->lnu)",

    "A(Z->ss)",

    "m_W",
    "AFB(Z->bb)",
    "BR(B+->Knunu)",
    "Rtaul(B->Dlnu)",
    "epsp/eps",
    ("<P5p>(B0->K*mumu)",4,6),
    ("<dBR/dq2>(Bs->phimumu)",1,6),
    "BR(K+->pinunu)",
    "BR(KL->pinunu)",
    "DeltaM_s",
    "eps_K",
    "BR(Bs->mumu)",
]


# ============================================================
# LOAD OPERATOR VOCABULARY
# ============================================================
        
warnings.filterwarnings("ignore")

def _build_catalogue_from_wcxf() -> dict[str, list[str]]:
    """
    Query the wcxf package for every WC in the SMEFT Warsaw basis and
    organise them by their wcxf 'sector' label.

    Returns
    -------
    dict  {sector_name: [wc_name, ...]}  — ordered by sector, then WC name
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
operator_vocab: list[str] = [wc for ops in OPERATOR_CATALOGUE.values() for wc in ops]
print(f"Loaded {len(operator_vocab)} operators")
print(operator_vocab[:100])

# ============================================================
# PRINT BASIC INFORMATION
# ============================================================

if rank == 0:

    print("=" * 70)
    print("Generating SMEFT EWP linear coefficients")
    print("=" * 70)

    print(f"MPI ranks       : {size}")
    print(f"Operators       : {len(operator_vocab)}")
    print(f"Observables     : {len(constraints_ewp)}")
    print(f"Scale           : {MZ} GeV")
    print(f"EFT             : {EFT}")
    print(f"Basis           : {BASIS}")
    print(f"Normalization   : {NORMALIZE}")
    print()


# ============================================================
# LINEARIZATION FUNCTION
# ============================================================

def get_linear(
    wc_list,
    scale,
    eft,
    basis,
    obs,
    normalize=1e6,
):
    """
    Calculate

        d(O/O_SM) / d(normalize * C_i)

    at C_i = 0.

    Therefore the resulting coefficients satisfy

        O/O_SM
        =
        1 + sum_i coeff_i * (normalize * C_i)

    """

    # --------------------------------------------------------
    # Function to differentiate
    # --------------------------------------------------------

    def f(x):

        wc_dict = {
            wc: x[i] / normalize
            for i, wc in enumerate(wc_list)
        }

        w = Wilson(
            wc_dict,
            scale,
            eft,
            basis,
        )

        prediction = flavio.np_prediction(
            obs,
            w,
        )

        sm = flavio.sm_prediction(
            obs,
        )

        return prediction / sm

    # --------------------------------------------------------
    # Numerical gradient
    # --------------------------------------------------------

    gradient = nd.Gradient(f)

    x0 = np.zeros(len(wc_list))

    result = gradient(x0)

    return np.asarray(result, dtype=float)


# ============================================================
# CALCULATE ONE OBSERVABLE
# ============================================================

def calculate_observable(obs):

    print(
        f"[Rank {rank}] Starting {obs}",
        flush=True,
    )

    # --------------------------------------------------------
    # SM prediction
    # --------------------------------------------------------

    sm_prediction = flavio.sm_prediction(obs)
    sm_unc = flavio.sm_uncertainty(obs)

    # --------------------------------------------------------
    # Linear coefficients
    # --------------------------------------------------------

    coefficients = get_linear(
        operator_vocab,
        MZ,
        EFT,
        BASIS,
        obs,
        normalize=NORMALIZE,
    )

    # --------------------------------------------------------
    # Convert numpy array -> dictionary
    # --------------------------------------------------------

    coefficient_dict = {}

    for operator, coefficient in zip(
        operator_vocab,
        coefficients,
    ):

        coefficient = float(coefficient)

        # Ignore numerical noise
        if abs(coefficient) > 1e-12:

            coefficient_dict[operator] = coefficient

    print(
        f"[Rank {rank}] Finished {obs} "
        f"({len(coefficient_dict)} non-zero coefficients)",
        flush=True,
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "sm_prediction": float(sm_prediction),
        "sm_uncertainty": float(sm_unc),

        "coefficients": coefficient_dict,
    }


# ============================================================
# DISTRIBUTE OBSERVABLES AMONG MPI RANKS
# ============================================================

my_observables = constraints_ewp[rank::size]


print(
    f"[Rank {rank}] "
    f"Assigned observables: {my_observables}",
    flush=True,
)


# ============================================================
# CALCULATE LOCAL RESULTS
# ============================================================

local_results = {}


for obs in my_observables:

    try:

        local_results[obs] = calculate_observable(obs)

    except Exception as e:

        print(
            f"[Rank {rank}] ERROR calculating {obs}: {e}",
            flush=True,
        )

        local_results[obs] = {
            "error": str(e)
        }


# ============================================================
# GATHER ALL RESULTS
# ============================================================

all_results = comm.gather(
    local_results,
    root=0,
)


# ============================================================
# ROOT: COMBINE AND SAVE JSON
# ============================================================

if rank == 0:

    final_results = {}

    for result in all_results:

        final_results.update(result)

    # --------------------------------------------------------
    # Keep observables in the original order
    # --------------------------------------------------------

    final_results = {
        obs: final_results[obs]
        for obs in constraints_ewp
        if obs in final_results
    }

    # --------------------------------------------------------
    # Complete JSON structure
    # --------------------------------------------------------

    output = {

        "metadata": {

            "scale": MZ,

            "eft": EFT,

            "basis": BASIS,

            "normalize": NORMALIZE,

#            "operator_file": str(),

            "operators": operator_vocab,

            "observables": constraints_ewp,

            "formula": (
                "O/O_SM = 1 + "
                "sum_i coefficient_i * "
                "(NORMALIZE * C_i)"
            ),
        },

        "observables": final_results,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Saved coefficients to: {OUTPUT_FILE}"
    )

    print(
        f"Observables calculated: "
        f"{len(final_results)}"
    )

    print()


    
