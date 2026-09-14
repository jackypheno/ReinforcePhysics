import numpy as np
import torch


def print_policy_diagnostics(
    policy,
    state,
    logits,
    attn_weights,
    top_k=10,
):
    """
    Print:
      - current pulls
      - attention matrix
      - operator logits/probabilities
      - operators ranked by probability
    """

    # ---------------------------------------------------------
    # Convert tensors to NumPy
    # ---------------------------------------------------------

    pulls = np.asarray(
        state.pulls,
        dtype=float,
    )

    logits_np = (
        logits.detach()
        .cpu()
        .numpy()
    )

    probs_np = (
        torch.softmax(
            logits,
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    attn_np = (
        attn_weights.detach()
        .cpu()
        .numpy()
    )

    # ---------------------------------------------------------
    # Current state
    # ---------------------------------------------------------

    print("\n" + "=" * 120)

    print(
        f"Current chi2: {state.chi2:.6f}"
    )

    print(
        "Fired operators:",
        state.fired_ops,
    )

    print("\nObservable pulls:")

    for obs, pull in zip(
        policy.obs_names,
        pulls,
    ):
        print(
            f"  {obs:<25s}"
            f" pull = {pull:+.5f}"
        )

    # ---------------------------------------------------------
    # Attention matrix
    # ---------------------------------------------------------

    print(
        "\nAttention matrix"
    )

    print(
        "Rows    = candidate operators"
    )

    print(
        "Columns = observables"
    )

    header = (
        f"{'Operator':<22s}"
    )

    for obs in policy.obs_names:
        header += (
            f"{obs[:14]:>16s}"
        )

    print(header)

    print(
        "-" * len(header)
    )

    fired_set = set(
        state.fired_ops
    )

    for i, op in enumerate(
        policy.op_vocab
    ):

        status = (
            "[FIRED]"
            if op in fired_set
            else ""
        )

        row = (
            f"{op:<18s}"
            f"{status:<4s}"
        )

        for weight in attn_np[i]:
            row += (
                f"{weight:16.4f}"
            )

        print(row)

    # ---------------------------------------------------------
    # Operator ranking
    # ---------------------------------------------------------

    print(
        "\nOperator ranking"
    )

    ranking = np.argsort(
        probs_np
    )[::-1]

    print(
        f"{'Rank':>4s} "
        f"{'Operator':<25s} "
        f"{'Logit':>12s} "
        f"{'Probability':>14s} "
        f"{'Most-attended observable':<30s} "
        f"{'Attention':>10s}"
    )

    print(
        "-" * 110
    )

    shown = 0

    for rank, i in enumerate(
        ranking,
        start=1,
    ):

        # Skip already fired operators
        if not np.isfinite(
            logits_np[i]
        ):
            continue

        most_attended_idx = int(
            np.argmax(
                attn_np[i]
            )
        )

        most_attended_obs = (
            policy.obs_names[
                most_attended_idx
            ]
        )

        max_attention = (
            attn_np[
                i,
                most_attended_idx
            ]
        )

        print(
            f"{rank:4d} "
            f"{policy.op_vocab[i]:<25s} "
            f"{logits_np[i]:12.5f} "
            f"{probs_np[i]:14.6f} "
            f"{most_attended_obs:<30s} "
            f"{max_attention:10.4f}"
        )

        shown += 1

        if shown >= top_k:
            break

    print(
        "=" * 120
    )
