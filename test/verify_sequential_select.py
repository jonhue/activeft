import sys
from typing import Tuple

sys.path.insert(0, "/users/jhbotter/ttt-reasoning/activeft")

import torch

from activeft.acquisition_functions import SequentialAcquisitionFunction


class ToyDataset:
    def __init__(self, n: int, d: int):
        torch.manual_seed(0)
        self.X = torch.randn(n, d)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.X[i]


class DummySequential(SequentialAcquisitionFunction[None, dict]):
    def initialize(self, model: None, data: torch.Tensor, device: torch.device | None) -> dict:
        return {"data": data.clone()}

    def compute(self, state: dict) -> torch.Tensor:
        return state["data"].sum(dim=1)

    def step(self, state: dict, i: int) -> dict:
        state["data"][i] = -1e9
        return state


def run_case(
    name: str,
    n: int,
    d: int,
    mini_batch_size: int,
    batch_size: int,
    *,
    subsample: bool = False,
    force_nonsequential: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, ToyDataset]:
    af = DummySequential(mini_batch_size=mini_batch_size, subsample=subsample, force_nonsequential=force_nonsequential)
    ds = ToyDataset(n, d)
    idx, vals = af.select(batch_size, None, ds, None)

    # Basic validations
    assert len(idx) == batch_size, f"{name}: expected {batch_size} selections, got {len(idx)}"
    assert int(idx.min()) >= 0 and int(idx.max()) < n, f"{name}: indices out of bounds"
    unique = len(set(idx.tolist()))
    assert unique == len(idx), f"{name}: duplicate indices detected ({unique} unique out of {len(idx)})"

    # Values should match the compute() outputs for the selected indices
    expected_vals = ds.X[idx].sum(dim=1)
    assert torch.allclose(vals, expected_vals), f"{name}: values mismatch"

    print(f"PASS: {name} -> len={len(idx)}, min={int(idx.min())}, max={int(idx.max())}")
    return idx, vals, ds


def main() -> None:
    # Case 1: Standard hierarchical reduction
    run_case(
        name="hierarchical_basic",
        n=63,
        d=3,
        mini_batch_size=10,
        batch_size=4,
        subsample=False,
        force_nonsequential=False,
    )

    # Case 2: Final batch smaller than batch_size encountered in some rounds
    run_case(
        name="mini_batch_tail_smaller_than_batch",
        n=25,
        d=3,
        mini_batch_size=10,
        batch_size=5,
        subsample=False,
        force_nonsequential=False,
    )

    # Case 3: Force non-sequential path (computes values once per overall pass)
    run_case(
        name="force_nonsequential_full_pass",
        n=80,
        d=4,
        mini_batch_size=16,
        batch_size=8,
        subsample=False,
        force_nonsequential=True,
    )

    # Case 4: Subsample=True path (single mini-batch per round)
    run_case(
        name="subsample_true_single_batch_per_round",
        n=57,
        d=3,
        mini_batch_size=12,
        batch_size=6,
        subsample=True,
        force_nonsequential=False,
    )

    print("All tests passed.")


if __name__ == "__main__":
    main()


