import numpy as np
import pytest
import torch
from ase import Atoms
from ase.constraints import ExternalForce

import torch_sim as ts
from torch_sim.enhanced_sampling.afir import (
    STATUS_COLLAPSED,
    STATUS_CONVERGED,
    STATUS_MAX_FORCE,
    AFIRBias,
    AFIRTargets,
    attach_slots,
    run_afir,
)
from torch_sim.models.interface import SumModel
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.optimizers.fire import fire_init, fire_step


DEVICE = torch.device("cpu")
DTYPE = torch.float64

# Argon covalent radius in ASE's table; LJ tests below build Ar systems whose
# forming/breaking thresholds derive from it (1.06 A).
AR_RCOV = 1.06


class ReferencePushConstraint:
    """om-data's AFIRPushConstraint (reactivity_utils.py), verbatim reference."""

    def __init__(self, a1: int, a2: int, f_ext: float, max_dist: float = 5.0) -> None:
        self.indices = [a1, a2]
        self.external_force = f_ext
        self.max_dist = max_dist

    def adjust_forces(self, atoms: Atoms, forces: np.ndarray) -> None:
        dist = np.subtract.reduce(atoms.positions[self.indices])
        if self.max_dist is not None and np.linalg.norm(dist) < self.max_dist:
            force = self.external_force * dist / np.linalg.norm(dist)
            forces[self.indices] += (force, -force)

    def adjust_potential_energy(self, atoms: Atoms) -> float:
        dist = np.subtract.reduce(atoms.positions[self.indices])
        if self.max_dist is not None and np.linalg.norm(dist) < self.max_dist:
            return -np.linalg.norm(dist) * self.external_force
        return 0.0


def single_state(atoms: Atoms) -> ts.SimState:
    state = ts.io.atoms_to_state([atoms], DEVICE, DTYPE)
    return attach_slots(state)


@pytest.fixture
def lj_model() -> LennardJonesModel:
    return LennardJonesModel(sigma=2.0, epsilon=0.1, device=DEVICE, dtype=DTYPE)


class TestAFIRBias:
    def test_matches_ase_reference(self) -> None:
        """Energy and forces agree with ASE ExternalForce + om-data's push."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            n = int(rng.integers(4, 12))
            atoms = Atoms("C" * n, positions=rng.normal(size=(n, 3)) * 2.5)
            k = float(rng.uniform(0.1, 3.9))
            idx = rng.permutation(n)
            forming = [(int(idx[0]), int(idx[1]))]
            breaking = [(int(idx[2]), int(idx[3]))]

            f_ref = np.zeros((n, 3))
            e_ref = 0.0
            constraints = [ExternalForce(i, j, -k) for i, j in forming] + [
                ReferencePushConstraint(i, j, k) for i, j in breaking
            ]
            for con in constraints:
                con.adjust_forces(atoms, f_ref)
                e_ref += con.adjust_potential_energy(atoms)

            bias = AFIRBias([forming], [breaking], k_start=k, dtype=DTYPE)
            out = bias.forward(single_state(atoms))
            torch.testing.assert_close(
                out["energy"][0], torch.tensor(e_ref, dtype=DTYPE), atol=1e-12, rtol=0
            )
            torch.testing.assert_close(
                out["forces"], torch.tensor(f_ref, dtype=DTYPE), atol=1e-12, rtol=0
            )

    def test_push_gated_beyond_max_dist(self) -> None:
        atoms = Atoms("CC", positions=[[0, 0, 0], [6.0, 0, 0]])
        bias = AFIRBias([[]], [[(0, 1)]], k_start=2.0, push_max_dist=5.0, dtype=DTYPE)
        out = bias.forward(single_state(atoms))
        assert float(out["energy"].abs().max()) == 0.0
        assert float(out["forces"].abs().max()) == 0.0

    def test_batch_matches_individual(self) -> None:
        """A shuffled sub-batch reproduces the per-system results exactly."""
        rng = np.random.default_rng(3)
        systems = [
            Atoms("Ar" * n, positions=rng.normal(size=(n, 3)) * 3) for n in (3, 5, 4)
        ]
        forming = [[(0, 1)], [(1, 4)], []]
        breaking = [[(1, 2)], [], [(0, 3)]]
        bias = AFIRBias(forming, breaking, k_start=[0.5, 1.0, 1.5], dtype=DTYPE)

        singles = {}
        for slot, atoms in enumerate(systems):
            state = ts.io.atoms_to_state([atoms], DEVICE, DTYPE)
            state._system_extras["afir_slot"] = torch.tensor([slot])  # noqa: SLF001
            singles[slot] = bias.forward(state)

        # batch systems 2 and 0, reordered relative to their slots
        batch = attach_slots(ts.io.atoms_to_state(systems, DEVICE, DTYPE))
        sub = ts.concatenate_states([batch.split()[2], batch.split()[0]])
        out = bias.forward(sub)
        torch.testing.assert_close(out["energy"][0], singles[2]["energy"][0])
        torch.testing.assert_close(out["energy"][1], singles[0]["energy"][0])
        n2 = len(systems[2])
        torch.testing.assert_close(out["forces"][:n2], singles[2]["forces"])
        torch.testing.assert_close(out["forces"][n2:], singles[0]["forces"])

    def test_validate_indices(self) -> None:
        atoms = Atoms("ArAr", positions=[[0, 0, 0], [3, 0, 0]])
        bias = AFIRBias([[(0, 5)]], [[]], dtype=DTYPE)
        with pytest.raises(ValueError, match="invalid pair"):
            bias.validate_indices(single_state(atoms))
        self_bond = AFIRBias([[(1, 1)]], [[]], dtype=DTYPE)
        with pytest.raises(ValueError, match="invalid pair"):
            self_bond.validate_indices(single_state(atoms))

    def test_mismatched_system_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="one entry per system"):
            AFIRBias([[(0, 1)]], [[], []], dtype=DTYPE)


class TestAFIRTargets:
    def test_forming_and_breaking_thresholds(self) -> None:
        # Ar-Ar: forming satisfied below 1.2 * 2.12 = 2.544, breaking above
        # 1.5 * 2.12 = 3.18.
        targets = AFIRTargets(
            atomic_numbers=[[18, 18], [18, 18]],
            bonds_forming=[[(0, 1)], []],
            bonds_breaking=[[], [(0, 1)]],
            dtype=DTYPE,
        )
        near = Atoms("Ar2", positions=[[0, 0, 0], [2.0, 0, 0]])
        far = Atoms("Ar2", positions=[[0, 0, 0], [4.0, 0, 0]])
        state = attach_slots(ts.io.atoms_to_state([near, far], DEVICE, DTYPE))
        assert targets.satisfied(state).tolist() == [True, True]
        state_swapped = attach_slots(ts.io.atoms_to_state([far, near], DEVICE, DTYPE))
        assert targets.satisfied(state_swapped).tolist() == [False, False]

    def test_product_aware_relaxation(self) -> None:
        """Product distances loosen the covalent-radius criteria, both ways."""
        # Forming pair sitting at 3.0 A: fails 2.544 A, passes once the product
        # distance (2.9 A) grants 2.9 * 1.05 = 3.045 A.
        plain = AFIRTargets([[18, 18]], [[(0, 1)]], [[]], dtype=DTYPE)
        aware = AFIRTargets(
            [[18, 18]],
            [[(0, 1)]],
            [[]],
            product_distances=[{(0, 1): 2.9}],
            dtype=DTYPE,
        )
        # reversed pair key must be found too
        aware_rev = AFIRTargets(
            [[18, 18]],
            [[(0, 1)]],
            [[]],
            product_distances=[{(1, 0): 2.9}],
            dtype=DTYPE,
        )
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
        state = single_state(atoms)
        assert not bool(plain.satisfied(state)[0])
        assert bool(aware.satisfied(state)[0])
        assert bool(aware_rev.satisfied(state)[0])

        # Breaking pair at 3.0 A: fails the covalent 3.18 A requirement, but a
        # product whose pair sits at only 2.9 A relaxes the limit to
        # 2.9 * 0.95 = 2.755 A -- product distances only ever loosen criteria,
        # in both directions, matching the serial implementation.
        plain_b = AFIRTargets([[18, 18]], [[]], [[(0, 1)]], dtype=DTYPE)
        aware_b = AFIRTargets(
            [[18, 18]],
            [[]],
            [[(0, 1)]],
            product_distances=[{(0, 1): 2.9}],
            dtype=DTYPE,
        )
        atoms_b = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
        state_b = single_state(atoms_b)
        assert not bool(plain_b.satisfied(state_b)[0])
        assert bool(aware_b.satisfied(state_b)[0])


class TestSlotTracking:
    def test_slots_survive_optimizer_and_batching(
        self, lj_model: LennardJonesModel
    ) -> None:
        from ase.build import molecule

        systems = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
        state = attach_slots(ts.initialize_state(systems, DEVICE, DTYPE))
        bias = AFIRBias([[(0, 1)], [(0, 2)], []], [[], [], [(0, 1)]], dtype=DTYPE)
        model = SumModel(lj_model, bias)
        fire_state = fire_init(state, model)
        assert fire_state.afir_slot.tolist() == [0, 1, 2]
        stepped = fire_step(state=fire_state, model=model)
        assert stepped.afir_slot.tolist() == [0, 1, 2]
        assert stepped.has_extras(bias.energy_label)
        reordered = ts.concatenate_states([stepped.split()[2], stepped.split()[0]])
        assert reordered.afir_slot.tolist() == [2, 0]


class TestRunAFIR:
    def test_forming_and_breaking_converge(self, lj_model: LennardJonesModel) -> None:
        """Systems finishing at different times all reach their targets."""
        pull = Atoms("Ar2", positions=[[0, 0, 0], [5.5, 0, 0]])
        push = Atoms("Ar2", positions=[[0, 0, 0], [2.2, 0, 0]])
        push2 = Atoms("Ar3", positions=[[0, 0, 0], [2.2, 0, 0], [0, 2.2, 0]])
        result = run_afir(
            [pull, push, push2],
            lj_model,
            bonds_forming=[[(0, 1)], [], []],
            bonds_breaking=[[], [(0, 1)], [(0, 1), (0, 2)]],
            max_steps=4000,
        )
        assert result.status == [STATUS_CONVERGED] * 3
        assert result.satisfied_at_start.tolist() == [False] * 3
        final = result.state.split()
        r_pull = torch.linalg.norm(final[0].positions[0] - final[0].positions[1])
        r_push = torch.linalg.norm(final[1].positions[0] - final[1].positions[1])
        assert float(r_pull) < 1.2 * 2 * AR_RCOV
        assert float(r_push) > 1.5 * 2 * AR_RCOV
        # original order restored
        assert [int(s.afir_slot) for s in final] == [0, 1, 2]

    def test_max_force_status(self, lj_model: LennardJonesModel) -> None:
        """A push gated off before the target distance can never satisfy it."""
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [2.2, 0, 0]])
        result = run_afir(
            [atoms],
            lj_model,
            bonds_forming=[[]],
            bonds_breaking=[[(0, 1)]],
            push_max_dist=2.5,  # push stops well below the 3.18 A criterion
            max_steps=4000,
        )
        assert result.status == [STATUS_MAX_FORCE]
        # the ramp was exhausted: one step below the 4.0 ceiling
        assert float(result.k_final[0]) == pytest.approx(3.9)

    def test_collapsed_status(self, lj_model: LennardJonesModel) -> None:
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [2.2, 0, 0]])
        result = run_afir(
            [atoms],
            lj_model,
            bonds_forming=[[]],
            bonds_breaking=[[(0, 1)]],
            min_atom_dist=3.0,  # current separation already "collapsed"
            max_steps=200,
        )
        assert result.status == [STATUS_COLLAPSED]

    def test_no_bond_change_rejected(self, lj_model: LennardJonesModel) -> None:
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [2.2, 0, 0]])
        with pytest.raises(ValueError, match="no forming or breaking bonds"):
            run_afir([atoms], lj_model, bonds_forming=[[]], bonds_breaking=[[]])

    def test_satisfied_at_start_flag(self, lj_model: LennardJonesModel) -> None:
        apart = Atoms("Ar2", positions=[[0, 0, 0], [5.0, 0, 0]])
        result = run_afir(
            [apart],
            lj_model,
            bonds_forming=[[]],
            bonds_breaking=[[(0, 1)]],
            max_steps=500,
        )
        assert result.satisfied_at_start.tolist() == [True]
        assert result.status == [STATUS_CONVERGED]
