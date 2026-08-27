"""Batched AFIR (Artificial Force Induced Reaction) path sampling.

AFIR drives a reactant toward a known product by applying a constant artificial
force to the atom pairs whose connectivity changes: pairs that must bond are
pulled together, pairs that must separate are pushed apart. The structure is
relaxed under that force, the force constant is ramped up, and the relaxation
repeats, so the system is dragged continuously along the reaction path. Every
relaxation step is an off-equilibrium geometry of the kind used for training
data. The functional form and default parameters follow the AFIR recipe used in
Open Catalyst's om-data (``omdata/reactivity_utils.py``), which in turn follows
Maeda's AFIR method (10.1002/wcms.1538).

Unlike the serial ASE implementation, every stage here is batched: one
:class:`AFIRBias` applies the artificial forces of all systems in a single
gather/scatter, and :class:`AFIRController` advances each system's force ramp
independently, so systems at different force levels relax together in one
optimizer batch. Finished systems are swapped out by the standard
:func:`torch_sim.optimize` machinery.

Systems are tracked by a per-system integer extra (``afir_slot`` by default)
that rides along when the batch is split, subset or reordered, so the bias and
controller stay attached to the right systems no matter what the autobatcher
does to the batch.

Example::

    result = run_afir(
        [atoms_a, atoms_b],
        model,
        bonds_forming=[[(0, 5)], []],
        bonds_breaking=[[(1, 2)], [(0, 3)]],
    )
    result.status  # e.g. ["converged", "max_force"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from torch_sim.models.interface import ModelInterface, SumModel
from torch_sim.optimizers import Optimizer
from torch_sim.runners import optimize
from torch_sim.state import SimState, initialize_state


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_sim.autobatching import InFlightAutoBatcher
    from torch_sim.trajectory import TrajectoryReporter
    from torch_sim.typing import StateLike


DEFAULT_SLOT_KEY = "afir_slot"

# Terminal statuses reported per system by AFIRController.
STATUS_RUNNING = "running"
STATUS_CONVERGED = "converged"
STATUS_MAX_FORCE = "max_force"
STATUS_COLLAPSED = "collapsed"

# Covalent radii in Angstrom, taken from ASE (Cordero et al.,
# 10.1039/B801115J) lazily to avoid a hard ASE dependency at import time.


def _covalent_radii(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    from ase.data import covalent_radii

    return torch.as_tensor(covalent_radii, device=device, dtype=dtype)


def _slots_in_state(state: SimState, slot_key: str) -> torch.Tensor:
    """Per-system slot ids carried by *state*, defaulting to identity."""
    if state.has_extras(slot_key):
        return getattr(state, slot_key).to(torch.long)
    return torch.arange(state.n_systems, device=state.device)


def _system_offsets(state: SimState) -> torch.Tensor:
    """Index of each system's first atom in the concatenated positions."""
    counts = state.n_atoms_per_system
    return torch.cumsum(counts, dim=0) - counts


def attach_slots(state: SimState, slot_key: str = DEFAULT_SLOT_KEY) -> SimState:
    """Tag each system in *state* with its index as a per-system extra.

    The tag survives splitting, concatenation and reordering of the batch, so
    :class:`AFIRBias` and :class:`AFIRController` can always map the systems
    currently in the batch back to their bond lists and ramp state.
    """
    slots = torch.arange(state.n_systems, device=state.device)
    # Registered directly as a system extra: shape-based classification would
    # misfile the tag as per-atom whenever n_atoms == n_systems.
    state._system_extras[slot_key] = slots  # noqa: SLF001
    return state


class AFIRBias(ModelInterface):
    """Constant artificial forces on the bond pairs that change in a reaction.

    For every system slot the bias holds two pair lists. *Forming* pairs feel a
    constant attractive force of magnitude ``k`` along the pair axis; *breaking*
    pairs feel a constant repulsive force of the same magnitude, switched off
    once the pair is further apart than ``push_max_dist`` so fragments are not
    accelerated forever after the bond has already separated. The energy is the
    corresponding linear potential, ``E = -sum_pairs f_ext * r`` with
    ``f_ext = -k`` for forming and ``+k`` (gated) for breaking, matching ASE's
    ``ExternalForce`` and om-data's ``AFIRPushConstraint`` exactly.

    ``k`` is a per-slot tensor and may be updated between (or during)
    relaxations -- :class:`AFIRController` raises it system by system to ramp
    the artificial force. Compose with the physical model through
    :class:`~torch_sim.models.interface.SumModel`.

    Atom indices in the pair lists are local to each system. The mapping to the
    concatenated batch is resolved on every call from the per-system slot extra
    (see :func:`attach_slots`), so the bias is oblivious to batch composition.

    Args:
        bonds_forming: Per-system lists of ``(i, j)`` atom pairs to pull
            together, indices local to the system.
        bonds_breaking: Per-system lists of ``(i, j)`` atom pairs to push
            apart, indices local to the system.
        k_start: Initial force constant in eV/Angstrom, scalar or per-system
            sequence. Defaults to 0.1.
        push_max_dist: Distance in Angstrom beyond which a breaking pair feels
            no push. Defaults to 5.0.
        slot_key: Name of the per-system extra carrying slot ids. Defaults to
            ``"afir_slot"``.
        energy_label: Non-canonical output key under which the bias energy is
            also reported so it survives SumModel and lands on the state as an
            extra (e.g. ``state.PE_AFIR``), letting the unbiased energy be
            recovered as ``state.energy - state.PE_AFIR``. Defaults to
            ``"PE_AFIR"``.
        device: Device for computations. Defaults to CPU.
        dtype: Floating-point dtype. Defaults to torch.float64.
    """

    def __init__(
        self,
        bonds_forming: Sequence[Sequence[tuple[int, int]]],
        bonds_breaking: Sequence[Sequence[tuple[int, int]]],
        k_start: float | Sequence[float] = 0.1,
        push_max_dist: float = 5.0,
        slot_key: str = DEFAULT_SLOT_KEY,
        energy_label: str = "PE_AFIR",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the AFIR bias from per-system bond lists."""
        super().__init__()
        if len(bonds_forming) != len(bonds_breaking):
            raise ValueError(
                f"bonds_forming and bonds_breaking must have one entry per system, "
                f"got {len(bonds_forming)} and {len(bonds_breaking)}"
            )
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"

        self.n_slots = len(bonds_forming)
        self.slot_key = str(slot_key)
        self.energy_label = str(energy_label)
        self.push_max_dist = float(push_max_dist)

        slots, iis, jjs, push = [], [], [], []
        for slot in range(self.n_slots):
            for i, j in bonds_forming[slot]:
                slots.append(slot), iis.append(i), jjs.append(j), push.append(False)
            for i, j in bonds_breaking[slot]:
                slots.append(slot), iis.append(i), jjs.append(j), push.append(True)
        as_long = lambda x: torch.tensor(x, device=self._device, dtype=torch.long)  # noqa: E731
        self.register_buffer("pair_slot", as_long(slots))
        self.register_buffer("pair_i", as_long(iis))
        self.register_buffer("pair_j", as_long(jjs))
        self.register_buffer(
            "pair_is_push", torch.tensor(push, device=self._device, dtype=torch.bool)
        )

        k = torch.as_tensor(k_start, device=self._device, dtype=self._dtype)
        if k.ndim == 0:
            k = k.expand(self.n_slots).clone()
        if k.shape != (self.n_slots,):
            raise ValueError(f"k_start must be scalar or length {self.n_slots}")
        self.register_buffer("k", k)

    def validate_indices(self, state: SimState) -> None:
        """Raise if any pair index is out of range for its system in *state*.

        Args:
            state: Batch whose per-system atom counts define the valid range.
        """
        slots = _slots_in_state(state, self.slot_key)
        counts = torch.full((self.n_slots,), -1, device=state.device, dtype=torch.long)
        counts[slots] = state.n_atoms_per_system.to(torch.long)
        n = counts[self.pair_slot]
        seen = n >= 0
        bad = seen & ((self.pair_i >= n) | (self.pair_j >= n))
        bad |= seen & ((self.pair_i < 0) | (self.pair_j < 0))
        bad |= seen & (self.pair_i == self.pair_j)
        if bool(bad.any()):
            idx = int(bad.nonzero()[0])
            raise ValueError(
                f"invalid pair ({int(self.pair_i[idx])}, {int(self.pair_j[idx])}) "
                f"for system slot {int(self.pair_slot[idx])} with "
                f"{int(n[idx])} atoms"
            )

    def forward(self, state: SimState, **_kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute bias energies and forces for the systems present in *state*.

        Args:
            state: Simulation state; systems are matched to their pair lists
                through the per-system slot extra.
            **_kwargs: Unused, accepted for interface compatibility.

        Returns:
            Dict with "energy" [n_systems], "forces" [n_atoms, 3], and the same
            bias energy under ``self.energy_label``.
        """
        n_systems = state.n_systems
        positions = state.positions
        slots = _slots_in_state(state, self.slot_key)

        # Row (batch position) of each slot present in this batch, -1 otherwise.
        row_of_slot = torch.full(
            (self.n_slots,), -1, device=positions.device, dtype=torch.long
        )
        row_of_slot[slots] = torch.arange(n_systems, device=positions.device)

        rows = row_of_slot[self.pair_slot.to(positions.device)]
        present = rows >= 0

        energy = torch.zeros(n_systems, device=positions.device, dtype=positions.dtype)
        forces = torch.zeros_like(positions)
        if not bool(present.any()):
            return {"energy": energy, "forces": forces, self.energy_label: energy}

        rows = rows[present]
        offsets = _system_offsets(state)
        gi = offsets[rows] + self.pair_i.to(positions.device)[present]
        gj = offsets[rows] + self.pair_j.to(positions.device)[present]
        is_push = self.pair_is_push.to(positions.device)[present]
        k = self.k.to(positions.device, positions.dtype)[
            self.pair_slot.to(positions.device)[present]
        ]

        # ASE convention: dist = r_i - r_j, force on i is f_ext * dist/|dist|,
        # energy is -|dist| * f_ext. Forming pairs use f_ext = -k (attractive);
        # breaking pairs use f_ext = +k, switched off beyond push_max_dist.
        dist = positions[gi] - positions[gj]
        r = torch.linalg.norm(dist, dim=-1)
        gate = (~is_push) | (r < self.push_max_dist)
        f_ext = torch.where(is_push, k, -k) * gate
        safe_r = torch.where(r > 0, r, torch.ones_like(r))
        pair_force = (f_ext / safe_r).unsqueeze(-1) * dist

        forces.index_add_(0, gi, pair_force)
        forces.index_add_(0, gj, -pair_force)
        energy.index_add_(0, rows, -r * f_ext)

        return {"energy": energy, "forces": forces, self.energy_label: energy}


class AFIRTargets:
    """Distance criteria that define when each system's reaction is complete.

    A forming pair counts as bonded once its distance drops below a threshold;
    a breaking pair counts as separated once its distance exceeds one. The
    baseline thresholds are multiples of the summed covalent radii
    (om-data's convention: 1.2x for forming, 1.5x for breaking). When the
    product geometry's own pair distance is supplied, the threshold is relaxed
    so that a pair also counts as done once it is as product-like as the
    product itself (within ``product_tol``): real products routinely miss the
    bare covalent-radius test for soft metal-ligand contacts, and without this
    the ramp would chase a target the product never reaches.

    Args:
        atomic_numbers: Per-system sequences of atomic numbers.
        bonds_forming: Per-system lists of forming ``(i, j)`` pairs.
        bonds_breaking: Per-system lists of breaking ``(i, j)`` pairs.
        form_cutoff: Multiple of summed covalent radii below which a forming
            pair counts as bonded. Defaults to 1.2.
        break_cutoff: Multiple of summed covalent radii above which a breaking
            pair counts as broken. Defaults to 1.5.
        product_distances: Optional per-system dicts mapping ``(i, j)`` pairs
            to their distance in the product geometry, in Angstrom.
        product_tol: Fractional slack applied to product distances.
            Defaults to 0.05.
        slot_key: Name of the per-system extra carrying slot ids.
        device: Device for computations. Defaults to CPU.
        dtype: Floating-point dtype. Defaults to torch.float64.
    """

    def __init__(
        self,
        atomic_numbers: Sequence[Sequence[int]],
        bonds_forming: Sequence[Sequence[tuple[int, int]]],
        bonds_breaking: Sequence[Sequence[tuple[int, int]]],
        form_cutoff: float = 1.2,
        break_cutoff: float = 1.5,
        product_distances: Sequence[dict[tuple[int, int], float]] | None = None,
        product_tol: float = 0.05,
        slot_key: str = DEFAULT_SLOT_KEY,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Precompute per-pair threshold distances from the bond lists."""
        device = device or torch.device("cpu")
        self.slot_key = str(slot_key)
        self.n_slots = len(bonds_forming)
        radii = _covalent_radii(device, dtype)

        slots, iis, jjs, is_break, thresholds = [], [], [], [], []
        for slot in range(self.n_slots):
            numbers = atomic_numbers[slot]
            prod = (product_distances or [None] * self.n_slots)[slot] or {}
            for pairs, breaking, scale in (
                (bonds_forming[slot], False, form_cutoff),
                (bonds_breaking[slot], True, break_cutoff),
            ):
                for i, j in pairs:
                    limit = float((radii[numbers[i]] + radii[numbers[j]]) * scale)
                    d_prod = prod.get((i, j), prod.get((j, i)))
                    if d_prod is not None:
                        if breaking:
                            limit = min(limit, d_prod * (1.0 - product_tol))
                        else:
                            limit = max(limit, d_prod * (1.0 + product_tol))
                    slots.append(slot), iis.append(i), jjs.append(j)
                    is_break.append(breaking), thresholds.append(limit)

        as_long = lambda x: torch.tensor(x, device=device, dtype=torch.long)  # noqa: E731
        self.pair_slot = as_long(slots)
        self.pair_i = as_long(iis)
        self.pair_j = as_long(jjs)
        self.pair_is_break = torch.tensor(is_break, device=device, dtype=torch.bool)
        self.threshold = torch.tensor(thresholds, device=device, dtype=dtype)

    def satisfied(self, state: SimState) -> torch.Tensor:
        """Whether every target pair of each system meets its criterion.

        Args:
            state: Batch to evaluate; systems are matched through the slot extra.

        Returns:
            Boolean tensor of shape [n_systems] in batch order.
        """
        n_systems = state.n_systems
        positions = state.positions
        slots = _slots_in_state(state, self.slot_key)
        row_of_slot = torch.full(
            (self.n_slots,), -1, device=positions.device, dtype=torch.long
        )
        row_of_slot[slots] = torch.arange(n_systems, device=positions.device)
        rows = row_of_slot[self.pair_slot.to(positions.device)]
        present = rows >= 0

        ok = torch.ones(n_systems, device=positions.device, dtype=torch.bool)
        if not bool(present.any()):
            return ok

        rows = rows[present]
        offsets = _system_offsets(state)
        gi = offsets[rows] + self.pair_i.to(positions.device)[present]
        gj = offsets[rows] + self.pair_j.to(positions.device)[present]
        r = torch.linalg.norm(positions[gi] - positions[gj], dim=-1)
        thr = self.threshold.to(positions.device, positions.dtype)[present]
        is_break = self.pair_is_break.to(positions.device)[present]
        pair_ok = torch.where(is_break, r > thr, r < thr)

        # A system is satisfied only if all of its pairs are.
        bad_rows = rows[~pair_ok]
        ok.index_put_((bad_rows,), torch.zeros_like(bad_rows, dtype=torch.bool))
        return ok


@dataclass
class AFIRController:
    """Per-system force-ramp control, packaged as an optimize convergence_fn.

    Called by :func:`torch_sim.optimize` every ``check_interval`` steps with the
    current (possibly subset and reordered) batch. For each system it decides
    whether the current force level's relaxation is finished -- biased forces
    below ``fmax``, or ``max_steps_per_level`` spent at this level -- and then
    either declares the system done (targets satisfied, or the ramp exhausted)
    or raises its force constant in place and lets it keep relaxing. Returning
    True hands the system to the optimizer's converged pool, which removes it
    from the batch.

    Args:
        bias: The :class:`AFIRBias` whose per-slot ``k`` this controller ramps.
        targets: Bond-distance criteria defining completion.
        fmax: Force convergence threshold per level, eV/Angstrom (biased
            forces). Defaults to 0.15.
        force_step: Ramp increment in eV/Angstrom. Defaults to 0.2.
        max_force: Ramp ceiling in eV/Angstrom; a system whose next level would
            reach it stops with status "max_force". Defaults to 4.0.
        max_steps_per_level: Optimizer steps allowed per force level before the
            ramp advances regardless of fmax. Defaults to 50.
        check_interval: Steps between controller calls; must equal the
            ``steps_between_swaps`` passed to optimize. Defaults to 5.
        min_atom_dist: Any interatomic distance below this stops the system
            with status "collapsed"; 0 disables the check. Defaults to 0.5.
    """

    bias: AFIRBias
    targets: AFIRTargets
    fmax: float = 0.15
    force_step: float = 0.2
    max_force: float = 4.0
    max_steps_per_level: int = 50
    check_interval: int = 5
    min_atom_dist: float = 0.5
    status: list[str] = field(init=False)
    level_steps: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        """Initialize per-slot ramp bookkeeping."""
        self.status = [STATUS_RUNNING] * self.bias.n_slots
        self.level_steps = torch.zeros(self.bias.n_slots, dtype=torch.long)

    def _collapsed(self, state: SimState) -> torch.Tensor:
        """Per-row flag for structures with atoms closer than min_atom_dist."""
        flags = torch.zeros(state.n_systems, dtype=torch.bool, device=state.device)
        if self.min_atom_dist <= 0:
            return flags
        offsets = _system_offsets(state)
        counts = state.n_atoms_per_system
        for row in range(state.n_systems):
            pos = state.positions[offsets[row] : offsets[row] + counts[row]]
            if len(pos) > 1:
                flags[row] = bool(torch.pdist(pos).min() < self.min_atom_dist)
        return flags

    def __call__(self, state: Any, _last_energy: Any = None) -> torch.Tensor:
        """Advance the ramp for the systems in *state*; return per-row done flags.

        Args:
            state: Optimizer state carrying positions and (biased) forces.
            _last_energy: Unused; accepted for convergence_fn compatibility.

        Returns:
            Boolean tensor of shape [n_systems]: True removes a system from
            the batch.
        """
        device = state.device
        slots = _slots_in_state(state, self.bias.slot_key)
        self.level_steps = self.level_steps.to(device)
        self.level_steps[slots] += self.check_interval

        force_norm = torch.linalg.norm(state.forces, dim=-1)
        fmax_per_row = torch.zeros(
            state.n_systems, device=device, dtype=force_norm.dtype
        ).scatter_reduce(
            0, state.system_idx, force_norm, reduce="amax", include_self=False
        )
        relaxed = (fmax_per_row < self.fmax) | (
            self.level_steps[slots] >= self.max_steps_per_level
        )

        satisfied = self.targets.satisfied(state)
        collapsed = self._collapsed(state)
        done = torch.zeros(state.n_systems, dtype=torch.bool, device=device)

        for row in torch.nonzero(collapsed).flatten().tolist():
            done[row] = True
            self.status[int(slots[row])] = STATUS_COLLAPSED

        for row in torch.nonzero(relaxed & ~collapsed).flatten().tolist():
            slot = int(slots[row])
            if bool(satisfied[row]):
                done[row] = True
                self.status[slot] = STATUS_CONVERGED
            elif float(self.bias.k[slot]) + self.force_step >= self.max_force:
                done[row] = True
                self.status[slot] = STATUS_MAX_FORCE
            else:
                self.bias.k[slot] += self.force_step
                self.level_steps[slot] = 0
        return done


@dataclass
class AFIRResult:
    """Outcome of a batched AFIR run.

    Attributes:
        state: Final states concatenated in the original system order.
        status: Terminal status per system: "converged", "max_force",
            "collapsed", or "running" (max_steps reached mid-ramp).
        k_final: Final force constant per system in eV/Angstrom.
        satisfied_at_start: Whether each system already met its bond criteria
            at the input geometry (such systems are still relaxed and ramped).
    """

    state: SimState
    status: list[str]
    k_final: torch.Tensor
    satisfied_at_start: torch.Tensor


def run_afir(
    system: StateLike,
    model: ModelInterface,
    bonds_forming: Sequence[Sequence[tuple[int, int]]],
    bonds_breaking: Sequence[Sequence[tuple[int, int]]],
    *,
    product_distances: Sequence[dict[tuple[int, int], float]] | None = None,
    start_force: float = 0.1,
    force_step: float = 0.2,
    max_force: float = 4.0,
    fmax: float = 0.15,
    max_steps_per_level: int = 50,
    max_steps: int = 10_000,
    check_interval: int = 5,
    push_max_dist: float = 5.0,
    form_cutoff: float = 1.2,
    break_cutoff: float = 1.5,
    product_tol: float = 0.05,
    min_atom_dist: float = 0.5,
    optimizer: Optimizer = Optimizer.bfgs,
    trajectory_reporter: TrajectoryReporter | dict | None = None,
    autobatcher: InFlightAutoBatcher | bool = False,
    **optimize_kwargs: Any,
) -> AFIRResult:
    """Drive a batch of systems from reactant toward product with ramped forces.

    Each system relaxes under its own artificial force (see :class:`AFIRBias`);
    a shared :class:`AFIRController` raises each force constant independently
    whenever a level's relaxation finishes without the target bonds reaching
    their formed/broken distances. All systems advance together in one batched
    optimization, and finished systems are swapped out by the optimize
    machinery, with an autobatcher pulling pending systems in if one is given.

    Trajectory frames record the *biased* total energy; the bias contribution
    is stored alongside under the bias's ``energy_label`` (``PE_AFIR``), so the
    physical energy is their difference. Recompute forces with the bare model
    if unbiased force labels are needed.

    Args:
        system: Input systems (ASE Atoms, Structures, SimState, or lists).
        model: Physical model; composed with the bias via SumModel.
        bonds_forming: Per-system lists of ``(i, j)`` pairs to pull together,
            atom indices local to each system.
        bonds_breaking: Per-system lists of ``(i, j)`` pairs to push apart.
        product_distances: Optional per-system ``{(i, j): distance}`` maps from
            the product geometry, used to make the completion criteria
            product-aware (see :class:`AFIRTargets`).
        start_force: Initial force constant in eV/Angstrom. Defaults to 0.1.
        force_step: Ramp increment in eV/Angstrom. Defaults to 0.2.
        max_force: Ramp ceiling in eV/Angstrom. Defaults to 4.0.
        fmax: Per-level force convergence threshold. Defaults to 0.15.
        max_steps_per_level: Step budget per force level. Defaults to 50.
        max_steps: Total optimizer step budget per system. Defaults to 10000.
        check_interval: Steps between controller checks (equals the optimize
            ``steps_between_swaps``). Defaults to 5.
        push_max_dist: Push cut-off distance for breaking pairs. Defaults to 5.
        form_cutoff: Forming criterion, multiple of covalent radii sum.
            Defaults to 1.2.
        break_cutoff: Breaking criterion, multiple of covalent radii sum.
            Defaults to 1.5.
        product_tol: Fractional slack on product distances. Defaults to 0.05.
        min_atom_dist: Collapse guard distance; 0 disables. Defaults to 0.5.
        optimizer: Batched optimizer to relax with. Defaults to BFGS, which
            matches the serial recipe's ASE BFGS. FIRE needs far more steps
            per force level here, so within the 50-step level budget it makes
            too little geometric progress and misclassifies reactions as
            "max_force" that BFGS completes.
        trajectory_reporter: Optional reporter; one file per system.
        autobatcher: Optional InFlightAutoBatcher (or True to auto-configure)
            for corpora larger than one batch.
        **optimize_kwargs: Passed through to :func:`torch_sim.optimize`.

    Returns:
        AFIRResult with final states (original order), per-system statuses,
        final force constants and the satisfied-at-start flags.
    """
    n_changed = [
        len(f) + len(b) for f, b in zip(bonds_forming, bonds_breaking, strict=True)
    ]
    if any(n == 0 for n in n_changed):
        idx = n_changed.index(0)
        raise ValueError(f"system {idx} has no forming or breaking bonds")

    state = initialize_state(system, model.device, model.dtype)
    if state.n_systems != len(bonds_forming):
        raise ValueError(
            f"got {state.n_systems} systems but bond lists for {len(bonds_forming)}"
        )
    attach_slots(state)

    bias = AFIRBias(
        bonds_forming,
        bonds_breaking,
        k_start=start_force,
        push_max_dist=push_max_dist,
        device=model.device,
        dtype=model.dtype,
    )
    bias.validate_indices(state)

    numbers_per_system = [s.atomic_numbers.tolist() for s in state.split()]
    targets = AFIRTargets(
        numbers_per_system,
        bonds_forming,
        bonds_breaking,
        form_cutoff=form_cutoff,
        break_cutoff=break_cutoff,
        product_distances=product_distances,
        product_tol=product_tol,
        device=model.device,
        dtype=model.dtype,
    )
    controller = AFIRController(
        bias=bias,
        targets=targets,
        fmax=fmax,
        force_step=force_step,
        max_force=max_force,
        max_steps_per_level=max_steps_per_level,
        check_interval=check_interval,
        min_atom_dist=min_atom_dist,
    )
    satisfied_at_start = targets.satisfied(state).clone()

    final_state = optimize(
        system=state,
        model=SumModel(model, bias),
        optimizer=optimizer,
        convergence_fn=controller,
        max_steps=max_steps,
        steps_between_swaps=check_interval,
        trajectory_reporter=trajectory_reporter,
        autobatcher=autobatcher,
        **optimize_kwargs,
    )

    # Restore the original system order for statuses and force constants.
    order = _slots_in_state(final_state, bias.slot_key).tolist()
    if sorted(order) != list(range(bias.n_slots)):
        raise RuntimeError(f"lost track of systems: final slots {order}")
    status = [controller.status[slot] for slot in order]

    return AFIRResult(
        state=final_state,
        status=status,
        k_final=bias.k[torch.tensor(order, device=bias.k.device)].clone(),
        satisfied_at_start=satisfied_at_start,
    )
