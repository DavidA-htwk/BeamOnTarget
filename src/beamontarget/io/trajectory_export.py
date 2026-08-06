"""Utilities for exporting per-particle EM trajectories to VTK PolyData
(.vtp) files for visualization in ParaView.
"""
import os
import threading
import numpy as np
import pyvista as pv


class OriginQuotaTracker:
    """Thread-safe per-origin (per ``source_index``) trajectory-recording
    quota, shared across all engine batches for a run.

    Distributes a total particle budget evenly across every distinct beam
    origin (``source_index``) so each origin contributes roughly the same
    number of recorded trajectories, instead of whichever origin happens to
    be processed first (e.g. earliest rows in the .bl file) consuming the
    entire budget.
    """

    def __init__(self, source_indices, total_budget):
        unique = sorted(set(int(s) for s in source_indices))
        self.num_origins = max(1, len(unique))
        total_budget = int(total_budget) if total_budget else 0
        self.quota_per_origin = max(1, total_budget // self.num_origins) if total_budget > 0 else 0
        self._remaining = {src: self.quota_per_origin for src in unique}
        self._total_remaining = self.quota_per_origin * self.num_origins
        self._lock = threading.Lock()

    def reserve(self, source_indices_per_particle):
        """*source_indices_per_particle*: array-like of ``source_index``
        values indexed by local particle_id (i.e. index == particle_id).

        Returns the set of local particle_ids allowed to be recorded,
        decrementing the corresponding origin's remaining quota for each
        particle accepted. Thread-safe — call once per engine batch.
        """
        if self._total_remaining <= 0:
            return set()
        allowed = set()
        with self._lock:
            if self._total_remaining <= 0:
                return set()
            for pid, src in enumerate(source_indices_per_particle):
                src = int(src)
                remaining = self._remaining.get(src, 0)
                if remaining > 0:
                    self._remaining[src] = remaining - 1
                    self._total_remaining -= 1
                    allowed.add(pid)
        return allowed


class TrajectoryRecorder:
    """Accumulates EM trajectory segments from potentially many particle
    batches (processed concurrently across worker threads) into a single
    set of recorded particle trajectories, then writes them all out as one
    VTK PolyData (.vtp) file — one polyline per recorded particle.

    A single instance is meant to be shared across the WHOLE run (not
    created per batch), since particle origins are scattered across many
    batches — one recorder per batch would produce one file per batch.
    ``add_segments`` is thread-safe so it can be called concurrently from
    multiple worker threads.
    """

    def __init__(self):
        self._points = {}           # global particle_id -> list of [x, y, z]
        self._charge = {}           # global particle_id -> list of charge_state
        self._energy = {}           # global particle_id -> list of kinetic_energy_eV
        self._step = {}             # global particle_id -> list of step_index
        self._lock = threading.Lock()

    def add_segments(self, segments, allowed_local_pids, particle_id_offset=0):
        """Feed a chunk of trajectory segments (structured array with fields
        particle_id, step_index, start_pos, end_pos, charge_state,
        kinetic_energy_ev) from one batch.

        *allowed_local_pids*: set of local particle_id values (as used in
        this batch's segments) that should be recorded — decided externally
        (e.g. by OriginQuotaTracker), so the selection is unbiased with
        respect to processing order.
        *particle_id_offset*: added to each local particle_id to form a
        globally unique key, since different batches reuse the same local
        particle_id numbering (0..batch_size-1).

        Safe to call multiple times per batch, and concurrently from
        multiple batches/threads.
        """
        if segments is None or len(segments) == 0 or not allowed_local_pids:
            return

        mask = np.isin(segments['particle_id'], list(allowed_local_pids))
        if not np.any(mask):
            return
        seg = segments[mask]

        # Defensive ordering — checkpoint chunks are already step-ordered.
        order = np.argsort(seg['step_index'], kind='stable')
        seg = seg[order]

        with self._lock:
            for row in seg:
                pid = int(row['particle_id']) + particle_id_offset
                if pid not in self._points:
                    self._points[pid] = [row['start_pos'].astype(np.float64)]
                    self._charge[pid] = [int(row['charge_state'])]
                    self._energy[pid] = [float(row['kinetic_energy_ev'])]
                    self._step[pid] = [int(row['step_index'])]
                self._points[pid].append(row['end_pos'].astype(np.float64))
                self._charge[pid].append(int(row['charge_state']))
                self._energy[pid].append(float(row['kinetic_energy_ev']))
                self._step[pid].append(int(row['step_index']) + 1)

    @property
    def n_particles_recorded(self):
        return len(self._points)

    def build_polydata(self):
        """Assemble accumulated per-particle point chains into a single
        pyvista PolyData with one polyline per particle. Call only after
        all batches have finished feeding this recorder."""
        if not self._points:
            return None

        all_points, lines = [], []
        pid_arr, charge_arr, energy_arr, step_arr = [], [], [], []
        offset = 0

        for pid in sorted(self._points.keys()):
            pts = self._points[pid]
            n = len(pts)
            if n < 2:
                continue
            all_points.extend(pts)
            lines.append(n)
            lines.extend(range(offset, offset + n))
            pid_arr.extend([pid] * n)
            charge_arr.extend(self._charge[pid])
            energy_arr.extend(self._energy[pid])
            step_arr.extend(self._step[pid])
            offset += n

        if not all_points:
            return None

        poly = pv.PolyData(np.asarray(all_points, dtype=np.float64))
        poly.lines = np.asarray(lines, dtype=np.int64)
        poly.point_data["ParticleID"] = np.asarray(pid_arr, dtype=np.int32)
        poly.point_data["ChargeState"] = np.asarray(charge_arr, dtype=np.int32)
        poly.point_data["KineticEnergy_eV"] = np.asarray(energy_arr, dtype=np.float64)
        poly.point_data["StepIndex"] = np.asarray(step_arr, dtype=np.int32)
        return poly

    def save_vtp(self, path):
        """Build the polydata and write it to *path*.
        Returns True if a file was written, False if there was nothing to record."""
        poly = self.build_polydata()
        if poly is None:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        poly.save(path, binary=True)
        return True



def write_pvd_manifest(pvd_path, vtp_paths):
    """Write a ParaView Data (.pvd) manifest grouping multiple .vtp pieces
    (one per batch) so they all load together as a single dataset in
    ParaView. Paths are stored relative to the manifest's directory."""
    if not vtp_paths:
        return
    manifest_dir = os.path.dirname(pvd_path)
    os.makedirs(manifest_dir, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for i, vtp_path in enumerate(vtp_paths):
        rel_path = os.path.relpath(vtp_path, manifest_dir).replace(os.sep, "/")
        lines.append(f'    <DataSet timestep="0" group="" part="{i}" file="{rel_path}"/>')
    lines.append('  </Collection>')
    lines.append('</VTKFile>')
    with open(pvd_path, "w") as f:
        f.write("\n".join(lines) + "\n")
