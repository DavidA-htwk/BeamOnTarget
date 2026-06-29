"""Species/reaction framework for EM particle tracking."""

import os
import numpy as np
from beamontarget.constants import ELEMENTARY_CHARGE_C, HYDROGEN_MASS_KG, DEUTERIUM_MASS_KG
from beamontarget.interactions import cross_sections
from scipy.interpolate import interp1d
from beamontarget.paths import get_project_root


_PROJECT_ROOT = str(get_project_root())


class ReactionModel:
    """Base reaction interface."""

    def apply(self, species_frame, positions_m, velocities_mps, dt_s, rng):
        """Apply reactions in-place to species properties.

        Returns:
            collision_dt_s: Array (N,) with sampled collision time in seconds for
                particles that reacted during this step; np.inf otherwise.
        """
        raise NotImplementedError


class NullReactionModel(ReactionModel):
    """No species transformations."""

    def apply(self, species_frame, positions_m, velocities_mps, dt_s, rng):
        n = len(species_frame.charge_state_e)
        return np.full(n, np.inf, dtype=np.float64)


class BeamCrossSectionReaction(ReactionModel):
    """Beam-gas reactions driven by fitted cross sections.

    Supported channels:
      - H-/D- -> H0/D0  (single stripping)
      - H-/D- -> H+/D+  (double stripping)
      - H0/D0 -> H+/D+  (projectile ionization)
      - H+/D+ -> H0/D0  (charge exchange)
    """

    def __init__(
        self,
        background_density_m3=0.0,
        density_profile_file=None,
        density_profile_scale=1.0,
        density_profile_direction=(1.0, 0.0, 0.0),
        fixed_cs=False,
        manual_cross_sections=None,
        verbose=False,
    ):
        self.background_density_m3 = float(background_density_m3)
        self.density_profile_file = density_profile_file
        self.density_profile_scale = float(density_profile_scale)
        if not np.isfinite(self.density_profile_scale) or self.density_profile_scale < 0.0:
            raise ValueError("density_profile_scale must be finite and >= 0")
        self.density_profile_file_resolved = None
        self.density_profile_positions_m = None
        self.density_profile_values_m3 = None
        self.density_profile_direction = self._normalize_direction(density_profile_direction)
        self.fixed_cs = bool(fixed_cs)
        self.manual_cross_sections = dict(manual_cross_sections) if manual_cross_sections else None
        self._cached_sigmas = None   # populated on first apply() call when fixed_cs=True

        if density_profile_file:
            self._load_density_profile(density_profile_file)

        if verbose:
            self._print_configuration_summary()

    @staticmethod
    def _normalize_direction(direction):
        vec = np.asarray(direction, dtype=np.float64)
        if vec.shape != (3,):
            raise ValueError("main beam axis direction must be a 3D vector")
        norm = np.linalg.norm(vec)
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("main beam axis direction must be finite and non-zero")
        return vec / norm

    def _load_density_profile(self, profile_path):
        path = str(profile_path)
        if not path.lower().endswith(".dens"):
            raise ValueError("density_profile_file must have .dens extension")

        if not os.path.isabs(path):
            base_dir = _PROJECT_ROOT
            if os.path.dirname(path):
                path = os.path.abspath(os.path.join(base_dir, path))
            else:
                path = os.path.abspath(os.path.join(base_dir, "InputFiles", path))

        self.density_profile_file_resolved = path

        try:
            data = np.loadtxt(path, dtype=np.float64)
        except Exception:
            try:
                # Fall back to a tolerant reader so .dens files may include a header row.
                data = np.genfromtxt(path, dtype=np.float64, invalid_raise=False)
            except Exception as exc:
                raise ValueError(f"Failed to load density profile file '{path}': {exc}") from exc

        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError(
                f"Density profile '{path}' must contain at least 2 columns: position[m], density[m^-3]"
            )

        pos = data[:, 0]
        den = data[:, 1]

        valid = np.isfinite(pos) & np.isfinite(den)
        pos = pos[valid]
        den = den[valid]

        if pos.size < 2:
            raise ValueError(f"Density profile '{path}' must contain at least 2 valid rows")

        den = np.maximum(den, 0.0)
        order = np.argsort(pos)
        pos = pos[order]
        den = den[order]

        # Deduplicate repeated coordinates by averaging their densities.
        uniq_pos, inv = np.unique(pos, return_inverse=True)
        uniq_den = np.zeros_like(uniq_pos)
        counts = np.zeros_like(uniq_pos)
        np.add.at(uniq_den, inv, den)
        np.add.at(counts, inv, 1.0)
        uniq_den /= np.maximum(counts, 1.0)

        if uniq_pos.size < 2:
            raise ValueError(
                f"Density profile '{path}' must contain at least 2 distinct position points"
            )

        self.density_profile_positions_m = uniq_pos
        self.density_profile_values_m3 = uniq_den * self.density_profile_scale

        # Build the interpolator once (reused by every _density_at_positions call)
        self._density_interp = interp1d(
            uniq_pos, self.density_profile_values_m3,
            kind='linear',
            fill_value=(self.density_profile_values_m3[0], self.density_profile_values_m3[-1]),
            bounds_error=False,
        )

    def _print_configuration_summary(self):
        direction_str = np.array2string(self.density_profile_direction, precision=6)
        if self.density_profile_positions_m is None:
            print(
                "BeamCrossSectionReaction: using uniform background density "
                f"n={self.background_density_m3:.6e} m^-3, "
                f"direction={direction_str}."
            )
            return

        points = int(self.density_profile_positions_m.size)
        s_min = float(self.density_profile_positions_m[0])
        s_max = float(self.density_profile_positions_m[-1])
        n_min = float(np.min(self.density_profile_values_m3))
        n_max = float(np.max(self.density_profile_values_m3))
        print(
            "BeamCrossSectionReaction: loaded density profile "
            f"'{self.density_profile_file_resolved}' with {points} points, "
            f"s-range=[{s_min:.6e}, {s_max:.6e}] m, "
            f"n-range=[{n_min:.6e}, {n_max:.6e}] m^-3, "
            f"scale={self.density_profile_scale:.6g}, "
            f"direction={direction_str}."
        )

    def _density_at_positions(self, positions_m):
        if self.density_profile_positions_m is None:
            return np.full(len(positions_m), self.background_density_m3, dtype=np.float64)
        # Project positions onto the profile direction to get coordinate s, then interpolate density at s.
        s = np.einsum("ij,j->i", np.asarray(positions_m, dtype=np.float64), self.density_profile_direction)
        return self._density_interp(s)

    def apply(self, species_frame, positions_m, velocities_mps, dt_s, rng):
        if self.background_density_m3 <= 0.0 and self.density_profile_positions_m is None:
            n0 = len(species_frame.charge_state_e)
            return np.full(n0, np.inf, dtype=np.float64)

        n = len(species_frame.charge_state_e)
        if n == 0:
            return np.empty(0, dtype=np.float64)

        speed = np.sqrt(np.maximum(np.einsum('ij,ij->i', velocities_mps, velocities_mps), 0.0))
        if not np.any(speed > 0.0):
            return np.full(n, np.inf, dtype=np.float64)

        dt = np.asarray(dt_s, dtype=np.float64)
        if dt.ndim == 0:
            dt = np.full(n, float(dt), dtype=np.float64)
        else:
            dt = np.asarray(dt, dtype=np.float64)
            if dt.shape != (n,):
                dt = np.broadcast_to(dt, (n,)).astype(np.float64, copy=False)
        dt = np.maximum(dt, 0.0)

        mass = np.asarray(species_frame.mass_kg, dtype=np.float64)
        energy_ev = 0.5 * mass * speed * speed / ELEMENTARY_CHARGE_C

        # Auto-detect isotope from particle mass
        mean_mass = float(np.mean(mass))
        isotope = "D" if abs(mean_mass - DEUTERIUM_MASS_KG) < abs(mean_mass - HYDROGEN_MASS_KG) else "H"

        density_m3 = self._density_at_positions(positions_m)

        if self.manual_cross_sections:
            # Use user-supplied constant cross-sections (scalar values in m²)
            sigmas = self.manual_cross_sections
            rates = {}
            for k, sigma in sigmas.items():
                rates[k] = np.asarray(sigma, dtype=np.float64) * speed * density_m3
        elif self.fixed_cs:
            # Compute cross-sections once at mean initial energy (scalar), reuse on subsequent calls.
            # Using mean energy produces scalar sigmas that broadcast with any batch size.
            if self._cached_sigmas is None:
                mean_energy = float(np.mean(energy_ev))
                self._cached_sigmas = cross_sections.channel_cross_sections(
                    energy_ev=mean_energy, isotope=isotope,
                )
            sigmas = self._cached_sigmas
            # Compute rates from cached sigmas: rate = sigma * speed * density
            rates = {}
            for k, sigma in sigmas.items():
                rates[k] = np.asarray(sigma, dtype=np.float64) * speed * density_m3
        else:
            rates = cross_sections.channel_rates_s(
                energy_ev=energy_ev,
                speed_mps=speed,
                background_density_m3=density_m3,
                isotope=isotope,
            )

        charge = species_frame.charge_state_e
        initial_charge = charge.copy()
        collision_dt_s = np.full(n, np.inf, dtype=np.float64)

        rate_single_all = np.maximum(
            np.asarray(rates[cross_sections.CH_SINGLE_STRIP], dtype=np.float64), 0.0
        )
        rate_double_all = np.maximum(
            np.asarray(rates[cross_sections.CH_DOUBLE_STRIP], dtype=np.float64), 0.0
        )
        rate_strip_all = np.maximum(
            np.asarray(rates[cross_sections.CH_NEUTRAL_STRIP], dtype=np.float64), 0.0
        )
        rate_cx_all = np.maximum(
            np.asarray(rates[cross_sections.CH_CHARGE_EXCHANGE], dtype=np.float64), 0.0
        )

        # Sample event time t = -ln(1-u) / rate_total for each charge family.
        u_time = rng.random(n)
        u_time = np.clip(u_time, 0.0, 1.0 - np.finfo(np.float64).eps)

        # H-/D-: competing single and double stripping channels.
        neg_mask = initial_charge == -1
        rate_tot_neg = rate_single_all + rate_double_all
        t_neg = np.where(
            rate_tot_neg > 0.0,
            -np.log1p(-u_time) / np.maximum(rate_tot_neg, 1e-300),
            np.inf,
        )
        reacted_neg = neg_mask & (t_neg < dt)
        collision_dt_s[reacted_neg] = t_neg[reacted_neg]

        # Channel selection conditioned on an actual reaction in this step.
        reacted_neg_idx = np.flatnonzero(reacted_neg)
        if reacted_neg_idx.size > 0:
            u2 = rng.random(reacted_neg_idx.size)
            sub_to_single = u2 < np.where(
                rate_tot_neg[reacted_neg_idx] > 0.0,
                rate_single_all[reacted_neg_idx] / rate_tot_neg[reacted_neg_idx],
                0.5,
            )
            charge[reacted_neg_idx[sub_to_single]] = 0
            charge[reacted_neg_idx[~sub_to_single]] = 1

        # H0/D0: stripping to positive ion.
        neu_mask = initial_charge == 0
        t_neu = np.where(
            rate_strip_all > 0.0,
            -np.log1p(-u_time) / np.maximum(rate_strip_all, 1e-300),
            np.inf,
        )
        to_pos_from_neu = neu_mask & (t_neu < dt)
        collision_dt_s[to_pos_from_neu] = t_neu[to_pos_from_neu]
        charge[to_pos_from_neu] = 1

        # H+/D+: charge exchange to neutral.
        pos_mask = initial_charge == 1
        t_pos = np.where(
            rate_cx_all > 0.0,
            -np.log1p(-u_time) / np.maximum(rate_cx_all, 1e-300),
            np.inf,
        )
        to_neu_from_pos = pos_mask & (t_pos < dt)
        collision_dt_s[to_neu_from_pos] = t_pos[to_neu_from_pos]
        charge[to_neu_from_pos] = 0
        return collision_dt_s


def create_reaction_model(config_dict=None):
    """Build a reaction model from config dictionary."""
    cfg = config_dict or {}
    model_type = str(cfg.get("type", "none")).strip().lower()
    density_direction = cfg.get(
        "density_direction",
        cfg.get(
            "DENSITY_DIRECTION",
            cfg.get(
                "main_beam_axis_direction",
                cfg.get(
                    "MAIN_BEAM_AXIS_DIRECTION",
                    cfg.get("Main beam axis direction", cfg.get("density_profile_direction", [1.0, 0.0, 0.0])),
                ),
            ),
        ),
    )

    if model_type in ("none", "off", "null"):
        return NullReactionModel()

    if model_type in ("beam_gas_cross_sections", "beam-gas-cross-sections"):
        return BeamCrossSectionReaction(
            background_density_m3=cfg.get("background_density_m3", 0.0),
            density_profile_file=cfg.get("density_profile_file", None),
            density_profile_scale=cfg.get("density_profile_scale", 1.0),
            density_profile_direction=density_direction,
            fixed_cs=cfg.get("fixed_cs", False),
            manual_cross_sections=cfg.get("manual_cross_sections", None),
        )

    raise ValueError(f"Unknown REACTION_MODEL type: {model_type}")



