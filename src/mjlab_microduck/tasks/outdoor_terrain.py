"""Outdoor uneven-ground terrain mix for Mjlab-Velocity-Outdoor-MicroDuck.

Continuous, duck-scale unevenness (lumpy grass/dirt, gravel, rolling ground,
stones/roots, hills) rather than the box stairs/grids of the Rough task. The
duck lifts its feet only ~1-2 cm, so every heightfield tops out at
OUTDOOR_MAX_HEIGHT_M at difficulty 1 — locked by tests/test_outdoor_terrain_cfg.py,
which measures the GENERATED heightfields, not the nominal params (e.g. the wave
generator's amplitude is half its peak-to-peak height).

curriculum=True: one column per terrain type, difficulty rises along the rows
and the terrain_levels curriculum promotes/demotes envs between rows. (The
TerrainGeneratorCfg default, curriculum=False, samples a random difficulty per
patch, so terrain_levels would have nothing to ramp.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.sensor import RingPatternCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg, TerrainOutput

OUTDOOR_MAX_HEIGHT_M = 0.02
OUTDOOR_MAX_SLOPE_DEG = 8.5

# foot_height_scan rays start this far ABOVE the foot site (heights are still
# measured from the site). A ray starting at/below a heightfield surface gets NO
# hit (boxes/planes return a backface hit, which the sensor maps to 0), and the
# miss fallback is the site's ABSOLUTE z — a stance foot on the hill platform read
# 33.6 cm on ~50% of steps. Must exceed contact sink (≤3 mm measured) + the ring
# rays' slope drop (4 cm × tan 8.5° ≈ 6 mm).
FOOT_SCAN_RAY_LIFT_M = 0.015


@dataclass
class RaisedRingPatternCfg(RingPatternCfg):
    """RingPatternCfg whose ray origins are lifted by z_offset (frame-local z)."""

    z_offset: float = 0.0

    @classmethod
    def from_ring(cls, ring: RingPatternCfg, z_offset: float) -> "RaisedRingPatternCfg":
        return cls(rings=ring.rings, include_center=ring.include_center,
                   direction=ring.direction, z_offset=z_offset)

    def generate_rays(self, mj_model, device):
        offsets, directions = super().generate_rays(mj_model, device)
        offsets = offsets.clone()
        offsets[:, 2] += self.z_offset
        return offsets, directions


@dataclass(kw_only=True)
class ScaledRandomUniformTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
    """HfRandomUniformTerrainCfg whose noise height is interpolated by difficulty.

    mjlab's version ignores difficulty (constant roughness in every curriculum
    row). Here the upper noise bound goes noise_range[0] → noise_range[1] as
    difficulty goes 0 → 1.
    """

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        lo, hi = self.noise_range
        d = float(np.clip(difficulty, 0.0, 1.0))
        scaled = replace(self, noise_range=(lo, lo + d * (hi - lo)))
        scaled.size = self.size
        return terrain_gen.HfRandomUniformTerrainCfg.function(scaled, difficulty, spec, rng)


MICRODUCK_OUTDOOR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=6,  # ignored in curriculum mode (one column per sub-terrain)
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.15),
        # Lumpy grass/dirt: smooth multi-octave noise, ~0.5-1 m features.
        "perlin": terrain_gen.HfPerlinNoiseTerrainCfg(
            proportion=0.25,
            height_range=(0.002, OUTDOOR_MAX_HEIGHT_M),
            horizontal_scale=0.1,
            resolution=0.05,
        ),
        # Gravel / fine texture: random heights on a 10 cm lattice, splined
        # onto a 5 cm grid. The cubic spline overshoots the sampled heights
        # (~1.9× span: 1.0 cm noise → ≤1.9 cm generated, measured over 8 seeds).
        "gravel": ScaledRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.0, 0.010),
            noise_step=0.001,
            downsampled_scale=0.1,
            horizontal_scale=0.05,
            vertical_scale=0.001,
        ),
        # Rolling ground: 4 waves per 8 m patch (~2 m wavelength);
        # peak-to-peak = 2 × amplitude.
        "waves": terrain_gen.HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.0025, OUTDOOR_MAX_HEIGHT_M / 2),
            num_waves=4,
            horizontal_scale=0.1,
            vertical_scale=0.001,
        ),
        # Stones / roots: scattered small blocks on otherwise flat ground.
        # "fixed" = bumps only; the default "choice" also digs −h holes next to
        # +h bumps (a 2h = 3 cm step).
        "stones": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.005, 0.015),
            obstacle_width_range=(0.05, 0.2),
            num_obstacles=60,
            platform_width=1.0,
            horizontal_scale=0.05,
            vertical_scale=0.001,
        ),
        # Hills: pyramid with the platform on TOP (never inverted: pit spawn).
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(math.tan(math.radians(3.0)), math.tan(math.radians(OUTDOOR_MAX_SLOPE_DEG))),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)
