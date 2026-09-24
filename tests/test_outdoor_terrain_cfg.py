"""Outdoor terrain task: duck-scale uneven ground (Mjlab-Velocity-Outdoor-MicroDuck)."""

import math

import mujoco
import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.outdoor_terrain import (
    MICRODUCK_OUTDOOR_TERRAINS_CFG,
    OUTDOOR_MAX_HEIGHT_M,
    OUTDOOR_MAX_SLOPE_DEG,
    ScaledRandomUniformTerrainCfg,
)

HF_TERRAINS = ("perlin", "gravel", "waves", "stones")


def _empty_terrain_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


def _generate(name, difficulty, seed=0):
    sub = MICRODUCK_OUTDOOR_TERRAINS_CFG.sub_terrains[name]
    sub.size = MICRODUCK_OUTDOOR_TERRAINS_CFG.size
    return sub.function(difficulty, _empty_terrain_spec(), np.random.default_rng(seed))


def _hfield_height(out):
    return out.geometries[0].hfield.size[2]


# ── Terrain generator ─────────────────────────────────────────────────────────

def test_terrain_mix():
    subs = MICRODUCK_OUTDOOR_TERRAINS_CFG.sub_terrains
    assert set(subs) == {"flat", *HF_TERRAINS, "slope"}
    assert math.isclose(sum(s.proportion for s in subs.values()), 1.0, abs_tol=1e-9)


def test_curriculum_mode_ramps_difficulty_by_row():
    # curriculum=False (the TerrainGeneratorCfg default) samples a random
    # difficulty per patch, so terrain_levels could never ramp anything.
    assert MICRODUCK_OUTDOOR_TERRAINS_CFG.curriculum is True
    assert MICRODUCK_OUTDOOR_TERRAINS_CFG.num_rows >= 5


@pytest.mark.parametrize("name", HF_TERRAINS)
def test_heightfields_stay_duck_scale_at_max_difficulty(name):
    # The duck lifts its feet ~1-2 cm: no bump may exceed OUTDOOR_MAX_HEIGHT_M.
    for seed in range(3):
        h = _hfield_height(_generate(name, 1.0, seed))
        assert 0.0 < h <= OUTDOOR_MAX_HEIGHT_M + 1e-6, (name, seed, h)


@pytest.mark.parametrize("name", HF_TERRAINS)
def test_heightfields_get_rougher_with_difficulty(name):
    easy = _hfield_height(_generate(name, 0.0))
    hard = _hfield_height(_generate(name, 1.0))
    assert hard > easy, (name, easy, hard)


def test_slope_capped():
    slope = MICRODUCK_OUTDOOR_TERRAINS_CFG.sub_terrains["slope"]
    assert math.degrees(math.atan(slope.slope_range[1])) <= OUTDOOR_MAX_SLOPE_DEG + 1e-6
    assert slope.inverted is False  # inverted pyramid = pit spawn below the floor


def test_scaled_random_uniform_interpolates_noise_by_difficulty():
    cfg = ScaledRandomUniformTerrainCfg(
        noise_range=(0.0, 0.012), noise_step=0.001, vertical_scale=0.001,
        horizontal_scale=0.05,
    )
    cfg.size = (2.0, 2.0)
    flat = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(0))
    half = cfg.function(0.5, _empty_terrain_spec(), np.random.default_rng(0))
    full = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    assert _hfield_height(half) < _hfield_height(full)
    assert _hfield_height(flat) <= 0.001 + 1e-9  # difficulty 0 → (near) flat
    # cubic spline upsampling overshoots the sampled range, but stays bounded
    assert _hfield_height(full) <= 2 * 0.012
    # The cfg itself must not be mutated (the generator reuses it per patch).
    assert cfg.noise_range == (0.0, 0.012)


# ── Env cfg / registration ────────────────────────────────────────────────────

def test_outdoor_env_cfg_uses_outdoor_generator_and_rough_sim_settings():
    cfg = make_microduck_velocity_env_cfg(outdoor=True)
    assert cfg.scene.terrain.terrain_type == "generator"
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not MICRODUCK_OUTDOOR_TERRAINS_CFG  # own copy, safe to mutate
    assert set(gen.sub_terrains) == set(MICRODUCK_OUTDOOR_TERRAINS_CFG.sub_terrains)
    assert gen.curriculum is True
    assert cfg.sim.nconmax >= 200
    assert cfg.sim.mujoco.iterations >= 30
    assert "terrain_levels" in cfg.curriculum


def test_outdoor_play_cfg_is_small_and_not_curriculum():
    cfg = make_microduck_velocity_env_cfg(play=True, outdoor=True)
    gen = cfg.scene.terrain.terrain_generator
    assert gen.curriculum is False
    assert gen.num_rows == 5 and gen.num_cols == 5
    # play must not mutate the shared module-level training cfg
    assert MICRODUCK_OUTDOOR_TERRAINS_CFG.curriculum is True


def test_rough_task_unchanged():
    cfg = make_microduck_velocity_env_cfg(rough=True)
    assert cfg.scene.terrain.terrain_generator is MICRODUCK_ROUGH_TERRAINS_CFG


def test_rough_and_outdoor_are_exclusive():
    with pytest.raises(ValueError):
        make_microduck_velocity_env_cfg(rough=True, outdoor=True)


def test_outdoor_tasks_registered():
    tasks = set(list_tasks())
    assert "Mjlab-Velocity-Outdoor-MicroDuck" in tasks
    assert "Mjlab-Velocity-Outdoor-Backlash-MicroDuck" in tasks


@pytest.mark.parametrize(
    "task_id", ["Mjlab-Velocity-Outdoor-MicroDuck", "Mjlab-Velocity-Outdoor-Backlash-MicroDuck"]
)
def test_registered_training_cfg_not_shrunk_by_play_cfg(task_id):
    # Registration builds the play cfg too; its 5×5 non-curriculum grid must not
    # leak into the TRAINING cfg through a shared generator object.
    from mjlab.tasks.registry import load_env_cfg
    gen = load_env_cfg(task_id, play=False).scene.terrain.terrain_generator
    assert gen.curriculum is True
    assert gen.num_rows == MICRODUCK_OUTDOOR_TERRAINS_CFG.num_rows == 10


def _foot_scan(cfg):
    return next(s for s in cfg.scene.sensors if s.name == "foot_height_scan")


def test_raised_ring_pattern_lifts_ray_origins_only():
    from mjlab.sensor import RingPatternCfg
    from mjlab_microduck.tasks.outdoor_terrain import RaisedRingPatternCfg
    base = RingPatternCfg.single_ring(radius=0.04, num_samples=2)
    raised = RaisedRingPatternCfg.from_ring(base, z_offset=0.015)
    o0, d0 = base.generate_rays(None, "cpu")
    o1, d1 = raised.generate_rays(None, "cpu")
    assert np.allclose(o1[:, :2].numpy(), o0[:, :2].numpy())
    assert np.allclose(o1[:, 2].numpy(), 0.015)
    assert np.allclose(d1.numpy(), d0.numpy())


def test_outdoor_foot_scan_rays_start_above_the_sole():
    # Heightfields report NO hit for a ray starting at/below their surface (boxes
    # report a backface hit instead), so a stance foot read its ABSOLUTE height
    # (e.g. 33.6 cm on the hill platform) ~50% of steps. Rays must start above
    # contact sink (≤3 mm) + ring-offset slope drop (4 cm × tan 8.5° ≈ 6 mm).
    from mjlab_microduck.tasks.outdoor_terrain import RaisedRingPatternCfg
    scan = _foot_scan(make_microduck_velocity_env_cfg(outdoor=True))
    assert isinstance(scan.pattern, RaisedRingPatternCfg)
    assert scan.pattern.z_offset >= 0.01
    assert scan.ray_alignment == "yaw"  # keeps the lift vertical


def test_flat_and_rough_foot_scan_unchanged():
    from mjlab_microduck.tasks.outdoor_terrain import RaisedRingPatternCfg
    for kw in ({}, {"rough": True}):
        scan = _foot_scan(make_microduck_velocity_env_cfg(**kw))
        assert not isinstance(scan.pattern, RaisedRingPatternCfg), kw


def test_outdoor_backlash_variant_mirrors_base_model():
    from mjlab_microduck.tasks import _BACKLASH_TASKS
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    row = next(r for r in _BACKLASH_TASKS if r[0] == "Mjlab-Velocity-Outdoor-Backlash-MicroDuck")
    assert row[2] == {"outdoor": True}
    assert row[4] is MICRODUCK_WALK_BACKLASH_ROBOT_CFG
