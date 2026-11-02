"""fits_width compares whole millimetres, so exact fits are met."""

from decimal import Decimal

import pytest

from livingproof_sim_mujoco.scenario import WALL_CLEARANCE, fits_width, millimetres, preconditions


@pytest.mark.parametrize(
    ("width", "clearance", "clear", "met"),
    [
        (1.30, 0.05, 1.40, True),  # exact fit, float arithmetic gives 1.4000000000000001
        (1.20, 0.10, 1.40, True),
        (0.70, 0.05, 0.80, True),
        (1.31, 0.05, 1.40, False),
        (1.30, 0.05, 1.399, False),
        (0.30, 0.10, 0.50, True),
    ],
)
def test_exact_and_near_fits(width, clearance, clear, met):
    assert fits_width(width, clearance, clear) is met


def test_millimetres_round_half_up():
    assert millimetres(1.4) == 1400
    assert millimetres(0.0005) == 1
    assert millimetres(0.0004) == 0
    assert millimetres(1.2345) == 1235


def test_one_centimetre_grid_matches_integer_rule():
    """Every width/clear pair on a 1 cm grid agrees with the exact decimal rule."""
    cases = 0
    for clearance in WALL_CLEARANCE.values():
        for w in range(30, 150):
            for c in range(w - 5, w + 30):
                width, clear = w / 100, c / 100
                exact = Decimal(w) / 100 + 2 * Decimal(str(clearance)) <= Decimal(c) / 100
                assert fits_width(width, clearance, clear) is exact, (width, clearance, clear)
                cases += 1
    assert cases > 360


def _robot(width, cls="humanoid"):
    return {"class": cls, "dimensions": {"width": {"value": width}}, "simulators": ["mujoco"]}


def _conf(name):
    return {"controller": {"interface": ["velocity_command"]}}


def test_exact_fit_precondition_is_met():
    world = {"dimensions": {"clear_width": 1.40}}
    gate = preconditions(_robot(1.30), world, "stock", _conf)
    assert gate[0]["name"] == "fits_width" and gate[0]["met"] is True
    gate = preconditions(_robot(1.31), world, "stock", _conf)
    assert gate[0]["met"] is False
