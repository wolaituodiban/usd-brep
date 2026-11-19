# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for one-sided native curve curvature evaluation."""

import math
import os
from pathlib import Path
import sys
import unittest

# Match the existing package tests: Windows needs explicit DLL directories
# for OpenUSD in a source build; changing PATH alone is insufficient.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _config = os.environ.get("BREP_TEST_CONFIG", "release")
    _usd = Path(__file__).resolve().parents[3] / "_build" / "target-deps" / "usd" / _config
    _dll_directories = [os.add_dll_directory(str(path))
                        for path in (_usd / "lib", _usd / "bin") if path.is_dir()]

import usd_brep as sm


class TestEvaluateCurveCurvature(unittest.TestCase):
    @staticmethod
    def repeated_knot_curve():
        # The knot at 0 joins a straight quadratic span to a curved one.
        return sm.create_canonical_curve(
            points=[
                (-1.0, 0.0, 0.0),
                (-0.5, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.25, 0.0),
            ],
            knots=[-1.0, 0.0, 1.0],
            knot_multiplicities=[3, 2, 3],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0],
            degree=2,
        )

    def test_repeated_knot_uses_requested_side_and_exports_both_aliases(self):
        curve = self.repeated_knot_curve()

        left = sm.evaluate_curve_curvature(curve, 0.0, "left")
        right = sm.curves.evaluate_curve_curvature(curve, 0.0, "right")

        self.assertAlmostEqual(left, 0.0, delta=1e-12)
        self.assertAlmostEqual(right, 0.5, delta=1e-9)
        self.assertEqual(
            sm.curves.evaluate_curve_curvature(curve, 0.0, "left"), left
        )
        self.assertEqual(
            sm.evaluate_curve_curvature(curve, 0.0, "right"), right
        )

    def test_circle_curvature_at_inward_endpoint_side(self):
        circle = sm.create_circle((0.0, 0.0, 0.0), 2.0)
        start, end = circle.parameter_range()

        self.assertAlmostEqual(
            sm.evaluate_curve_curvature(circle, start, "right"), 0.5, delta=1e-10
        )
        self.assertAlmostEqual(
            sm.curves.evaluate_curve_curvature(circle, end, "left"), 0.5, delta=1e-10
        )

    def test_invalid_parameters_and_side(self):
        curve = self.repeated_knot_curve()
        for parameter in (
            math.nan,
            math.inf,
            -math.inf,
            math.nextafter(curve.parameter_range()[0], -math.inf),
            math.nextafter(curve.parameter_range()[1], math.inf),
        ):
            with self.subTest(parameter=parameter):
                with self.assertRaises(ValueError):
                    sm.evaluate_curve_curvature(curve, parameter, "left")

        with self.assertRaises(ValueError):
            sm.evaluate_curve_curvature(curve, 0.0, "middle")

    def test_invalid_curve_and_required_side(self):
        for curve in (None, object()):
            with self.subTest(curve=curve):
                with self.assertRaises(TypeError):
                    sm.evaluate_curve_curvature(curve, 0.0, "left")

        with self.assertRaises(TypeError):
            sm.evaluate_curve_curvature(self.repeated_knot_curve(), 0.0)
