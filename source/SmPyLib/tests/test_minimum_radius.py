# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and boundary contracts for native minimum-radius bindings."""

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


class TestMinimumRadius(unittest.TestCase):
    def parabola(self):
        # C(t)=(2t-1, (2t-1)^2, 0), with minimum radius 1/2 at t=1/2.
        return sm.create_canonical_curve(
            points=[(-1.0, 1.0, 0.0), (0.0, -1.0, 0.0), (1.0, 1.0, 0.0)],
            knots=[0.0, 1.0], knot_multiplicities=[3, 3],
            weights=[1.0, 1.0, 1.0], degree=2,
        )

    def test_point_solution_and_both_exports(self):
        curve = self.parabola()
        before = sm.evaluate_curve(curve, 0.25)
        results = [fn(curve, 0.0, 1.0) for fn in
                   (sm.minimum_radius_solutions, sm.curves.minimum_radius_solutions)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["status"], 1000)
        self.assertEqual(len(results[0]["solutions"]), 1)
        row = results[0]["solutions"][0]
        self.assertEqual(set(row), {"solution_type", "num_variables", "parameter",
                                    "value", "end_parameter", "end_value"})
        self.assertEqual(row["solution_type"], 1)
        self.assertEqual(row["num_variables"], 1)
        self.assertAlmostEqual(row["parameter"], 0.5, delta=1e-10)
        self.assertAlmostEqual(row["value"], 0.5, delta=1e-12)
        self.assertIsNone(row["end_parameter"])
        self.assertIsNone(row["end_value"])
        self.assertEqual(sm.evaluate_curve(curve, 0.25), before)
        self.assertEqual(curve.parameter_range(), (0.0, 1.0))

    def test_trimmed_interval_minimum_is_at_endpoint(self):
        result = sm.minimum_radius_solutions(self.parabola(), 0.75, 1.0, 1e-10)
        self.assertEqual(len(result["solutions"]), 1)
        row = result["solutions"][0]
        self.assertAlmostEqual(row["parameter"], 0.75, delta=1e-10)
        self.assertAlmostEqual(row["value"], math.sqrt(2.0), delta=1e-12)

    def test_circle_radii(self):
        circle = sm.create_circle((0.0, 0.0, 0.0), 2.0)
        lo, hi = circle.parameter_range()
        result = sm.minimum_radius_solutions(circle, lo, hi)
        self.assertEqual(result["status"], 1000)
        self.assertTrue(result["solutions"])
        for row in result["solutions"]:
            self.assertAlmostEqual(row["value"], 2.0, delta=1e-10)
            self.assertEqual(row["num_variables"], 1)
            if row["solution_type"] == 2:
                self.assertIsNotNone(row["end_parameter"])
                self.assertAlmostEqual(row["end_value"], 2.0, delta=1e-10)
                self.assertLessEqual(lo, row["parameter"])
                self.assertLess(row["parameter"], row["end_parameter"])
                self.assertLessEqual(row["end_parameter"], hi)

    def test_native_range_end_fields(self):
        # Native parameter identity tolerances classify this narrow interval as
        # a range. Preserve its two endpoint records, not a reconstructed point.
        lo, hi = 0.5, 0.5 + 2e-12
        result = sm.minimum_radius_solutions(self.parabola(), lo, hi)
        self.assertEqual(result, {"status": 1000, "solutions": [{
            "solution_type": 2, "num_variables": 1,
            "parameter": lo, "value": 0.5,
            "end_parameter": hi, "end_value": 0.5,
        }]})

    def test_straight_line_preserves_empty_native_result(self):
        line = sm.create_line_segment((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertEqual(sm.minimum_radius_solutions(line, 0.0, 1.0),
                         {"status": 1000, "solutions": []})

    def test_invalid_intervals(self):
        curve = self.parabola()
        for lo, hi in [(0.5, 0.5), (1.0, 0.0),
                       (math.nextafter(0.0, -math.inf), 1.0),
                       (0.0, math.nextafter(1.0, math.inf)),
                       (math.nan, 1.0), (0.0, math.nan),
                       (-math.inf, 1.0), (0.0, math.inf)]:
            with self.subTest(lo=lo, hi=hi):
                with self.assertRaises(ValueError):
                    sm.minimum_radius_solutions(curve, lo, hi)

    def test_invalid_tolerances(self):
        curve = self.parabola()
        for tolerance in (-1.0, math.nan, math.inf, -math.inf):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    sm.minimum_radius_solutions(curve, 0.0, 1.0, tolerance)
        result = sm.minimum_radius_solutions(curve, 0.0, 1.0, 0.0)
        self.assertEqual(result["status"], 1000)
        self.assertTrue(result["solutions"])

    def test_invalid_curve_type(self):
        for curve in (None, object()):
            with self.subTest(curve=curve):
                with self.assertRaises(TypeError):
                    sm.minimum_radius_solutions(curve, 0.0, 1.0)
