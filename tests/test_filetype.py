#!/usr/bin/env python3
"""Tests for _detect_filetype in app.py."""

from __future__ import annotations

from pathlib import Path

from moltui.app import _detect_filetype


class TestDetectFiletype:
    def test_xyz_detected(self, tmp_path: Path):
        path = tmp_path / "water.xyz"
        path.write_text("3\nwater\nO 0 0 0\nH 0.758 0 0.504\nH -0.758 0 0.504\n")
        assert _detect_filetype(str(path)) == "xyz"

    def test_molden_detected(self, tmp_path: Path):
        path = tmp_path / "sample.molden"
        path.write_text("[MOLDEN FORMAT]\n[Atoms] AU\nH 1 1 0.0 0.0 0.0\n")
        assert _detect_filetype(str(path)) == "molden"

    def test_cube_detected(self, tmp_path: Path):
        path = tmp_path / "sample.cube"
        path.write_text("comment 1\ncomment 2\n2 0.0 0.0 0.0\n")
        assert _detect_filetype(str(path)) == "cube"

    def test_gbw_by_extension(self):
        # .gbw detection is by extension, doesn't read content
        path = Path("fake.gbw")
        assert _detect_filetype(str(path)) == "gbw"

    def test_xyz_content_detection(self, tmp_path: Path):
        """XYZ is detected by first non-empty line being an integer."""
        path = tmp_path / "molecule.txt"
        path.write_text("3\ncomment\nH 0 0 0\nH 1 0 0\nH 0 1 0\n")
        assert _detect_filetype(str(path)) == "xyz"

    def test_molden_content_detection(self, tmp_path: Path):
        """Molden is detected by [Molden Format] marker."""
        path = tmp_path / "molecule.txt"
        path.write_text("[Molden Format]\n[Atoms] AU\n")
        assert _detect_filetype(str(path)) == "molden"
