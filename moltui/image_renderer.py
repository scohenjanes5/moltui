import math

import numpy as np
from PIL import Image

from .elements import Molecule


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class ImageRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        bg_color: tuple[int, int, int] = (0, 0, 0),
    ):
        self.width = width
        self.height = height
        self.fov = 1.5
        self.light_dir = np.array([0.4, 0.7, -0.6])
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.ambient = 0.25
        self.atom_scale = 0.35
        self.bond_radius = 0.08
        self.bg_color = np.array(bg_color, dtype=np.uint8)
        self.clear()

    def clear(self):
        self.pixels = np.tile(self.bg_color, (self.height, self.width, 1))
        self.z_buf = np.full((self.height, self.width), float("inf"))

    def _project(self, point: np.ndarray) -> tuple[float, float, float]:
        x, y, z = point
        if z <= 0.1:
            return (float("nan"), float("nan"), z)
        scale = min(self.width, self.height) / 2
        px = x * self.fov / z
        py = y * self.fov / z
        sx = self.width / 2 + px * scale
        sy = self.height / 2 - py * scale
        return (sx, sy, z)

    def render_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: tuple[int, int, int],
    ):
        sx, sy, sz = self._project(center)
        if math.isnan(sx):
            return

        scale = min(self.width, self.height) / 2
        pr = radius * self.fov / sz * scale
        if pr < 0.5:
            return

        x_min = max(0, int(sx - pr - 1))
        x_max = min(self.width - 1, int(sx + pr + 1))
        y_min = max(0, int(sy - pr - 1))
        y_max = min(self.height - 1, int(sy + pr + 1))

        if x_min > x_max or y_min > y_max:
            return

        ys = np.arange(y_min, y_max + 1)
        xs = np.arange(x_min, x_max + 1)
        px_grid, py_grid = np.meshgrid(xs, ys)

        dx = (px_grid - sx) / pr
        dy = (py_grid - sy) / pr
        dist_sq = dx * dx + dy * dy

        mask = dist_sq <= 1.0

        dz = np.sqrt(np.maximum(0.0, 1.0 - dist_sq))
        norm_len = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-10
        nx = dx / norm_len
        ny = -dy / norm_len
        nz = dz / norm_len

        diffuse = np.maximum(
            0.0,
            nx * self.light_dir[0] + ny * self.light_dir[1] + nz * self.light_dir[2],
        )
        intensity = np.minimum(1.0, self.ambient + (1.0 - self.ambient) * diffuse)

        point_z = sz - radius * dz

        z_slice = self.z_buf[y_min : y_max + 1, x_min : x_max + 1]
        valid = mask & (point_z < z_slice)

        z_slice[valid] = point_z[valid]

        color_arr = np.array(color, dtype=np.float64)
        for c in range(3):
            channel = self.pixels[y_min : y_max + 1, x_min : x_max + 1, c]
            channel[valid] = np.minimum(
                255, (color_arr[c] * intensity[valid])
            ).astype(np.uint8)

    def render_bond(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        color1: tuple[int, int, int],
        color2: tuple[int, int, int],
    ):
        sx1, sy1, sz1 = self._project(p1)
        sx2, sy2, sz2 = self._project(p2)
        if math.isnan(sx1) or math.isnan(sx2):
            return

        scale = min(self.width, self.height) / 2
        mid_z = (sz1 + sz2) / 2
        pr = self.bond_radius * self.fov / mid_z * scale

        dx = sx2 - sx1
        dy = sy2 - sy1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1.0:
            return

        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux  # perpendicular

        half_w = max(1.0, pr)
        steps = int(length) + 1

        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            cx = sx1 + dx * t
            cy = sy1 + dy * t
            cz = sz1 + (sz2 - sz1) * t
            color = color1 if t < 0.5 else color2

            hw = int(half_w + 1)
            offsets = np.arange(-hw, hw + 1, dtype=np.float64)
            d = offsets / half_w
            valid_mask = np.abs(d) <= 1.0
            offsets = offsets[valid_mask]
            d = d[valid_mask]

            pxs = np.round(cx + nx * offsets).astype(int)
            pys = np.round(cy + ny * offsets).astype(int)

            bounds = (
                (pxs >= 0) & (pxs < self.width) & (pys >= 0) & (pys < self.height)
            )
            pxs, pys, d = pxs[bounds], pys[bounds], d[bounds]

            if len(pxs) == 0:
                continue

            cyl_nz = np.sqrt(1.0 - d * d)
            cyl_nx = nx * d
            cyl_ny = -ny * d

            norm_len = np.sqrt(cyl_nx**2 + cyl_ny**2 + cyl_nz**2) + 1e-10
            cyl_nx /= norm_len
            cyl_ny /= norm_len
            cyl_nz /= norm_len

            diffuse = np.maximum(
                0.0,
                cyl_nx * self.light_dir[0]
                + cyl_ny * self.light_dir[1]
                + cyl_nz * self.light_dir[2],
            )
            intensity = np.minimum(1.0, self.ambient + (1.0 - self.ambient) * diffuse)
            pz = cz - self.bond_radius * cyl_nz

            z_vals = self.z_buf[pys, pxs]
            z_valid = pz < z_vals

            idxs_y = pys[z_valid]
            idxs_x = pxs[z_valid]
            ints = intensity[z_valid]

            self.z_buf[idxs_y, idxs_x] = pz[z_valid]
            for c in range(3):
                self.pixels[idxs_y, idxs_x, c] = np.minimum(
                    255, (color[c] * ints)
                ).astype(np.uint8)

    def render_molecule(
        self,
        molecule: Molecule,
        rot: np.ndarray,
        camera_distance: float,
    ):
        self.clear()
        if not molecule.atoms:
            return

        centroid = molecule.center()
        transformed = []
        for atom in molecule.atoms:
            pos = rot @ (atom.position - centroid)
            pos[2] += camera_distance
            transformed.append(pos)

        for i, j in molecule.bonds:
            self.render_bond(
                transformed[i],
                transformed[j],
                molecule.atoms[i].element.cpk_color,
                molecule.atoms[j].element.cpk_color,
            )

        atom_order = sorted(
            range(len(molecule.atoms)),
            key=lambda idx: -transformed[idx][2],
        )
        for i in atom_order:
            atom = molecule.atoms[i]
            radius = atom.element.covalent_radius * self.atom_scale
            self.render_sphere(transformed[i], radius, atom.element.cpk_color)

    def to_pil_image(self) -> Image.Image:
        return Image.fromarray(self.pixels, "RGB")
