import math

import numpy as np
from PIL import Image

from .elements import Molecule
from .isosurface import IsosurfaceMesh


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

        nx, ny = -dy / length, dx / length  # perpendicular
        half_w = max(1.0, pr)
        steps = int(length) + 1

        # Vectorize: all (step, offset) combinations at once
        hw = int(half_w + 1)
        ts = np.linspace(0, 1, steps + 1)
        offsets = np.arange(-hw, hw + 1, dtype=np.float64)
        d_norm = offsets / half_w
        off_mask = np.abs(d_norm) <= 1.0
        offsets = offsets[off_mask]
        d_norm = d_norm[off_mask]

        # Outer product: (n_steps, n_offsets)
        cxs = sx1 + dx * ts
        cys = sy1 + dy * ts
        czs = sz1 + (sz2 - sz1) * ts

        # All pixel positions: (n_steps, n_offsets)
        all_px = np.round(cxs[:, None] + nx * offsets[None, :]).astype(int)
        all_py = np.round(cys[:, None] + ny * offsets[None, :]).astype(int)

        # Cylinder shading (only varies with offset, not step)
        cyl_nz = np.sqrt(1.0 - d_norm * d_norm)
        cyl_nx_v = nx * d_norm
        cyl_ny_v = -ny * d_norm
        norm_len = np.sqrt(cyl_nx_v**2 + cyl_ny_v**2 + cyl_nz**2) + 1e-10
        cyl_nx_v /= norm_len
        cyl_ny_v /= norm_len
        cyl_nz /= norm_len
        diffuse = np.maximum(
            0.0,
            cyl_nx_v * self.light_dir[0]
            + cyl_ny_v * self.light_dir[1]
            + cyl_nz * self.light_dir[2],
        )
        intensity = np.minimum(1.0, self.ambient + (1.0 - self.ambient) * diffuse)
        # Broadcast to (n_steps, n_offsets)
        pz = czs[:, None] - self.bond_radius * cyl_nz[None, :]
        intensity_2d = np.broadcast_to(intensity[None, :], pz.shape)

        # Flatten everything
        flat_px = all_px.ravel()
        flat_py = all_py.ravel()
        flat_pz = pz.ravel()
        flat_int = intensity_2d.ravel()

        # Color: first half = color1, second half = color2
        n_steps = len(ts)
        n_off = len(offsets)
        step_idx = np.repeat(np.arange(n_steps), n_off)
        half = n_steps // 2
        is_first_half = step_idx <= half

        # Bounds check
        valid = (
            (flat_px >= 0)
            & (flat_px < self.width)
            & (flat_py >= 0)
            & (flat_py < self.height)
        )
        flat_px, flat_py, flat_pz, flat_int, is_first_half = (
            flat_px[valid],
            flat_py[valid],
            flat_pz[valid],
            flat_int[valid],
            is_first_half[valid],
        )

        if len(flat_px) == 0:
            return

        # Z-buffer test
        z_current = self.z_buf[flat_py, flat_px]
        z_pass = flat_pz < z_current
        flat_px = flat_px[z_pass]
        flat_py = flat_py[z_pass]
        flat_pz = flat_pz[z_pass]
        flat_int = flat_int[z_pass]
        is_first_half = is_first_half[z_pass]

        if len(flat_px) == 0:
            return

        self.z_buf[flat_py, flat_px] = flat_pz

        c1 = np.array(color1, dtype=np.float64)
        c2 = np.array(color2, dtype=np.float64)
        colors = np.where(is_first_half[:, None], c1[None, :], c2[None, :])
        shaded = np.minimum(255, colors * flat_int[:, None]).astype(np.uint8)
        self.pixels[flat_py, flat_px] = shaded

    def render_isosurface(
        self,
        mesh: IsosurfaceMesh,
        rot: np.ndarray,
        camera_distance: float,
        centroid: np.ndarray,
    ):
        if len(mesh.faces) == 0:
            return

        # Transform all vertices at once
        centered = mesh.vertices - centroid
        transformed = (rot @ centered.T).T
        transformed[:, 2] += camera_distance

        # Transform normals
        rot_normals = (rot @ mesh.normals.T).T

        # Project all vertices
        scale = min(self.width, self.height) / 2
        valid_z = transformed[:, 2] > 0.1
        projected = np.zeros((len(transformed), 3))
        z = transformed[:, 2]
        safe_z = np.where(valid_z, z, 1.0)
        projected[:, 0] = self.width / 2 + transformed[:, 0] * self.fov / safe_z * scale
        projected[:, 1] = (
            self.height / 2 - transformed[:, 1] * self.fov / safe_z * scale
        )
        projected[:, 2] = z

        # Process each triangle
        color = np.array(mesh.color, dtype=np.float64)
        for face in mesh.faces:
            i0, i1, i2 = face
            if not (valid_z[i0] and valid_z[i1] and valid_z[i2]):
                continue

            s0, s1, s2 = projected[i0], projected[i1], projected[i2]
            n0, n1, n2 = rot_normals[i0], rot_normals[i1], rot_normals[i2]

            # Backface cull using screen-space winding
            edge1x = s1[0] - s0[0]
            edge1y = s1[1] - s0[1]
            edge2x = s2[0] - s0[0]
            edge2y = s2[1] - s0[1]
            cross = edge1x * edge2y - edge1y * edge2x
            if cross <= 0:
                continue

            # Bounding box
            x_min = max(0, int(min(s0[0], s1[0], s2[0])))
            x_max = min(self.width - 1, int(max(s0[0], s1[0], s2[0])) + 1)
            y_min = max(0, int(min(s0[1], s1[1], s2[1])))
            y_max = min(self.height - 1, int(max(s0[1], s1[1], s2[1])) + 1)

            if x_min > x_max or y_min > y_max:
                continue

            # Vectorized barycentric coordinates
            ys = np.arange(y_min, y_max + 1, dtype=np.float64)
            xs = np.arange(x_min, x_max + 1, dtype=np.float64)
            px, py = np.meshgrid(xs, ys)

            d = (s1[1] - s2[1]) * (s0[0] - s2[0]) + (s2[0] - s1[0]) * (
                s0[1] - s2[1]
            )
            if abs(d) < 1e-10:
                continue

            inv_d = 1.0 / d
            w0 = (
                (s1[1] - s2[1]) * (px - s2[0]) + (s2[0] - s1[0]) * (py - s2[1])
            ) * inv_d
            w1 = (
                (s2[1] - s0[1]) * (px - s2[0]) + (s0[0] - s2[0]) * (py - s2[1])
            ) * inv_d
            w2 = 1.0 - w0 - w1

            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue

            # Interpolate z
            tri_z = w0 * s0[2] + w1 * s1[2] + w2 * s2[2]

            # Z-buffer test
            z_slice = self.z_buf[y_min : y_max + 1, x_min : x_max + 1]
            valid = inside & (tri_z < z_slice)
            if not valid.any():
                continue

            # Interpolate normals for smooth shading
            inx = w0 * n0[0] + w1 * n1[0] + w2 * n2[0]
            iny = w0 * n0[1] + w1 * n1[1] + w2 * n2[1]
            inz = w0 * n0[2] + w1 * n1[2] + w2 * n2[2]
            in_len = np.sqrt(inx * inx + iny * iny + inz * inz) + 1e-10
            inx /= in_len
            iny /= in_len
            inz /= in_len

            # Ensure normals face camera (negative z = toward camera)
            facing_away = inz > 0
            inx = np.where(facing_away, -inx, inx)
            iny = np.where(facing_away, -iny, iny)
            inz = np.where(facing_away, -inz, inz)

            diffuse = np.maximum(
                0.0,
                inx * self.light_dir[0]
                + iny * self.light_dir[1]
                + inz * self.light_dir[2],
            )
            intensity = np.minimum(1.0, self.ambient + (1.0 - self.ambient) * diffuse)

            z_slice[valid] = tri_z[valid]
            for c in range(3):
                channel = self.pixels[y_min : y_max + 1, x_min : x_max + 1, c]
                channel[valid] = np.minimum(
                    255, (color[c] * intensity[valid])
                ).astype(np.uint8)

    def render_molecule(
        self,
        molecule: Molecule,
        rot: np.ndarray,
        camera_distance: float,
        isosurfaces: list[IsosurfaceMesh] | None = None,
    ):
        self.clear()
        if not molecule.atoms:
            return

        centroid = molecule.center()

        # Render isosurfaces first (they go behind atoms/bonds via z-buffer)
        if isosurfaces:
            for mesh in isosurfaces:
                self.render_isosurface(mesh, rot, camera_distance, centroid)

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
