# High-quality PNG export recipe (artifact-free)

This document captures the settings that produced the cleanest PNG exports for orbital isosurfaces.

## What changed

1. **HQ export path** renders to a dedicated offscreen image, not braille cells.
2. **Isosurface extraction for export** is higher resolution than interactive mode.
3. **Watertight triangle rasterization** avoids cracks/holes on the surface.
4. **Post-extraction smoothing + normal recomputation** removes faceting artifacts.

## Renderer settings (PNG export)

Use these `ImageRenderer` kwargs for export:

| Setting | Value |
|---|---|
| `fov` | `1.5 * 2.5` |
| `ambient` | `0.31` |
| `diffuse_strength` | `0.72` |
| `specular_strength` | `0.42` |
| `shininess` | `96.0` |
| `light_dir` | `(0.58, 0.56, -0.60)` |

And camera flattening:

- `camera_distance_export = camera_distance_view * 2.5`

## Isosurface extraction settings (export only)

For `extract_isosurfaces(...)`:

- `smooth_sigma = 0.5`
- `mesh_smooth_alpha = 0.35`
- Adaptive:
  - if `prod(cube_data.n_points) <= 1_000_000`: `upsample=4`, `mesh_smooth_iters=3`
  - else: `upsample=3`, `mesh_smooth_iters=2`

## Rasterization requirements to avoid seams

In triangle rasterization:

1. Sample barycentric coordinates at **pixel centers** (`x+0.5`, `y+0.5`)
2. Use epsilon edge inclusion:
   - `inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)`
3. Do **not** backface-cull isosurface triangles (marching-cubes winding can be locally inconsistent)

## Export image sizing

- Output long edge: `3200`
- Render scale: `2`
- Max render pixels: `24_000_000`
- Final resample: `LANCZOS`

## Quick integration checklist

1. Build export-only isosurfaces with adaptive HQ parameters.
2. Render with export renderer profile above.
3. Save PNG from offscreen RGB buffer (not terminal braille buffer).
4. If artifacts appear, increase `upsample` first, then `mesh_smooth_iters`.

## Archived code

See `archive/hq_export_recipe.py` for the parameterized, reusable profile definitions.
