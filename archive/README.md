# Archive

## kitty_viewer.py

Kitty graphics protocol rendering backend, removed from `moltui/app.py` on 2026-04-14.

Used the [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) to send rendered pixel buffers directly to the terminal via temp file + zlib compression. This gave true-pixel rendering but only worked in Kitty-compatible terminals.

The project now uses braille-based text rendering exclusively, which works in any terminal (including over SSH).

The file is self-contained: it includes the `KittyViewer` class and all helper functions (`_kitty_send`, `_kitty_delete`, `_get_terminal_pixel_size`, `_get_temp_path`).

## renderer.py

Rich/Textual Strip-based software renderer, removed on 2026-04-14.

Rendered molecules using Rich `Segment` and Textual `Strip` objects with per-character RGB styling. Replaced by `image_renderer.py` which renders to a pixel buffer that is then converted to braille characters in the Textual widget.

## hq_export_recipe.py

Archived reference implementation of the high-quality PNG export recipe:
- renderer material/camera tuning for PNG output
- artifact-resistant isosurface rasterization settings
- adaptive high-resolution isosurface extraction settings for export

Use this file as a stable "known-good" parameter source if HQ export needs to be rebuilt.

## hq_isosurface_algorithms.py

Archived mesh-processing helpers used in the HQ export pipeline:
- Laplacian mesh smoothing
- vertex normal recomputation from face geometry

## HQ_EXPORT_QUALITY.md

Practical guide for reproducing artifact-free, high-quality PNG exports from MolTUI, including recommended values and troubleshooting notes.
