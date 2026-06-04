# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`dygen` is a single-purpose image post-processing tool that turns a source `.png` into a stylized "painting." It has no tests, no CLI args, and no package entry point beyond the script — it's a parameter-driven art generator.

## Commands

```bash
pip install -e .        # install package + deps (Pillow, tqdm) in editable mode
python dygen.py         # run the full pipeline; reads ref/<init_file>.png, writes 3 PNGs to out/
```

There is no test suite, linter, or build step. To run on a different image, place `<name>.png` in `ref/` and edit `init_file = "ref"` near the top of `dygen.py` to `init_file = "<name>"`.

## Architecture

Two files do everything:

- **`dg_lib.py`** — defines `DYImage`, a thin immutable-style wrapper around a PIL `Image`. Almost every method returns a *new* `DYImage`, so processing is written as method chains (`.quantized().cleanup().pixilated()`). The internal `__pilimage` is private; use `.pil()`, `.pixels()`, `.size()`, `.palette()` to reach into it. A handful of methods mutate in place and return `self` instead (`scaled`, `resized`) or return `None` (`blend`) — watch for these when chaining. Operations are pure Python double-loops over pixels (no numpy), so runtime scales with image area × palette size and is slow on large images.

- **`dygen.py`** — the pipeline script and the only "configuration." All knobs are module-level constants at the top (`color_count`, `cluster`, `pixel_step`, `glow_amount`, `stencils`, feature flags). `random.seed(2)` makes runs deterministic; brush variation comes from seeded `random` calls inside `DYImage.paint`.

### Pipeline stages (in `dygen.py`, top to bottom)

1. **Quantize** — reduce to `color_count` colors (`quantized`), then `cleanup` (mode filter) to remove speckle, then `pixilated` to enlarge pixel blocks.
2. **Extract masks** — `palette()` lists the surviving colors; `extract()` produces one grayscale mask per color. `rounded` versions (blur + clamp) give looser blob shapes.
3. **Load stencils** — files in `tex/` (e.g. `tex1.png`, `tex2.png`) are tiled to image size via `expanded()` and used as brush textures. Which stencil is applied per color is chosen randomly per stroke.
4. **Paint** — three passes over the palette: flat color blobs → `out/<name>.blobs.v<version>.png`, then loose brushed strokes multiplied with stencils, then (optional) `comp` to reintroduce pieces of the original image, then optional `glow_blur`, saved to `out/<name>.painted.v<version>.png`.
5. **Sharpen** — final `SHARPEN` filter → `out/<name>.sharpened.v<version>.png`.

`version` is appended to output filenames so experimental runs don't overwrite each other.

### Key directories

- `ref/` — input images (and optional `<name>.map.png` when `use_map=True`, used by `cleanup_map` for spatially-varying smoothing).
- `tex/` — stencil/brush textures referenced by the `stencils` list.
- `out/` — generated images.

## Conventions

- New image transforms belong on `DYImage` and should return a new `DYImage` to stay chainable.
- `paint` is the most complex method — it branches on `threshold`, `spread`, `only_add`, and `volume_diff`. Read its branches before modifying; the script relies on specific combinations (e.g. `threshold=False, volume_diff=0.03` for the loose pass).
