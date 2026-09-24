# Pixel view (OpenTTD-style graphics) — design

Goal: an OpenTTD-look view of the city in the web viewer, next to the existing three.js scene.

## Decisions

- New 2D canvas renderer, toggled with the 3D view (toolbar button + `X`), remembered in localStorage. Pixel is the default.
- Sprites are painted procedurally in JS (no binary assets, no GPL art), cached per key.
- Vehicle movement logic moves out of `life.js` into `lifesim.js`, shared by both renderers.
- v1 scope: vehicles, overlays, night + seasons, construction animation.

## Files

- `viewer/pixel.js` — `PixelScene`, same public interface as `CityScene`
  (`setMeta`, `setFrame`, `setOverlay`, `setNight`, `setQuality`, `resetView`, `topView`, `flyTo`, `panBy`,
  `rotateBy`, `zoomBy`, `setAutoRotate`, `cameraFootprint`, `screenshot`, `setHover`, `setSelected`, `highlightRect`,
  `setActive`, `onHover`/`onClick`).
- `viewer/sprites.js` — pixel painter (iso box, roofs, cylinder, windows, outline, dither) and recipes for every tile
  character, level, density, variant; ground tiles (grass, auto-tiled roads, water/shore, lots, fields).
- `viewer/lifesim.js` — renderer-agnostic cars, buses, trains, boats, planes, people, birds.
  `life.js` keeps only the three.js drawing.

## Rendering

- 2:1 dimetric projection, 64×32 px tiles at 1×; light from the top-left (top light, left mid, right dark), 1 px outline.
- Zoom steps ½×, 1×, 2×, 4× with image smoothing off. Rotation in 90° steps; recipes are painted per rotation.
- Ground baked into 16×16-tile chunk canvases per (frame ground key, rotation, zoom-independent), re-baked on change.
- Objects (buildings, trees, vehicles, smoke, scaffolds) depth-sorted and blitted every animation frame; only on-screen.
- Water shimmer by pre-rendered frame cycling; turbines, blinking lights, smoke drifting with `frame.wind`.
- Construction: new / levelled-up tiles go lot → foundations → scaffold → building over ~1.5 s; removals puff dust.
- Night: darkened sprite variants with lit windows, streetlights, headlights. Winter: snow on ground and roofs.
- Overlays: ground tinted with `OVERLAY_RAMPS`, buildings drawn translucent; transit layer draws bus/metro routes.
- Picking: sprite bounding boxes front-to-back, then inverse-projected ground diamond.

## Testing

Viewer-only change: Python suite and benchmark numbers unaffected. Playwright screenshots of `demo_baseline_s0` and a
region run in all rotations, zooms, several overlays, night, winter; check the 3D view still works after the
`lifesim` split.
