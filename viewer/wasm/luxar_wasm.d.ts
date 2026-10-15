/* tslint:disable */
/* eslint-disable */

/**
 * Calculate effective radii for nD points when sliced.
 *
 * For each point, computes the effective radius visible in the 3D slice
 * based on the point's distance from the slice plane in hidden dimensions.
 *
 * # Arguments
 * * `positions` - Point positions [numPoints * ndim] (flattened nD coordinates)
 * * `radii` - Original point radii [numPoints]
 * * `display_dims` - Dimensions to display (typically first 3) [numDisplayDims]
 * * `slice_position` - Current slice position [ndim]
 * * `spatial_extend_dims` - Which dims are spatial (1) vs discrete (0) [ndim]
 * * `ndim` - Total number of dimensions (max 16)
 * * `num_points` - Number of points
 * * `output` - Output effective radii [numPoints]
 *
 * # Returns
 * Number of points with non-zero effective radius (visible points)
 *
 * # Algorithm
 * For each point:
 * 1. Check discrete dimension match (exact or within tolerance)
 * 2. Calculate squared distance in non-displayed spatial dimensions
 * 3. Apply Pythagorean theorem: R_effective = sqrt(R² - D²)
 * 4. Set R_effective = 0 if point is outside original radius
 *
 * # Panics
 * Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
 */
export function calculate_effective_radii(positions: Float32Array, radii: Float32Array, display_dims: Uint32Array, slice_position: Float32Array, spatial_extend_dims: Uint8Array, ndim: number, num_points: number, output: Float32Array): number;

/**
 * Calculate 3D segment lengths for visible segments.
 *
 * # Arguments
 * - `start_positions`: Clipped start positions [visibleCount * 3]
 * - `end_positions`: Clipped end positions [visibleCount * 3]
 * - `visible_count`: Number of visible segments
 * - `output`: Output segment lengths [visibleCount]
 */
export function calculate_segment_lengths(start_positions: Float32Array, end_positions: Float32Array, visible_count: number, output: Float32Array): void;

/**
 * Clip a single segment to the nD slice and return interpolation parameters.
 *
 * Returns (visible, t1, t2) where:
 * - visible: true if any part of segment intersects slice
 * - t1: interpolation parameter for clipped start (0.0 = original start)
 * - t2: interpolation parameter for clipped end (1.0 = original end)
 *
 * The clipped points can be computed as:
 * - clipped_p1 = p1 + t1 * (p2 - p1)
 * - clipped_p2 = p1 + t2 * (p2 - p1)
 */
export function clip_segment_single(p1: Float32Array, p2: Float32Array, slice_position: Float32Array, tolerance: Float32Array, display_dims: Uint32Array, ndim: number): Float32Array;

/**
 * Batch clip all segments and output visibility mask and interpolation parameters.
 *
 * This is the main workhorse function - processes all segments in one WASM call.
 *
 * # Arguments
 * - `positions`: Vertex positions [numVertices * ndim]
 * - `segments`: Segment indices [numSegments * 2]
 * - `slice_position`: Current slice position [ndim]
 * - `tolerance`: Per-dimension tolerance [ndim]
 * - `display_dims`: Which dimensions to display [numDisplayDims]
 * - `ndim`: Number of dimensions
 * - `num_segments`: Number of segments
 * - `output_visibility`: Output visibility mask [numSegments]
 * - `output_t1`: Output t1 parameters [numSegments]
 * - `output_t2`: Output t2 parameters [numSegments]
 *
 * # Returns
 * Number of visible segments
 */
export function clip_segments_batch(positions: Float32Array, segments: Uint32Array, slice_position: Float32Array, tolerance: Float32Array, display_dims: Uint32Array, ndim: number, num_segments: number, output_visibility: Uint8Array, output_t1: Float32Array, output_t2: Float32Array): number;

/**
 * Compact `faces` to those whose three vertices are all visible.
 *
 * Writes **original (un-remapped)** vertex indices into `output` — see the
 * "No vertex compaction" note in the module docs.
 *
 * # Winding
 *
 * The authored per-face index order is preserved exactly, so this kernel is
 * winding-agnostic. Restoring front-facing winding under a reflected display
 * permutation is a separate post-pass owned by the caller (§5.4).
 *
 * # Out-of-range indices
 *
 * A face index `>= vertex_mask.len()` causes the whole face to be **dropped**
 * rather than indexing out of bounds.
 *
 * This guard exists because the values come from the *store*, not from the
 * caller: the viewer loads arbitrary (externally produced, possibly corrupted)
 * datasets, and an out-of-bounds slice read here would not be a node-scoped
 * error — the crate is `panic = "abort"`, so the trap escapes as an opaque,
 * uncatchable `RuntimeError: unreachable`, and the trapped instance carries no
 * state guarantees afterwards. The TypeScript backend would instead read
 * `undefined` and silently diverge. Dropping the face keeps both backends in
 * parity and fails safe.
 *
 * The loader validates face indices up front and fails the node with a
 * `LoaderError` before reaching here (§3.5 Stage 2), so on the sanctioned path
 * this guard is unreachable. It is defense in depth for a `panic = "abort"`
 * blast radius, not a substitute for that gate.
 *
 * Note the asymmetry with the `debug_assert`s above, which is deliberate:
 * store-supplied *values* are range-checked in release builds, while *shape*
 * parameters (`num_faces`, buffer sizes) the caller computed are trusted, as
 * they were in every sibling kernel (e.g. the former `compact_by_mask` /
 * `count_visible`).
 *
 * # Arguments
 * * `faces` - Triangle vertex indices [num_faces * 3]
 * * `vertex_mask` - Per-vertex visibility from
 *   [`mesh_vertex_visibility_mask`]; its **length defines the valid vertex
 *   range**, so pass a view sized exactly `num_vertices` rather than a larger
 *   reused scratch buffer
 * * `num_faces` - Number of triangles
 * * `output` - Output indices [num_faces * 3] worst case
 *
 * # Returns
 *
 * Number of visible faces written. **Slice `output` to `3 ×` this before use.**
 * Everything past that point is left untouched — deliberately, to avoid a second
 * pass over the buffer — so on a reused buffer it holds the *previous* frame's
 * indices, and on a fresh one it holds zeros. Uploading the whole buffer as an
 * index range therefore draws stale or degenerate triangles rather than
 * nothing, which is the failure this return value exists to prevent.
 */
export function compact_visible_faces(faces: Uint32Array, vertex_mask: Uint8Array, num_faces: number, output: Uint32Array): number;

/**
 * Per-endpoint **joint code**: how the line shader should treat this endpoint,
 * and — when it is an ordinary two-segment joint — which segment it joins.
 *
 * One f32 per endpoint, in the two texel slots that used to carry the cap
 * suppression scalar:
 *
 * | value          | meaning                                                     |
 * |----------------|-------------------------------------------------------------|
 * | ` 0`           | free polyline end — keep the soft cap                        |
 * | `-1`           | slice-clipped — the real endpoint is outside the slice, so no |
 * |                | neighbour will ever arrive; suppress the cap entirely        |
 * | `-2`           | degree->=3 hub — several quads already stack here            |
 * | `+(slot + 1)`  | joins visible segment `slot`, at that segment's START        |
 * | `-(slot + 3)`  | joins visible segment `slot`, at that segment's END          |
 *
 * `slot` is the segment's index in the VISIBLE (output) stream, i.e. its
 * storage slot in the line texture. The shader's own `aSortedIndex` maps a
 * DRAW slot to a storage slot, so a storage-space partner reference is read
 * directly and needs no adjustment when the depth-sort worker permutes draw
 * order.
 *
 * # Why a code and not an angle
 *
 * The previous kernel dereferenced the partner only to take one dot product of
 * the two segment directions, and stored `cos(theta)` as a cap-suppression
 * scalar. Storing the partner instead lets the vertex stage do that itself,
 * which buys three things:
 *
 * - The bend angle is measured in SCREEN space, per frame, so it tracks the
 *   camera. The stored angle could not: a gentle 3D bend that projected sharp
 *   kept suppression near 1 while the quads genuinely overlapped (issue #795).
 * - The vertex stage can build real join geometry (a screen-space miter),
 *   which closes the uncovered wedge outside every bend (issue #790) — a
 *   scalar multiplier never could, because there are no fragments there.
 * - This kernel loses its direction table entirely: no per-segment normalize,
 *   no `dirs` allocation (12 B per visible segment — ~32 MB on the largest
 *   bundled lines scene, on wasm linear memory that is never returned to the
 *   OS), and no f64-vs-f32 care to keep the two backends bit-identical, since
 *   integer index arithmetic agrees trivially.
 *
 * The fallback the shader derives is exactly the old value:
 * `clamp(-dot(away_a, away_b), 0, 1)` with each `away` pointing from the
 * shared vertex back along its segment reduces to
 * `clamp(dot(dir_mine, dir_partner), 0, 1)` — the same dot product the miter
 * limit needs anyway.
 *
 * # Joint recognition
 *
 * Only **degree-2** vertices become joints; a hub keeps the cap. A joint is
 * recognised only between two endpoints that both actually *reach* the shared
 * vertex, so a culled or slice-trimmed neighbour does not anchor one.
 *
 * Joints are matched by vertex **index**, not by position. A chain whose
 * segments each carry their own duplicate copy of the shared point (what
 * `line_type="segments"` emits for abutting segments) has no shared index, so
 * it keeps the cap at every joint. That is deliberate: a shared index means
 * "the same vertex of the same polyline", whereas position matching would also
 * fuse two unrelated lines that merely touch. Author connected geometry as
 * `line_type="polyline"` (or reuse indices) to get continuous joints.
 *
 * # Cost
 *
 * Two vertex-indexed tables, 5 B per source vertex (`code_sum` + `degree`),
 * and no per-segment allocation at all.
 *
 * # Arguments
 * - `segments`: Vertex index pairs [numSegments * 2]
 * - `visibility`: Visibility mask [numSegments]
 * - `t1_params`: Start interpolation parameters [numSegments]
 * - `t2_params`: End interpolation parameters [numSegments]
 * - `num_segments`: Total number of segments
 * - `num_vertices`: Total number of source vertices (bounds the touch tables)
 * - `output_start`: Output start joint codes [visibleCount]
 * - `output_end`: Output end joint codes [visibleCount]
 *
 * # Returns
 * Number of visible segments written
 */
export function compute_joint_codes(segments: Uint32Array, visibility: Uint8Array, t1_params: Float32Array, t2_params: Float32Array, num_segments: number, num_vertices: number, output_start: Float32Array, output_end: Float32Array): number;

/**
 * Broadcast a value to all points.
 *
 * Strict contract (mirrors `wasm/typescript/decode.ts::decode_broadcasted`):
 * `value.len()` must be either `1` (a scalar broadcast to every slot) or
 * exactly `elements_per_point` (a per-element vector replicated across every
 * point). Any other length is INVALID and rejected upstream by the worker
 * wrapper (`workers/data-worker/decode/broadcasted.ts`) with a catchable error
 * before this kernel is reached. The old "mixed broadcast" pad/truncate
 * fallback (e.g. `value=[v0,v1], epp=3 -> [v0,v1,v0]`) was removed because it
 * silently diverged from the TS reference.
 *
 * # Arguments
 * * `value` - `1` value (scalar) or `elements_per_point` values (vector)
 * * `num_points` - Number of points
 * * `elements_per_point` - Elements per point (1 for scalar, 3 for vec3, etc.)
 * * `output` - Output buffer [num_points * elements_per_point]
 */
export function decode_broadcasted(value: Float32Array, num_points: number, elements_per_point: number, output: Float32Array): void;

/**
 * Decode per-channel TRUE-log uint16 codes to float32.
 */
export function decode_geolog_perchannel_u16(data: Uint16Array, col_lo: Float64Array, col_hi: Float64Array, col_offset: number, output: Float32Array): void;

/**
 * Decode per-channel TRUE-log uint8 codes to float32 (`x = exp(y)`).
 *
 * The HDR-color encoding: per-column min/max-anchored geometric grid in
 * log space (`col_lo`/`col_hi` are ln(min+)/ln(max+)), uniform relative
 * precision across each column's dynamic range. The reserved zero level is
 * part of this encoding's NAME contract (no legacy variant), so there is
 * no `zero_level` parameter — it is always on.
 */
export function decode_geolog_perchannel_u8(data: Uint8Array, col_lo: Float64Array, col_hi: Float64Array, col_offset: number, output: Float32Array): void;

/**
 * Decode geometric-log quantized uint16 data to float32.
 */
export function decode_geolog_scalar_u16(data: Uint16Array, min_log: number, max_log: number, output: Float32Array): void;

/**
 * Decode geometric-log quantized uint8 data to float32.
 *
 * Min/max-anchored true-log quantization with a RESERVED ZERO LEVEL:
 * level 0 decodes to exactly 0.0; levels [1, 255] decode to
 * `exp(min_log + (u - 1)/254 * (max_log - min_log))` — uniform relative
 * precision over the array's own nonzero range. Mirrors the TS reference
 * bit-for-bit; Python's f64 decoder can differ because its anchors are not
 * rounded to f32 first.
 */
export function decode_geolog_scalar_u8(data: Uint8Array, min_log: number, max_log: number, output: Float32Array): void;

/**
 * Decode per-channel LINEAR (fixed-point) uint16 codes to float32.
 */
export function decode_linear_perchannel_u16(data: Uint16Array, col_lo: Float64Array, col_hi: Float64Array, col_offset: number, output: Float32Array): void;

/**
 * Decode per-channel LINEAR (fixed-point) uint8 codes to float32.
 *
 * Identity transform with per-column `[lo, hi]` scales — the COORDINATE
 * encoding (positions / centers / vertices). No reserved zero level (mirrors
 * the Python `_decode_linear_perchannel`, which has no `zero_level` branch).
 */
export function decode_linear_perchannel_u8(data: Uint8Array, col_lo: Float64Array, col_hi: Float64Array, col_offset: number, output: Float32Array): void;

/**
 * Decode per-channel LOG uint16 codes to float32.
 */
export function decode_log_perchannel_u16(data: Uint16Array, col_lo: Float64Array, col_hi: Float64Array, zero_level: boolean, col_offset: number, output: Float32Array): void;

/**
 * Decode per-channel LOG uint8 codes to float32 (`x = expm1(y)`).
 *
 * The Cholesky-diagonal encoding. `zero_level: true` (current writer) =
 * reserved zero code + nonzero codes over the nonzero-anchored scale;
 * `false` = legacy all-levels mapping.
 */
export function decode_log_perchannel_u8(data: Uint8Array, col_lo: Float64Array, col_hi: Float64Array, zero_level: boolean, col_offset: number, output: Float32Array): void;

/**
 * Decode log-space quantized uint16 data to float32.
 */
export function decode_log_scalar_u16(data: Uint16Array, max_log: number, output: Float32Array): void;

/**
 * Decode log-space quantized uint8 data to float32.
 *
 * Used for positive scalars with wide dynamic range (e.g., radii).
 * Decoding: expm1(normalized * max_log)
 *
 * # Arguments
 * * `data` - Quantized uint8 values
 * * `max_log` - Maximum log value (from encoding)
 * * `output` - Output float32 buffer
 */
export function decode_log_scalar_u8(data: Uint8Array, max_log: number, output: Float32Array): void;

/**
 * Decode LUT-encoded uint16 indices to float32 (row mode).
 */
export function decode_lut_row_u16(indices: Uint16Array, lut: Float32Array, k: number, output: Float32Array): void;

/**
 * Decode LUT-encoded uint8 indices to float32 (row mode).
 *
 * Each index maps to k consecutive float values from the LUT.
 * Used for vector attributes like positions (xyz) or colors (rgb).
 *
 * # Arguments
 * * `indices` - uint8 indices into the LUT
 * * `lut` - Lookup table of float values [numEntries * k]
 * * `k` - Number of values per LUT entry
 * * `output` - Output float32 buffer [indices.len() * k]
 */
export function decode_lut_row_u8(indices: Uint8Array, lut: Float32Array, k: number, output: Float32Array): void;

/**
 * Decode LUT-encoded uint16 indices to float32 (scalar mode).
 */
export function decode_lut_scalar_u16(indices: Uint16Array, lut: Float32Array, output: Float32Array): void;

/**
 * Decode LUT-encoded uint8 indices to float32 (scalar mode).
 *
 * Each index maps to a single float value from the LUT.
 *
 * # Arguments
 * * `indices` - uint8 indices into the LUT
 * * `lut` - Lookup table of float values
 * * `output` - Output float32 buffer (same length as indices)
 */
export function decode_lut_scalar_u8(indices: Uint8Array, lut: Float32Array, output: Float32Array): void;

/**
 * Decode quantized uint16 data to float32.
 *
 * Maps uint16 [0, 65535] to [min_val, max_val] linearly.
 *
 * # Arguments
 * * `data` - Quantized uint16 values
 * * `min_val` - Minimum output value
 * * `max_val` - Maximum output value
 * * `output` - Output float32 buffer (same length as data)
 */
export function decode_quantized_u16(data: Uint16Array, min_val: number, max_val: number, output: Float32Array): void;

/**
 * Decode quantized uint8 data to float32.
 *
 * Maps uint8 [0, 255] to [min_val, max_val] linearly.
 *
 * # Arguments
 * * `data` - Quantized uint8 values
 * * `min_val` - Minimum output value
 * * `max_val` - Maximum output value
 * * `output` - Output float32 buffer (same length as data)
 */
export function decode_quantized_u8(data: Uint8Array, min_val: number, max_val: number, output: Float32Array): void;

/**
 * Decode per-channel SIGNED-LOG uint16 codes to float32.
 */
export function decode_signed_log_perchannel_u16(data: Uint16Array, col_lo: Float64Array, col_hi: Float64Array, zero_level: boolean, col_offset: number, output: Float32Array): void;

/**
 * Decode per-channel SIGNED-LOG uint8 codes to float32
 * (`y = ...; x = sign(y)·expm1(|y|)`).
 *
 * The Cholesky off-diagonal encoding (signed, zero-centred). `zero_level`
 * as in [`decode_log_perchannel_u8`].
 */
export function decode_signed_log_perchannel_u8(data: Uint8Array, col_lo: Float64Array, col_hi: Float64Array, zero_level: boolean, col_offset: number, output: Float32Array): void;

/**
 * Extract 3D positions from nD positions using display dimension indices.
 *
 * # Arguments
 * * `positions_nd` - Input nD positions [numPoints * ndim]
 * * `display_dims` - Which dimensions to display as X,Y,Z [3 or fewer]
 * * `ndim` - Total number of dimensions
 * * `num_points` - Number of points
 * * `output` - Output 3D positions [numPoints * 3]
 *
 * # Example
 * For 5D data with display_dims=[0,2,4]:
 * - X = position[dim 0]
 * - Y = position[dim 2]
 * - Z = position[dim 4]
 */
export function extract_3d_positions(positions_nd: Float32Array, display_dims: Uint32Array, ndim: number, num_points: number, output: Float32Array): void;

/**
 * Interpolate clipped positions to 3D display space.
 *
 * For visible segments, computes clipped 3D positions using t1/t2 parameters.
 *
 * # Arguments
 * - `positions`: Vertex positions [numVertices * ndim]
 * - `segments`: Segment indices [numSegments * 2]
 * - `visibility`: Visibility mask [numSegments]
 * - `t1`: Start interpolation parameters [numSegments]
 * - `t2`: End interpolation parameters [numSegments]
 * - `display_dims`: Which dimensions to display [3]
 * - `ndim`: Number of dimensions
 * - `num_segments`: Total number of segments
 * - `output_start`: Output start positions [visibleCount * 3]
 * - `output_end`: Output end positions [visibleCount * 3]
 *
 * # Returns
 * Number of visible segments written
 */
export function interpolate_clipped_positions(positions: Float32Array, segments: Uint32Array, visibility: Uint8Array, t1_params: Float32Array, t2_params: Float32Array, display_dims: Uint32Array, ndim: number, num_segments: number, output_start: Float32Array, output_end: Float32Array): number;

/**
 * Batch interpolate RGB color attributes for visible segments.
 *
 * # Arguments
 * - `colors`: Per-vertex RGB colors [numVertices * 3]
 * - `segments`: Segment indices [numSegments * 2]
 * - `visibility`: Visibility mask [numSegments]
 * - `t1`: Start interpolation parameters [numSegments]
 * - `t2`: End interpolation parameters [numSegments]
 * - `num_segments`: Total number of segments
 * - `output_start`: Output interpolated start colors [visibleCount * 3]
 * - `output_end`: Output interpolated end colors [visibleCount * 3]
 *
 * # Returns
 * Number of visible segments written
 */
export function interpolate_colors_batch(colors: Float32Array, segments: Uint32Array, visibility: Uint8Array, t1_params: Float32Array, t2_params: Float32Array, num_segments: number, output_start: Float32Array, output_end: Float32Array): number;

/**
 * Batch interpolate scalar attributes for visible segments.
 *
 * Interpolates values using t1/t2 parameters and compacts to visible-only output.
 *
 * # Arguments
 * - `values`: Per-vertex attribute values [numVertices]
 * - `segments`: Segment indices [numSegments * 2]
 * - `visibility`: Visibility mask [numSegments]
 * - `t1`: Start interpolation parameters [numSegments]
 * - `t2`: End interpolation parameters [numSegments]
 * - `num_segments`: Total number of segments
 * - `output_start`: Output interpolated start values [visibleCount]
 * - `output_end`: Output interpolated end values [visibleCount]
 *
 * # Returns
 * Number of visible segments written
 */
export function interpolate_scalars_batch(values: Float32Array, segments: Uint32Array, visibility: Uint8Array, t1_params: Float32Array, t2_params: Float32Array, num_segments: number, output_start: Float32Array, output_end: Float32Array): number;

/**
 * Compute Mahalanobis distance for a single point using packed Cholesky factor.
 *
 * Given L (lower-triangular Cholesky of covariance),
 * Mahalanobis distance = ||L⁻¹ · (x - μ)||
 *
 * We use forward substitution to solve L·y = d, then ||y|| is the Mahalanobis distance.
 *
 * # Arguments
 * * `diff` - Difference vector (x - μ) for the dimensions [ndim]
 * * `packed_l` - Packed Cholesky factor [packedSize]
 * * `ndim` - Dimensionality of the Cholesky (max 16)
 *
 * # Returns
 * Mahalanobis distance
 *
 * # Panics
 * Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
 *
 * # Optimization Notes
 * - Direct division (bit-consistent with the TS reference)
 * - Hoisted epsilon constant
 * - Loop fusion for norm computation
 */
export function mahalanobis_distance(diff: Float32Array, packed_l: Float32Array, ndim: number): number;

/**
 * Compute per-vertex nD slab membership.
 *
 * For each non-displayed ("hidden") dimension `d`, with
 * `slice_min = slice_position[d] - tolerance[d]` and
 * `slice_max = slice_position[d] + tolerance[d]`, a vertex is **in** iff
 * `v[d] >= slice_min && v[d] <= slice_max` for *every* such `d`. Displayed
 * dimensions are skipped entirely (they are what gets rendered).
 *
 * This is precisely the `p1_in` branch of
 * [`crate::lines_clipping::clip_segment_single`], applied per vertex.
 *
 * # Non-finite coordinates
 *
 * A `NaN` or `±Inf` coordinate on any hidden dimension makes the vertex
 * **invisible** — the #806 rule, enforced identically in both lines backends
 * and now both mesh backends.
 *
 * The explicit `is_finite` test is load-bearing, not decorative. `NaN` fails
 * both comparisons and `±Inf` fails against any finite bound, so the bare
 * range test looks sufficient — but a caller passing a genuinely infinite
 * `tolerance` (rather than the finite `EXTEND_TO_ALL_TOLERANCE` sentinel,
 * `1e10`) makes `slice_max = +Inf`, and `+Inf <= +Inf` is **true**. Without
 * this test an infinite coordinate would then be reported visible.
 *
 * The rule covers the *coordinate* only, and the slab parameters behave the
 * OPPOSITE way — worth knowing before reading a surprising screen:
 *
 * - a `NaN` in `slice_position` or `tolerance` makes every **finite** vertex
 *   visible, because `value < NaN` and `value > NaN` are both false, so the
 *   slab test degenerates to "not non-finite". It fails OPEN, not closed.
 * - a **negative** `tolerance` inverts the slab (`min > max`) and culls
 *   everything.
 *
 * Both are caller bugs — `slice_position` and `tolerance` are viewer-computed,
 * not store-supplied — and both backends agree exactly (verified across the
 * full non-finite matrix), so neither is guarded here. Callers deriving a
 * tolerance from possibly-absent dimension metadata should not let a `NaN`
 * reach this kernel expecting it to be culled.
 *
 * # Arguments
 * * `positions` - Vertex positions [num_vertices * ndim]
 * * `slice_position` - Current slice position [ndim]
 * * `tolerance` - Per-dimension tolerance [ndim]
 * * `display_dims` - Which dimensions are displayed [num_display_dims]
 * * `ndim` - Number of dimensions
 * * `num_vertices` - Number of vertices
 * * `output` - Output visibility mask [num_vertices] (1 = in, 0 = out)
 *
 * # Returns
 * Number of visible vertices.
 *
 * # Panics
 * Via `validate_ndim` when `ndim > 16`. Callers must route above that to the
 * TypeScript backend (`pickBackend` does this transparently).
 */
export function mesh_vertex_visibility_mask(positions: Float32Array, slice_position: Float32Array, tolerance: Float32Array, display_dims: Uint32Array, ndim: number, num_vertices: number, output: Uint8Array): number;

/**
 * Fused nD→3D GSplat projection in a SINGLE pass over the splats.
 *
 * Collapses the previous 6-call worker pipeline
 * (`compute_gsplats_attenuation` + `extract_3d_positions` + `compact_by_mask`
 * for centers + `extract_visible_cholesky_3d` + `compact_attenuated_amplitudes`
 * + `compact_by_mask` for colors) into one kernel, eliminating ~5 full passes
 * over `splat_count` and the repeated copies of the large `positions` /
 * `cholesky` arrays across the wasm-bindgen boundary.
 *
 * For each splat it: (1) applies the precomputed discrete-visibility gate,
 * (2) computes continuous attenuation (marginal Cholesky + shifted Gaussian),
 * (3) decides visibility via `amplitude * attenuation >= min_amplitude`, and
 * (4) writes COMPACTED outputs (visible centers3D, cholesky3D[6], attenuated
 * amplitudes, colors) densely from index 0. The compaction stride and
 * visibility gating are exercised by `test_fused_compaction_stride`.
 *
 * Colors are coerced to normalized f32 on the TS side (wasm-bindgen can't take
 * a typed-array union), so `colors` is `[splat_count * color_components]` f32
 * (white-filled when the dataset has no colors). `color_components` is 3 (RGB)
 * or 4 (RGBA — alpha is per-splat opacity and compacts with its splat).
 *
 * Outputs must be sized for the `splat_count` worst case; the caller slices
 * each to the returned visible count.
 *
 * # Arguments
 * * `positions` - Splat centers [splat_count * ndim]
 * * `cholesky` - Packed Cholesky factors [splat_count * packedSize]
 * * `amplitudes` - Splat amplitudes [splat_count]
 * * `colors` - Pre-normalized RGB(A) [splat_count * color_components]
 * * `color_components` - 3 (RGB) or 4 (RGBA)
 * * `discrete_visibility` - Precomputed discrete-dim gate [splat_count] (all 1 if none)
 * * `slice_position` - Current slice [ndim]
 * * `continuous_hidden_dims` - Sorted continuous hidden dims [num_continuous]
 * * `display_dims` - Display dims in requested order [2 or 3]
 * * `ndim`, `splat_count`, `min_amplitude`, `truncate`
 * * `out_centers3d` [splat_count * 3], `out_cholesky3d` [splat_count * 6],
 *   `out_amplitudes` [splat_count], `out_colors` [splat_count * color_components]
 * * `out_source_indices` - Per emitted splat, the SOURCE index `i` it came
 *   from [splat_count]. Compaction destroys that mapping, and picking needs it
 *   to translate a storage slot back into an on-disk element index (issue
 *   #1423). Pass an EMPTY slice to opt out — the recording is then skipped
 *   entirely and costs nothing.
 *
 * # Returns
 * Number of visible splats written (dense prefix length / stride).
 *
 * # Panics
 * Panics if `ndim > 16`. The worker routes >16D to the uncapped TS reference.
 */
export function project_gsplats_nd_to_3d(positions: Float32Array, cholesky: Float32Array, amplitudes: Float32Array, colors: Float32Array, discrete_visibility: Uint8Array, slice_position: Float32Array, continuous_hidden_dims: Uint32Array, display_dims: Uint32Array, ndim: number, splat_count: number, color_components: number, min_amplitude: number, truncate: number, out_centers3d: Float32Array, out_cholesky3d: Float32Array, out_amplitudes: Float32Array, out_colors: Float32Array, out_source_indices: Uint32Array): number;

/**
 * Sort splats back-to-front by camera-space depth.
 *
 * # Arguments
 * - `centers3`: Projected 3D splat centers `[count * 3]` (x, y, z triplets)
 * - `model_view`: Column-major 4x4 model-view matrix `[16]`
 *   (`camera.matrixWorldInverse × mesh.matrixWorld`, THREE.js layout)
 * - `ordering`: Output permutation `[count]` — `ordering[j]` is the original
 *   splat index drawn at instance slot `j` (slot 0 = farthest)
 * - `count`: Number of splats
 *
 * # Returns
 * Number of splats placed via depth keys, or `0` when the identity
 * fallback was taken (degenerate depth range — the ordering is still
 * fully written).
 */
export function sort_splats_by_depth(centers3: Float32Array, model_view: Float32Array, ordering: Uint32Array, count: number): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly calculate_effective_radii: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => number;
    readonly calculate_segment_lengths: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly clip_segment_single: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => void;
    readonly clip_segments_batch: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number) => number;
    readonly compact_visible_faces: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly compute_joint_codes: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number) => number;
    readonly decode_broadcasted: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly decode_geolog_perchannel_u16: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
    readonly decode_geolog_perchannel_u8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
    readonly decode_geolog_scalar_u16: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly decode_geolog_scalar_u8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly decode_linear_perchannel_u16: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
    readonly decode_linear_perchannel_u8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
    readonly decode_log_perchannel_u16: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
    readonly decode_log_perchannel_u8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
    readonly decode_log_scalar_u16: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly decode_log_scalar_u8: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly decode_lut_row_u16: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly decode_lut_row_u8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly decode_lut_scalar_u16: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly decode_lut_scalar_u8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly decode_quantized_u16: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly decode_quantized_u8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly decode_signed_log_perchannel_u16: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
    readonly decode_signed_log_perchannel_u8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
    readonly extract_3d_positions: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly interpolate_clipped_positions: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number) => number;
    readonly interpolate_colors_batch: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => number;
    readonly interpolate_scalars_batch: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => number;
    readonly mahalanobis_distance: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly mesh_vertex_visibility_mask: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => number;
    readonly project_gsplats_nd_to_3d: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number, b1: number, c1: number, d1: number, e1: number, f1: number, g1: number, h1: number, i1: number, j1: number) => number;
    readonly sort_splats_by_depth: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
