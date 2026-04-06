"""
3D simplex (tetrahedron) visualisation using Plotly.

The 3D simplex represents a 4-strategy evolutionary game: every point
(b0, b1, b2, b3) with bi ≥ 0, Σbi = 1 is a population state.

The key idea for interior visualisation is *slicing*: fix one strategy at a
constant value c (e.g. b0 = c) and show the resulting triangular cross-section
with the replicator dynamics projected onto it.  Multiple slices at different
levels of the same (or different) strategies build up an intuition of the
interior flow.

Arrows are rendered as Plotly ``go.Cone`` glyphs — true 3D cones with correct
WebGL depth sorting, so occlusion is handled automatically.

Coordinate convention
---------------------
Strategy indices 0-3 map to the four vertices of a regular tetrahedron
embedded in ℝ³::

    V = np.array([
        [0,          0,                   0      ],   # strategy 0  (bottom-left)
        [1,          0,                   0      ],   # strategy 1  (bottom-right)
        [0.5,        np.sqrt(3)/2,        0      ],   # strategy 2  (bottom-back)
        [0.5,        np.sqrt(3)/6,        np.sqrt(6)/3],  # strategy 3 (top)
    ])

A barycentric point b is mapped to 3D via  xyz = b @ V.
A replicator velocity db/dt maps to 3D via  dxyz/dt = (db/dt) @ V.
"""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.colors as pc
except ImportError as _plotly_err:
    raise ImportError(
        "egttools.plotting.Simplex3D requires plotly.  "
        "Install it with:  pip install plotly"
    ) from _plotly_err


def _sample_colorscale(colorscale: str, values: np.ndarray) -> List[str]:
    """Map an array of values in [0, 1] to RGB colour strings.

    Uses ``plotly.colors.sample_colorscale`` which returns hex/rgb strings
    directly, one per value.
    """
    return pc.sample_colorscale(colorscale, list(values.clip(0, 1)))

# ---------------------------------------------------------------------------
# Regular tetrahedron vertices in ℝ³
# ---------------------------------------------------------------------------
VERTICES: np.ndarray = np.array([
    [0.0,       0.0,              0.0            ],  # strategy 0
    [1.0,       0.0,              0.0            ],  # strategy 1
    [0.5,       np.sqrt(3) / 2,  0.0            ],  # strategy 2
    [0.5,       np.sqrt(3) / 6,  np.sqrt(6) / 3],  # strategy 3
], dtype=float)

# Edges of the tetrahedron (pairs of vertex indices)
_EDGES: List[Tuple[int, int]] = list(itertools.combinations(range(4), 2))

# Faces of the tetrahedron (triples of vertex indices, outward-facing normal)
_FACES: List[Tuple[int, int, int]] = list(itertools.combinations(range(4), 3))


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _integrate_streamline(
        gradient_fn: Callable,
        b0: np.ndarray,
        fixed_idx: int,
        fixed_value: float,
        dt: float = 0.005,
        max_steps: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the gradient field forward from ``b0`` until it hits a boundary.

    The step direction is *normalised* at each point so the streamline always
    advances at a fixed arc-length per step (like matplotlib's streamplot).
    The actual gradient magnitude at each point is returned separately for
    colouring.

    The fixed strategy is clamped throughout so the trajectory stays on the
    slice ``b[fixed_idx] = fixed_value``.

    Returns
    -------
    path : np.ndarray, shape (T, 4)
    magnitudes : np.ndarray, shape (T,)
        Raw gradient magnitude at each point (for colorscale mapping).
    """
    free = [i for i in range(4) if i != fixed_idx]
    b = b0.copy()
    path = [b.copy()]
    mags = []

    for _ in range(max_steps):
        db_raw = gradient_fn(b)
        db_raw[fixed_idx] = 0.0
        db_raw[free] -= db_raw[free].sum() / len(free)
        mag = np.linalg.norm(db_raw)
        mags.append(mag)

        if mag < 1e-10:
            break  # equilibrium reached

        # Normalised step: always advance dt in arc-length
        db_unit = db_raw / mag

        b_new = b + dt * db_unit
        b_new[fixed_idx] = fixed_value
        b_new = np.clip(b_new, 0, 1)
        b_new /= b_new.sum()

        if np.any(b_new < 1e-3):          # hit boundary
            path.append(b_new)
            mags.append(mag)
            break

        b = b_new
        path.append(b.copy())

    return np.array(path), np.array(mags)


def _integrate_streamline_3d(
        gradient_fn: Callable,
        b0: np.ndarray,
        dt: float = 0.005,
        max_steps: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the gradient field freely in the full 3-simplex.

    No slice constraint — the trajectory moves in all four barycentric
    dimensions and stops only when it reaches a boundary (any coordinate
    < threshold) or an equilibrium (gradient ≈ 0).

    Like ``_integrate_streamline``, steps are normalised to a fixed
    arc-length so line length is uniform regardless of gradient magnitude.
    Magnitude is returned separately for colour mapping.

    Returns
    -------
    path : np.ndarray, shape (T, 4)
    magnitudes : np.ndarray, shape (T,)
    """
    b = b0.copy()
    path = [b.copy()]
    mags = []

    for _ in range(max_steps):
        db_raw = gradient_fn(b)
        # Project onto the simplex tangent plane: subtract mean so sum = 0
        db_raw = db_raw - db_raw.mean()
        mag = np.linalg.norm(db_raw)
        mags.append(mag)

        if mag < 1e-10:
            break  # equilibrium reached

        db_unit = db_raw / mag
        b_new = b + dt * db_unit
        b_new = np.clip(b_new, 0, 1)
        b_new /= b_new.sum()

        if np.any(b_new < 1e-3):  # hit a face/edge of the tetrahedron
            path.append(b_new)
            mags.append(mag)
            break

        b = b_new
        path.append(b.copy())

    return np.array(path), np.array(mags)


def _slice_normal(fixed_idx: int) -> np.ndarray:
    """Return the unit normal of the slice plane ``b[fixed_idx] = const``.

    The normal is the cross product of two edge vectors of the slice triangle.
    """
    free = [i for i in range(4) if i != fixed_idx]
    # Three corners of the sub-simplex (value=0 case is fine for the normal)
    c = [VERTICES[j] for j in free]
    n = np.cross(c[1] - c[0], c[2] - c[0])
    return n / np.linalg.norm(n)


def _flat_arrowhead(
        pos: np.ndarray,
        direction_3d: np.ndarray,
        slice_normal: np.ndarray,
        head_length: float,
        head_width: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a flat triangular arrowhead lying in the slice plane.

    The arrowhead is a filled triangle with:
      - tip  : ``pos + head_length * direction_3d``
      - left : ``pos + (head_width/2) * perp``
      - right: ``pos - (head_width/2) * perp``

    where ``perp = slice_normal × direction_3d`` (in-plane perpendicular).

    Parameters
    ----------
    pos : (3,) array
        Base (tail) of the arrowhead in 3D.
    direction_3d : (3,) array
        Unit direction vector in the plane.
    slice_normal : (3,) array
        Unit normal of the slice plane.
    head_length, head_width : float

    Returns
    -------
    verts : (3, 3) array — [tip, left, right] in 3D
    face  : (3,) int array — [0, 1, 2] connectivity
    """
    perp = np.cross(slice_normal, direction_3d)
    perp_len = np.linalg.norm(perp)
    if perp_len < 1e-12:
        perp = np.array([0.0, 0.0, 1.0])
    else:
        perp /= perp_len

    tip   = pos + head_length * direction_3d
    left  = pos + (head_width / 2) * perp
    right = pos - (head_width / 2) * perp
    return np.array([tip, left, right]), np.array([0, 1, 2])


def barycentric_to_cartesian(b: np.ndarray) -> np.ndarray:
    """Convert barycentric coordinates to 3D Cartesian.

    Parameters
    ----------
    b : array_like, shape (..., 4)
        Barycentric coordinates summing to 1.

    Returns
    -------
    np.ndarray, shape (..., 3)
    """
    b = np.asarray(b, dtype=float)
    return b @ VERTICES


def gradient_to_cartesian(db_dt: np.ndarray) -> np.ndarray:
    """Project a replicator-dynamics velocity vector from Δ³ to ℝ³.

    Parameters
    ----------
    db_dt : array_like, shape (..., 4)
        Velocity in barycentric coordinates (db/dt from replicator equation).

    Returns
    -------
    np.ndarray, shape (..., 3)
    """
    db_dt = np.asarray(db_dt, dtype=float)
    return db_dt @ VERTICES


def _slice_grid(fixed_idx: int, fixed_value: float,
                n: int) -> np.ndarray:
    """
    Generate a triangular grid of barycentric points on the slice
    ``b[fixed_idx] = fixed_value``.

    The remaining three strategies are sampled on a uniform triangular grid
    with resolution ``n`` (number of divisions per edge of the sub-simplex).

    Parameters
    ----------
    fixed_idx : int
        Which strategy index is fixed (0–3).
    fixed_value : float
        Value of the fixed strategy (must be in [0, 1]).
    n : int
        Number of divisions along each edge of the sub-simplex.

    Returns
    -------
    np.ndarray, shape (N, 4)
        Barycentric coordinates of the grid points.
    """
    free = [i for i in range(4) if i != fixed_idx]
    scale = 1.0 - fixed_value          # remaining probability mass
    points = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            # barycentric coords within the sub-simplex
            a, b_, c_ = i / n, j / n, k / n
            bary = np.zeros(4)
            bary[fixed_idx] = fixed_value
            bary[free[0]] = a * scale
            bary[free[1]] = b_ * scale
            bary[free[2]] = c_ * scale
            points.append(bary)
    return np.array(points)


def _slice_triangles(fixed_idx: int, fixed_value: float,
                     n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the vertex positions and triangle connectivity for a filled slice.

    Returns
    -------
    xyz : np.ndarray, shape (N, 3)
    faces : np.ndarray, shape (M, 3)   — integer indices into xyz
    """
    free = [i for i in range(4) if i != fixed_idx]
    scale = 1.0 - fixed_value

    # Build an index map: (i, j) -> vertex index
    idx_map: dict = {}
    xyz_list = []
    v = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            bary = np.zeros(4)
            bary[fixed_idx] = fixed_value
            bary[free[0]] = (i / n) * scale
            bary[free[1]] = (j / n) * scale
            bary[free[2]] = (k / n) * scale
            xyz_list.append(barycentric_to_cartesian(bary))
            idx_map[(i, j)] = v
            v += 1

    faces_list = []
    for i in range(n):
        for j in range(n - i):
            # lower triangle
            faces_list.append([idx_map[(i, j)],
                                idx_map[(i + 1, j)],
                                idx_map[(i, j + 1)]])
            if i + j + 2 <= n:
                # upper triangle
                faces_list.append([idx_map[(i + 1, j)],
                                   idx_map[(i + 1, j + 1)],
                                   idx_map[(i, j + 1)]])

    return np.array(xyz_list), np.array(faces_list, dtype=int)


# ---------------------------------------------------------------------------
# Simplex3D class
# ---------------------------------------------------------------------------

class Simplex3D:
    """
    3D simplex (tetrahedron) visualisation for 4-strategy evolutionary games.

    Interior dynamics are shown by adding cross-sectional slices at a fixed
    strategy value.  Each slice is a triangle embedded in 3D; the replicator
    dynamics on that slice are drawn as cone-arrow glyphs.

    The figure is a Plotly ``go.Figure`` with a single ``go.Scene`` (3D axes).
    Call :meth:`show` to display it in a notebook or browser, or use
    ``.figure`` to access the underlying Plotly object for further customisation.

    Parameters
    ----------
    figure_size : Tuple[int, int]
        Width and height of the Plotly figure in pixels.

    Examples
    --------
    >>> import numpy as np
    >>> from egttools.plotting import Simplex3D
    >>> def gradient(b):
    ...     # replicator dynamics for a 4-strategy game
    ...     payoffs = np.array([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]])
    ...     f = b @ payoffs          # mean fitness per strategy
    ...     fbar = b @ f             # mean population fitness
    ...     return b * (f - fbar)
    >>> s = Simplex3D()
    >>> (s.draw_tetrahedron()
    ...   .draw_slice(fixed_strategy=0, value=0.25,
    ...               gradient_fn=gradient, n_grid=8, n_arrows=6)
    ...   .add_vertex_labels(['A', 'B', 'C', 'D'])
    ...   .show())
    """

    def __init__(self, figure_size: Tuple[int, int] = (700, 600)) -> None:
        self._traces: List[go.BaseTraceType] = []
        self._figure_size = figure_size
        # Filled in by draw_slice / draw_streamlines for the shared colorbar
        self._colorbar_colorscale: Optional[str] = None
        self._colorbar_vmax: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_tetrahedron(
            self,
            edge_color: str = '#111111',
            edge_width: float = 5.0,
            face_color: str = 'lightblue',
            face_opacity: float = 0.05,
    ) -> 'Simplex3D':
        """Draw the wireframe and optionally semi-transparent faces.

        Parameters
        ----------
        edge_color : str
            Colour of the 6 edges.
        edge_width : float
            Width of the edges in pixels.
        face_color : str
            Fill colour of the 4 triangular faces.
        face_opacity : float
            Opacity of the faces (0 = invisible, 1 = opaque).  Keep low so
            interior slices remain visible.
        """
        # --- edges ---
        for i, j in _EDGES:
            v0, v1 = VERTICES[i], VERTICES[j]
            self._traces.append(go.Scatter3d(
                x=[v0[0], v1[0], None],
                y=[v0[1], v1[1], None],
                z=[v0[2], v1[2], None],
                mode='lines',
                line=dict(color=edge_color, width=edge_width),
                showlegend=False,
                hoverinfo='skip',
            ))

        # --- faces ---
        if face_opacity > 0:
            for tri in _FACES:
                a, b_, c_ = VERTICES[tri[0]], VERTICES[tri[1]], VERTICES[tri[2]]
                self._traces.append(go.Mesh3d(
                    x=[a[0], b_[0], c_[0]],
                    y=[a[1], b_[1], c_[1]],
                    z=[a[2], b_[2], c_[2]],
                    i=[0], j=[1], k=[2],
                    color=face_color,
                    opacity=face_opacity,
                    showlegend=False,
                    hoverinfo='skip',
                ))

        return self

    def draw_slice(
            self,
            fixed_strategy: int,
            value: float,
            gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            n_grid: int = 10,
            n_seeds: int = 5,
            slice_color: str = 'orange',
            arrow_color: str = '#c0392b',
            min_line_width: float = 1.5,
            max_line_width: float = 5.0,
            cone_scale: float = 0.008,
            arrow_fraction: float = 0.5,
            colorscale: str = 'Viridis',
            show_slice_mesh: bool = True,
            min_distance: float = 0.04,
            dt: float = 0.005,
            max_steps: int = 400,
    ) -> 'Simplex3D':
        """Add a cross-sectional slice at ``b[fixed_strategy] = value``.

        The slice is a filled triangle embedded in 3D.  If ``gradient_fn`` is
        given, the in-plane dynamics are visualised as a **2D streamplot**
        embedded on the slice — constrained integration, one arrowhead per
        line at the mid-point, linewidth and colour both encoding magnitude,
        and a density filter that prevents overlapping lines.  This matches
        the appearance of :class:`Simplex2D`'s streamplot.

        Parameters
        ----------
        fixed_strategy : int
            Index of the strategy to fix (0–3).
        value : float
            Value of the fixed strategy in [0, 1).
        gradient_fn : callable, optional
            ``f(b) -> db/dt``.  Receives a 1-D array of shape (4,) and returns
            an array of the same shape.
        n_grid : int
            Resolution of the triangular mesh used to fill the slice.
        n_seeds : int
            Seed grid resolution along each edge of the sub-simplex.
        slice_color : str
            CSS colour of the slice fill panel.
        arrow_color : str
            Flat colour for lines and cones when ``colorscale`` is ``None``.
        min_line_width : float
            Narrowest shaft width (pixels), at minimum magnitude.
        max_line_width : float
            Widest shaft width (pixels), at maximum magnitude.
        cone_scale : float
            Cone head length in data units.
        arrow_fraction : float
            Position of the arrowhead along each streamline (0 = start,
            1 = end, 0.5 = mid-point, matching matplotlib streamplot).
        colorscale : str
            Plotly colorscale for shaft and cone colouring by magnitude.
        show_slice_mesh : bool
            Whether to draw the filled triangular panel.
        min_distance : float
            Minimum Cartesian distance between streamlines (density mask).
        dt : float
            Arc-length step size for integration.
        max_steps : int
            Maximum integration steps per streamline.
        """
        if not (0 <= fixed_strategy <= 3):
            raise ValueError("fixed_strategy must be 0, 1, 2, or 3")
        if not (0.0 <= value < 1.0):
            raise ValueError("value must be in [0, 1)")

        # --- filled slice mesh ---
        if show_slice_mesh:
            xyz_mesh, faces = _slice_triangles(fixed_strategy, value, n_grid)
            self._traces.append(go.Mesh3d(
                x=xyz_mesh[:, 0],
                y=xyz_mesh[:, 1],
                z=xyz_mesh[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=slice_color,
                opacity=0.25,
                showlegend=False,
                hoverinfo='skip',
                flatshading=True,
            ))

        # --- slice streamplot (constrained to the slice plane) ---
        if gradient_fn is not None:
            seeds = _slice_grid(fixed_strategy, value, n_seeds)
            seeds = seeds[np.all(seeds > 2e-2, axis=1)]

            if len(seeds) == 0:
                return self

            # Integrate with density filter — same logic as draw_streamlines
            all_paths: List[np.ndarray] = []
            all_mags: List[np.ndarray] = []
            drawn_xyz: List[np.ndarray] = []

            for seed in seeds:
                seed_xyz = barycentric_to_cartesian(seed)
                if drawn_xyz:
                    all_drawn = np.vstack(drawn_xyz)
                    if np.min(np.linalg.norm(all_drawn - seed_xyz, axis=1)) < min_distance:
                        continue
                path, mags = _integrate_streamline(
                    gradient_fn, seed, fixed_strategy, value,
                    dt=dt, max_steps=max_steps,
                )
                if len(path) < 2:
                    continue
                all_paths.append(path)
                all_mags.append(mags)
                drawn_xyz.append(barycentric_to_cartesian(path))

            if not all_paths:
                return self

            global_max = max(m.max() for m in all_mags)
            if global_max < 1e-12:
                global_max = 1.0

            # Record for shared colorbar
            if colorscale is not None:
                self._colorbar_colorscale = colorscale
                self._colorbar_vmax = max(self._colorbar_vmax, global_max)

            # ---- shafts: linewidth ∝ mean magnitude, colour ∝ local magnitude
            for path, mags in zip(all_paths, all_mags):
                xyz_path = barycentric_to_cartesian(path)
                norm_mags = mags / global_max
                lw = min_line_width + float(norm_mags.mean()) * (max_line_width - min_line_width)
                colors = (_sample_colorscale(colorscale, norm_mags)
                          if colorscale is not None
                          else [arrow_color] * len(path))
                self._traces.append(go.Scatter3d(
                    x=list(xyz_path[:, 0]),
                    y=list(xyz_path[:, 1]),
                    z=list(xyz_path[:, 2]),
                    mode='lines',
                    line=dict(color=colors, width=lw,
                              colorscale=colorscale if colorscale else None),
                    showlegend=False,
                    hoverinfo='skip',
                ))

            # ---- one flat 2D arrowhead per streamline at arrow_fraction ----
            snormal = _slice_normal(fixed_strategy)
            free = [i for i in range(4) if i != fixed_strategy]
            head_length = cone_scale
            head_width  = cone_scale * 0.6

            for path, mags in zip(all_paths, all_mags):
                n = len(path)
                idx = max(0, min(n - 1, int(round(arrow_fraction * (n - 1)))))
                b_pt = path[idx]
                db = gradient_fn(b_pt).copy()
                db[fixed_strategy] = 0.0
                db[free] -= db[free].sum() / len(free)
                uvw = gradient_to_cartesian(db)
                mag = np.linalg.norm(uvw)
                if mag < 1e-12:
                    continue
                direction = uvw / mag
                xyz_pt = barycentric_to_cartesian(b_pt)

                verts, face = _flat_arrowhead(
                    xyz_pt, direction, snormal, head_length, head_width
                )
                # Colour by local magnitude
                norm_mag = float(mags[idx] / global_max)
                color = (_sample_colorscale(colorscale, np.array([norm_mag]))[0]
                         if colorscale is not None else arrow_color)

                self._traces.append(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=[face[0]], j=[face[1]], k=[face[2]],
                    color=color,
                    opacity=1.0,
                    showlegend=False,
                    hoverinfo='skip',
                ))

        return self

    def draw_streamlines(
            self,
            gradient_fn: Callable[[np.ndarray], np.ndarray],
            seeds: Optional[np.ndarray] = None,
            fixed_strategy: Optional[int] = None,
            fixed_value: Optional[float] = None,
            n_seeds: int = 5,
            colorscale: str = 'Viridis',
            arrow_color: str = '#c0392b',
            min_line_width: float = 1.5,
            max_line_width: float = 5.0,
            cone_scale: float = 0.008,
            arrow_fraction: float = 0.5,
            dt: float = 0.005,
            max_steps: int = 600,
            min_distance: float = 0.04,
    ) -> 'Simplex3D':
        """Integrate and draw free 3D streamlines, inspired by matplotlib streamplot.

        Mimics matplotlib's streamplot behaviour in 3D:

        * **Uniform coverage** — new streamlines are rejected if their seed is
          within ``min_distance`` (in Cartesian data units) of any already-drawn
          point, so lines are spread evenly rather than clumped.
        * **One arrowhead per streamline** — placed at ``arrow_fraction`` of the
          total arc-length (default 50 %, i.e. mid-point), exactly as
          streamplot places its arrow near the middle of each line.
        * **Linewidth encodes speed** — each streamline's width scales
          linearly with its mean gradient magnitude between ``min_line_width``
          and ``max_line_width``, analogous to streamplot's ``linewidth``
          parameter when set to a speed array.
        * **Color encodes speed** — ``colorscale`` maps local magnitude to
          colour, interpolated per-vertex along the shaft.

        Parameters
        ----------
        gradient_fn : callable
            ``f(b) -> db/dt``, shape (4,) → (4,).
        seeds : np.ndarray, shape (K, 4), optional
            Explicit seed points.  When provided the density filter is still
            applied, so some seeds may be skipped.
        fixed_strategy : int, optional
            Strategy index for automatic seed generation on a slice plane.
        fixed_value : float, optional
            Slice value for automatic seed generation.
        n_seeds : int
            Grid resolution for automatic seed generation.
        colorscale : str
            Plotly colorscale for shaft/cone colouring by magnitude.
        arrow_color : str
            Flat colour when ``colorscale`` is ``None``.
        min_line_width : float
            Narrowest shaft width (pixels), used at minimum magnitude.
        max_line_width : float
            Widest shaft width (pixels), used at maximum magnitude.
        cone_scale : float
            Cone head length in data units.
        arrow_fraction : float
            Position of the arrowhead along each streamline as a fraction of
            total arc-length (0 = start, 1 = end, 0.5 = mid-point).
        dt : float
            Arc-length step size for integration.
        max_steps : int
            Maximum integration steps per streamline.
        min_distance : float
            Minimum Cartesian distance between any two streamline points from
            *different* streamlines.  Acts as the density mask from streamplot.
        """
        if seeds is None:
            if fixed_strategy is None or fixed_value is None:
                raise ValueError(
                    "Provide either 'seeds' or both 'fixed_strategy' and "
                    "'fixed_value' for automatic seed generation."
                )
            seeds = _slice_grid(fixed_strategy, fixed_value, n_seeds)
            seeds = seeds[np.all(seeds > 2e-2, axis=1)]

        if len(seeds) == 0:
            return self

        # --- integrate, applying density filter ---
        all_paths: List[np.ndarray] = []
        all_mags: List[np.ndarray] = []
        # Accumulate all drawn Cartesian points for distance checking
        drawn_xyz: List[np.ndarray] = []

        for seed in seeds:
            seed_xyz = barycentric_to_cartesian(seed)
            # Reject seed if too close to an already-drawn point
            if drawn_xyz:
                all_drawn = np.vstack(drawn_xyz)
                if np.min(np.linalg.norm(all_drawn - seed_xyz, axis=1)) < min_distance:
                    continue

            path, mags = _integrate_streamline_3d(
                gradient_fn, seed, dt=dt, max_steps=max_steps,
            )
            if len(path) < 2:
                continue

            all_paths.append(path)
            all_mags.append(mags)
            drawn_xyz.append(barycentric_to_cartesian(path))

        if not all_paths:
            return self

        global_max = max(m.max() for m in all_mags)
        if global_max < 1e-12:
            global_max = 1.0

        # Record for shared colorbar
        if colorscale is not None:
            self._colorbar_colorscale = colorscale
            self._colorbar_vmax = max(self._colorbar_vmax, global_max)

        # --- shaft lines (one trace per streamline, width ∝ mean magnitude) ---
        for path, mags in zip(all_paths, all_mags):
            xyz_path = barycentric_to_cartesian(path)
            norm_mags = mags / global_max

            # Linewidth proportional to mean speed (like streamplot linewidth=speed)
            mean_norm = float(norm_mags.mean())
            lw = min_line_width + mean_norm * (max_line_width - min_line_width)

            colors = (_sample_colorscale(colorscale, norm_mags)
                      if colorscale is not None
                      else [arrow_color] * len(path))

            self._traces.append(go.Scatter3d(
                x=list(xyz_path[:, 0]),
                y=list(xyz_path[:, 1]),
                z=list(xyz_path[:, 2]),
                mode='lines',
                line=dict(color=colors, width=lw,
                          colorscale=colorscale if colorscale else None),
                showlegend=False,
                hoverinfo='skip',
            ))

        # --- one cone per streamline, at arrow_fraction of arc-length -------
        cx, cy, cz, cu, cv, cw = [], [], [], [], [], []

        for path in all_paths:
            n = len(path)
            arrow_idx = max(0, min(n - 1, int(round(arrow_fraction * (n - 1)))))
            b_pt = path[arrow_idx]
            db = gradient_fn(b_pt)
            uvw = gradient_to_cartesian(db - db.mean())
            mag = np.linalg.norm(uvw)
            if mag < 1e-12:
                continue
            direction = uvw / mag * cone_scale
            xyz_pt = barycentric_to_cartesian(b_pt)
            cx.append(xyz_pt[0]); cy.append(xyz_pt[1]); cz.append(xyz_pt[2])
            cu.append(direction[0]); cv.append(direction[1]); cw.append(direction[2])

        if cx:
            cone_kw = dict(
                x=cx, y=cy, z=cz,
                u=cu, v=cv, w=cw,
                sizemode='scaled', sizeref=0.3,
                anchor='tail',
                showlegend=False, hoverinfo='skip',
            )
            if colorscale is not None:
                cone_kw.update(colorscale=colorscale, showscale=False)
            else:
                cone_kw.update(
                    colorscale=[[0, arrow_color], [1, arrow_color]],
                    showscale=False,
                )
            self._traces.append(go.Cone(**cone_kw))

        return self

    def draw_trajectory(
            self,
            points: np.ndarray,
            color: str = 'blue',
            width: float = 4.0,
            name: str = '',
    ) -> 'Simplex3D':
        """Draw a trajectory curve inside the tetrahedron.

        Parameters
        ----------
        points : np.ndarray, shape (T, 4)
            Sequence of barycentric coordinates along the trajectory.
        color : str
            Line colour.
        width : float
            Line width in pixels.
        name : str
            Label shown in the Plotly legend.
        """
        xyz = barycentric_to_cartesian(np.asarray(points))
        self._traces.append(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='lines',
            line=dict(color=color, width=width),
            name=name,
            showlegend=bool(name),
            hoverinfo='skip',
        ))
        return self

    def draw_stationary_points(
            self,
            points: np.ndarray,
            stability: Optional[Sequence[int]] = None,
            stable_color: str = 'black',
            unstable_color: str = 'white',
            saddle_color: str = 'grey',
            size: float = 8.0,
    ) -> 'Simplex3D':
        """Draw stationary points inside or on the boundary of the tetrahedron.

        Parameters
        ----------
        points : np.ndarray, shape (K, 4)
            Barycentric coordinates of each stationary point.
        stability : sequence of int, optional
            Stability label per point: 1 = stable, -1 = unstable, 0 = saddle.
            If None all points are drawn with ``stable_color``.
        stable_color, unstable_color, saddle_color : str
            Marker colours for each stability class.
        size : float
            Marker size in pixels.
        """
        points = np.asarray(points)
        xyz = barycentric_to_cartesian(points)

        if stability is None:
            stability = [1] * len(points)

        color_map = {1: stable_color, -1: unstable_color, 0: saddle_color}
        colors = [color_map.get(s, stable_color) for s in stability]

        self._traces.append(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=colors,
                line=dict(color='black', width=1),
            ),
            showlegend=False,
            hoverinfo='skip',
        ))
        return self

    def add_vertex_labels(
            self,
            labels: Sequence[str],
            fontsize: int = 16,
            color: str = 'black',
            offset: float = 0.06,
    ) -> 'Simplex3D':
        """Add text labels near the four vertices.

        Parameters
        ----------
        labels : sequence of str
            Four labels in strategy order (0–3).
        fontsize : int
            Font size in points.
        color : str
            Text colour.
        offset : float
            How far to push each label away from the centroid, in data units.
        """
        if len(labels) != 4:
            raise ValueError("Exactly 4 labels required (one per strategy).")

        centroid = VERTICES.mean(axis=0)
        for i, (vertex, label) in enumerate(zip(VERTICES, labels)):
            direction = vertex - centroid
            pos = vertex + offset * direction / np.linalg.norm(direction)
            self._traces.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='text',
                text=[label],
                textfont=dict(size=fontsize, color=color),
                showlegend=False,
                hoverinfo='skip',
            ))
        return self

    # ------------------------------------------------------------------
    # Figure assembly
    # ------------------------------------------------------------------

    def build(
            self,
            colorbar: bool = True,
            colorbar_label: str = 'gradient magnitude',
            colorbar_thickness: int = 15,
            colorbar_len: float = 0.5,
    ) -> go.Figure:
        """Assemble and return the Plotly Figure.

        Parameters
        ----------
        colorbar : bool
            Whether to add a colorbar for the gradient magnitude scale.
            Only shown when a colorscale was used in ``draw_slice`` or
            ``draw_streamlines``.  Default ``True``.
        colorbar_label : str
            Title shown next to the colorbar.
        colorbar_thickness : int
            Colorbar width in pixels.
        colorbar_len : float
            Colorbar length as a fraction of the plot height.
        """
        traces = list(self._traces)

        # Invisible Scatter3d that carries only the colorbar
        if colorbar and self._colorbar_colorscale is not None:
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    color=[0, self._colorbar_vmax],
                    colorscale=self._colorbar_colorscale,
                    cmin=0,
                    cmax=self._colorbar_vmax,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=colorbar_label, side='right'),
                        thickness=colorbar_thickness,
                        len=colorbar_len,
                        x=1.02,
                    ),
                    size=0.001,  # effectively invisible
                ),
                showlegend=False,
                hoverinfo='skip',
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            width=self._figure_size[0],
            height=self._figure_size[1],
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                zaxis=dict(visible=False, showgrid=False, zeroline=False),
                bgcolor='white',
                aspectmode='data',
            ),
            paper_bgcolor='white',
            margin=dict(l=0, r=60, t=0, b=0),
        )
        return fig

    @property
    def figure(self) -> go.Figure:
        """The assembled Plotly Figure with colorbar (builds on first access)."""
        return self.build()

    def show(self, colorbar: bool = True, **kwargs) -> None:
        """Display the figure in a browser or Jupyter notebook.

        Parameters
        ----------
        colorbar : bool
            Whether to include the colorbar.  Default ``True``.
        **kwargs
            Forwarded to ``build()``.
        """
        self.build(colorbar=colorbar, **kwargs).show()
