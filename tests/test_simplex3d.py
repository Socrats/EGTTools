"""Tests for egttools.plotting.simplex3d."""
import numpy as np
import pytest
import egttools as egt

from egttools.plotting.simplex3d import (
    Simplex3D,
    VERTICES,
    barycentric_to_cartesian,
    gradient_to_cartesian,
    _slice_grid,
    _slice_triangles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diagonal_gradient(payoff_diag):
    """Replicator dynamics for a diagonal payoff matrix."""
    P = np.diag(payoff_diag)
    def _grad(b):
        f = b @ P
        fbar = b @ f
        return b * (f - fbar)
    return _grad


GRADIENT = _diagonal_gradient([1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def test_vertices_map_to_themselves():
    """Each pure-strategy barycentric vector maps to the corresponding vertex."""
    for i in range(4):
        b = np.zeros(4)
        b[i] = 1.0
        assert np.allclose(barycentric_to_cartesian(b), VERTICES[i])


def test_centroid_maps_to_centroid():
    b = np.ones(4) / 4
    assert np.allclose(barycentric_to_cartesian(b), VERTICES.mean(axis=0))


def test_barycentric_to_cartesian_batch():
    """Batch input (N, 4) should give (N, 3) output."""
    b = np.random.dirichlet(np.ones(4), size=20)
    xyz = barycentric_to_cartesian(b)
    assert xyz.shape == (20, 3)


def test_gradient_to_cartesian_shape():
    b = np.random.dirichlet(np.ones(4), size=15)
    db = np.array([GRADIENT(bi) for bi in b])
    uvw = gradient_to_cartesian(db)
    assert uvw.shape == (15, 3)


def test_gradient_zero_at_pure_strategies():
    """Replicator dynamics are zero at pure strategies (vertices)."""
    for i in range(4):
        b = np.zeros(4); b[i] = 1.0
        db = GRADIENT(b)
        assert np.allclose(db, 0, atol=1e-12)
        uvw = gradient_to_cartesian(db)
        assert np.allclose(uvw, 0, atol=1e-12)


# ---------------------------------------------------------------------------
# Slice grid
# ---------------------------------------------------------------------------

def test_slice_grid_fixed_value():
    """All returned points should have the fixed strategy at the specified value."""
    for fixed in range(4):
        for val in [0.0, 0.1, 0.5]:
            pts = _slice_grid(fixed, val, n=6)
            assert np.allclose(pts[:, fixed], val, atol=1e-12)


def test_slice_grid_sums_to_one():
    pts = _slice_grid(2, 0.25, n=8)
    assert np.allclose(pts.sum(axis=1), 1.0, atol=1e-12)


def test_slice_grid_non_negative():
    pts = _slice_grid(1, 0.3, n=10)
    assert np.all(pts >= -1e-12)


def test_slice_grid_resolution():
    """Number of points grows with n as expected for a triangular grid."""
    n = 5
    pts = _slice_grid(0, 0.2, n=n)
    expected = (n + 1) * (n + 2) // 2
    assert len(pts) == expected


def test_slice_triangles_shape():
    xyz, faces = _slice_triangles(0, 0.2, n=5)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3


def test_slice_triangles_index_bounds():
    xyz, faces = _slice_triangles(3, 0.1, n=6)
    assert faces.min() >= 0
    assert faces.max() < len(xyz)


# ---------------------------------------------------------------------------
# Simplex3D construction
# ---------------------------------------------------------------------------

def test_draw_tetrahedron_returns_self():
    s = Simplex3D()
    result = s.draw_tetrahedron()
    assert result is s


def test_draw_tetrahedron_adds_traces():
    """draw_tetrahedron should add 6 edge traces + 4 face traces."""
    s = Simplex3D()
    s.draw_tetrahedron(face_opacity=0.1)
    assert len(s._traces) == 10  # 6 edges + 4 faces


def test_draw_tetrahedron_no_faces():
    s = Simplex3D()
    s.draw_tetrahedron(face_opacity=0.0)
    assert len(s._traces) == 6  # only edges


def test_draw_slice_returns_self():
    s = Simplex3D()
    result = s.draw_slice(fixed_strategy=0, value=0.2)
    assert result is s


def test_draw_slice_adds_mesh_trace():
    s = Simplex3D()
    n_before = len(s._traces)
    s.draw_slice(fixed_strategy=1, value=0.3, show_slice_mesh=True)
    assert len(s._traces) > n_before


def test_draw_slice_with_gradient_adds_arrowheads():
    """Slice streamplot uses flat Mesh3d arrowheads, not Cone glyphs."""
    s = Simplex3D()
    s.draw_slice(fixed_strategy=0, value=0.2, gradient_fn=GRADIENT,
                 n_grid=4, n_seeds=5)
    types = [type(t).__name__ for t in s._traces]
    # Flat arrowheads are Mesh3d; no Cone should be present
    assert 'Mesh3d' in types
    assert 'Cone' not in types


def test_draw_slice_without_gradient_no_cone():
    s = Simplex3D()
    s.draw_slice(fixed_strategy=0, value=0.2, gradient_fn=None)
    types = [type(t).__name__ for t in s._traces]
    assert 'Cone' not in types


def test_draw_slice_invalid_strategy():
    s = Simplex3D()
    with pytest.raises(ValueError, match="fixed_strategy"):
        s.draw_slice(fixed_strategy=5, value=0.2)


def test_draw_slice_invalid_value():
    s = Simplex3D()
    with pytest.raises(ValueError, match="value"):
        s.draw_slice(fixed_strategy=0, value=1.5)


def test_add_vertex_labels_returns_self():
    s = Simplex3D()
    result = s.add_vertex_labels(['A', 'B', 'C', 'D'])
    assert result is s


def test_add_vertex_labels_wrong_count():
    s = Simplex3D()
    with pytest.raises(ValueError, match="4 labels"):
        s.add_vertex_labels(['A', 'B', 'C'])


def test_add_vertex_labels_adds_four_text_traces():
    s = Simplex3D()
    n_before = len(s._traces)
    s.add_vertex_labels(['A', 'B', 'C', 'D'])
    assert len(s._traces) - n_before == 4


def test_draw_trajectory_returns_self():
    pts = np.random.dirichlet(np.ones(4), size=20)
    s = Simplex3D()
    result = s.draw_trajectory(pts)
    assert result is s


def test_draw_stationary_distribution_returns_self():
    """draw_stationary_distribution returns self for chaining."""
    Z = 10
    nb_states = egt.calculate_nb_states(Z, 4)
    sd = np.ones(nb_states) / nb_states
    s = Simplex3D()
    result = s.draw_stationary_distribution(sd, population_size=Z, colorbar=False)
    assert result is s


def test_draw_stationary_distribution_adds_traces():
    Z = 10
    nb_states = egt.calculate_nb_states(Z, 4)
    sd = np.ones(nb_states) / nb_states
    s = Simplex3D()
    n_before = len(s._traces)
    s.draw_stationary_distribution(sd, population_size=Z, colorbar=False)
    assert len(s._traces) > n_before


def test_draw_stationary_distribution_threshold():
    """With threshold=0.5, only top half of states are drawn."""
    Z = 10
    nb_states = egt.calculate_nb_states(Z, 4)
    sd = np.zeros(nb_states)
    sd[:5] = 1.0  # only first 5 states have mass
    sd /= sd.sum()

    s_all = Simplex3D()
    s_all.draw_stationary_distribution(sd, population_size=Z, threshold=0.0, colorbar=False)

    s_thresh = Simplex3D()
    s_thresh.draw_stationary_distribution(sd, population_size=Z, threshold=0.5, colorbar=False)

    # threshold should produce fewer or equal traces
    assert len(s_thresh._traces) <= len(s_all._traces)


def test_draw_stationary_distribution_top_k():
    Z = 10
    nb_states = egt.calculate_nb_states(Z, 4)
    sd = np.random.default_rng(0).exponential(size=nb_states)
    sd /= sd.sum()
    s = Simplex3D()
    # With top_k=5 there should be very few non-empty bucket traces
    s.draw_stationary_distribution(sd, population_size=Z, top_k=5,
                                   threshold=0.0, colorbar=False)
    # Total points across all bucket traces should be ≤ 5
    total_pts = sum(len(t.x) for t in s._traces if hasattr(t, 'x') and t.x is not None)
    assert total_pts <= 5


def test_draw_stationary_points_returns_self():
    pts = np.array([[0.25, 0.25, 0.25, 0.25]])
    s = Simplex3D()
    result = s.draw_stationary_points(pts)
    assert result is s


# ---------------------------------------------------------------------------
# Figure build
# ---------------------------------------------------------------------------

def test_build_returns_figure():
    import plotly.graph_objects as go
    s = Simplex3D()
    s.draw_tetrahedron()
    fig = s.build()
    assert isinstance(fig, go.Figure)


def test_build_scene_axes_hidden():
    s = Simplex3D()
    fig = s.build()
    scene = fig.layout.scene
    assert scene.xaxis.visible is False
    assert scene.yaxis.visible is False
    assert scene.zaxis.visible is False


def test_draw_streamlines_returns_self():
    s = Simplex3D()
    result = s.draw_streamlines(GRADIENT, fixed_strategy=0, fixed_value=0.2, n_seeds=3)
    assert result is s


def test_draw_streamlines_adds_traces():
    s = Simplex3D()
    n_before = len(s._traces)
    s.draw_streamlines(GRADIENT, fixed_strategy=0, fixed_value=0.2, n_seeds=3)
    assert len(s._traces) > n_before


def test_draw_streamlines_with_explicit_seeds():
    seeds = np.array([
        [0.2, 0.2, 0.3, 0.3],
        [0.3, 0.3, 0.2, 0.2],
    ])
    s = Simplex3D()
    result = s.draw_streamlines(GRADIENT, seeds=seeds)
    assert result is s


def test_draw_streamlines_raises_without_seeds_or_slice():
    s = Simplex3D()
    with pytest.raises(ValueError, match="seeds"):
        s.draw_streamlines(GRADIENT)


def test_method_chaining_full():
    """Full chain should build without error."""
    s = Simplex3D()
    fig = (s
           .draw_tetrahedron()
           .draw_slice(fixed_strategy=0, value=0.2, gradient_fn=GRADIENT,
                       n_grid=6, n_seeds=5)
           .draw_slice(fixed_strategy=0, value=0.4, gradient_fn=GRADIENT,
                       n_grid=6, n_seeds=5)
           .add_vertex_labels(['A', 'B', 'C', 'D'])
           .build())
    assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# Visual test (writes HTML for manual inspection)
# ---------------------------------------------------------------------------

def test_visual_simplex3d(tmp_path):
    """Visual test: synthetic gradient only.  Passes if no exception raised."""
    s = Simplex3D(figure_size=(700, 600))
    (s.draw_tetrahedron(face_opacity=0.06)
      .draw_slice(fixed_strategy=0, value=0.1, gradient_fn=GRADIENT,
                  n_grid=10, n_seeds=5, colorscale='Viridis')
      .draw_slice(fixed_strategy=0, value=0.3, gradient_fn=GRADIENT,
                  n_grid=10, n_seeds=5, colorscale='Viridis')
      .draw_slice(fixed_strategy=0, value=0.5, gradient_fn=GRADIENT,
                  n_grid=10, n_seeds=5, colorscale='Viridis')
      .add_vertex_labels(['A', 'B', 'C', 'D']))

    out = tmp_path / "simplex3d_visual.html"
    s.build().write_html(str(out), include_plotlyjs='cdn')
    assert out.exists()
    print(f"\nVisual output saved to: {out}")


def test_visual_simplex3d_normalform_game(tmp_path):
    """Visual test using NormalFormGame + PairwiseComparison gradient.

    Defines a 4-strategy cooperation game (AllC / AllD / TFT / Grim),
    wraps it with PairwiseComparison, and plots the gradient of selection
    on three cross-sectional slices inside the tetrahedron.
    Saves an interactive HTML figure to tmp_path for manual inspection.
    """
    import egttools as egt
    from egttools.analytical import PairwiseComparison

    # 4-strategy cooperation / defection game
    payoff_matrix = np.array([
        [ 3,  0,  3,  3],   # AllC
        [ 5,  1,  1,  1],   # AllD
        [ 3,  1,  3,  3],   # TFT
        [ 3,  1,  3,  3],   # Grim
    ], dtype=float)

    population_size = 50
    beta = 1.0
    # Matrix2PlayerGameHolder explicitly sets nb_strategies so PairwiseComparison
    # returns a gradient of the correct length (4).
    game = egt.games.Matrix2PlayerGameHolder(4, payoff_matrix)
    evolver = PairwiseComparison(population_size=population_size, game=game)

    def gradient(b: np.ndarray) -> np.ndarray:
        """Gradient of selection via PairwiseComparison (finite-pop model)."""
        state = np.floor(b * population_size).astype(np.int64)
        # clamp rounding errors so counts sum to population_size
        diff = population_size - state.sum()
        state[np.argmax(b)] += diff
        return evolver.calculate_gradient_of_selection(beta, state)

    s = Simplex3D(figure_size=(800, 650))
    (s.draw_tetrahedron(edge_color='#444', face_opacity=0.05)
      .draw_slice(fixed_strategy=1, value=0.05,
                  gradient_fn=gradient, n_grid=8, n_seeds=5,
                  slice_color='rgba(100,180,255,0.20)',
                  colorscale='Blues')
      .draw_slice(fixed_strategy=1, value=0.20,
                  gradient_fn=gradient, n_grid=8, n_seeds=5,
                  slice_color='rgba(100,180,255,0.20)',
                  colorscale='Blues')
      .draw_slice(fixed_strategy=1, value=0.40,
                  gradient_fn=gradient, n_grid=8, n_seeds=5,
                  slice_color='rgba(100,180,255,0.20)',
                  colorscale='Blues')
      .add_vertex_labels(['AllC', 'AllD', 'TFT', 'Grim']))

    out = tmp_path / "simplex3d_normalform.html"
    s.build().write_html(str(out), include_plotlyjs='cdn')
    assert out.exists()
    print(f"\nVisual (NormalFormGame) saved to: {out}")


def test_visual_simplex3d_stationary_distribution(tmp_path):
    """Visual test: stationary distribution as transparent spheres (Z=15).

    Uses a real PairwiseComparison + stationary distribution computation.
    Z=15 keeps computation under ~0.5 s.
    """
    from egttools.analytical import PairwiseComparison

    payoff_matrix = np.array([
        [ 3,  0,  3,  3],   # AllC
        [ 5,  1,  1,  1],   # AllD
        [ 3,  1,  3,  3],   # TFT
        [ 3,  1,  3,  3],   # Grim
    ], dtype=float)

    Z = 15; beta = 5.0; mu = beta / Z
    game = egt.games.Matrix2PlayerGameHolder(4, payoff_matrix)
    pc = PairwiseComparison(Z, game)
    # Note: T must be transposed for correct stationary distribution
    T = pc.calculate_transition_matrix(beta, mu)
    sd = egt.utils.calculate_stationary_distribution(T.T)

    def gradient(b):
        state = np.floor(b * Z).astype(np.int64)
        state[np.argmax(b)] += Z - state.sum()
        return pc.calculate_gradient_of_selection(beta, state)

    s = Simplex3D(figure_size=(900, 650))
    (s.draw_tetrahedron()
       # reference slices with in-plane streamplot
      .draw_slice(fixed_strategy=1, value=0.10, gradient_fn=gradient,
                  n_grid=8, n_seeds=6, slice_color='lightblue',
                  colorscale='Blues', cone_scale=0.025)
      .draw_slice(fixed_strategy=1, value=0.40, gradient_fn=gradient,
                  n_grid=8, n_seeds=6, slice_color='lightcyan',
                  colorscale='Blues', cone_scale=0.025)
      # stationary distribution: grayscale spheres, opacity ∝ probability
      .draw_stationary_distribution(
          sd, population_size=Z,
          colorscale='Greys', opacity_scale=4.0,
          min_opacity=0.0, max_opacity=0.95,
          marker_size=6, threshold=0.3, colorbar=True,
          colorbar_label='stationary distribution')
      .add_vertex_labels(['AllC', 'AllD', 'TFT', 'Grim']))

    out = tmp_path / "simplex3d_stationary.html"
    s.build(colorbar=True, colorbar_label='gradient of selection').write_html(
        str(out), include_plotlyjs='cdn')
    assert out.exists()
    print(f"\nVisual (stationary distribution) saved to: {out}")
