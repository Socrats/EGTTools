# NET: Games and dynamics on networks

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: network models that are correct against the literature and a reference
implementation, and fast enough for large sweeps. Decide whether the engine
stays in C++ or moves to numba.

Findings so far:
- Group games on networks compute a node's fitness from the single group
  centred on that node. The standard model (Santos, Santos & Pacheco 2008) has
  each node play in the k+1 groups centred on itself and on each neighbour.
- Thresholds are not scaled by group size on heterogeneous networks.
- The fitness cache builds `std::string` keys through `std::ostringstream` on
  every call.
- The exact average gradient of selection leaves out mutation, while the
  simulated dynamics include it.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| NET-01 | Reference tests: same graph/seed/parameters vs an independent reference implementation | todo | P0 | M | 0.3 | — | — |
| NET-02 | k+1-group payoff for group games on networks | todo | P0 | L | 0.3 | NET-01 | — |
| NET-03 | Optional degree-scaled threshold for heterogeneous networks | todo | P1 | S | 0.3 | NET-02 | — |
| NET-04 | Replace string-keyed fitness cache with an integer/hash key | todo | P1 | M | 0.3 | — | — |
| NET-05 | Decide and implement the mutation term in the exact network gradient of selection | todo | P1 | S | 0.3 | — | — |
| NET-06 | Benchmark C++ engine vs a numba engine (CSR adjacency, pluggable `njit` payoffs) | todo | P0 | M | 0.3 | NET-02, NET-04 | — |
| NET-07 | (if NET-06 favours numba) numba network engine; numba becomes a dependency | todo | P1 | L | 0.3 | NET-06 | — |
| NET-08 | JAX `vmap` ensemble backend for synchronous rules / many-replicate sweeps | todo | P3 | L | later | NET-06 | — |

## Notes

- **NET-01**: a node's fitness must match the reference to floating-point
  precision on a fixed configuration, and long-run strategy frequencies must
  agree within MC error.
- **NET-02**: the fitness can no longer be cached on (focal strategy,
  neighbour counts) alone, because the payoff depends on second neighbours.
  The cache key or strategy must change (see NET-04).
- **NET-06**: decide on throughput (updates/s) at N ∈ {10³, 10⁴, 10⁵}, regular
  and scale-free graphs. Also weigh extensibility: a numba engine lets users
  write new network games without compiling C++. Record the outcome in
  `DECISIONS.md`.
- **NET-08**: single-chain asynchronous updates map poorly to GPUs. A GPU only
  pays off when vectorising across replicates or with synchronous update rules.
