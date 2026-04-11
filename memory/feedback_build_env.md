---
name: Always build inside egtenv
description: EGTtools C++ extension must be built with the egtenv conda environment (Python 3.10), not the base env (Python 3.12)
type: feedback
---

Always build using `conda run -n egtenv python setup.py build_ext --inplace`.

**Why:** The egtenv environment uses Python 3.10. Building with the base environment's Python 3.12 produces a `.cpython-312` `.so` that egtenv cannot import, causing `ImportError` at test time.

**How to apply:** Whenever a build command is needed, prefix it with `conda run -n egtenv`. Likewise, run tests with `conda run -n egtenv python -m pytest ...`.
