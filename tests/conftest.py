# Copyright (c) 2019-2020  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

import glob
import os
import sys

# Prefer the local in-tree build over site-packages, but ONLY when the compiled
# extension (.so / .pyd) is present in src/ (i.e. after `build_ext --inplace`).
# When the wheel is installed into a fresh venv (CI / cibuildwheel test step),
# the extension lives only in site-packages, so this block is skipped and the
# installed package is used instead — no breakage.
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_has_inplace_build = any(
    glob.glob(os.path.join(_src, "egttools", "numerical", pat))
    for pat in ("numerical_*.so", "numerical_*.pyd", "numerical_*.dylib")
)
if _has_inplace_build and _src not in sys.path:
    sys.path.insert(0, _src)
