# build_tools/versioning/read_version.py

import re
from pathlib import Path


def get_version():
    version_file = Path(__file__).resolve().parents[2] / "cpp" / "src" / "version.h"
    content = version_file.read_text(encoding="utf-8")

    match = re.search(r'#define\s+EGTTOOLS_VERSION_STRING\s+"([\d.]+)"', content)
    if not match:
        raise RuntimeError("Could not find EGTTOOLS_VERSION_STRING in version.h")

    return match.group(1)
