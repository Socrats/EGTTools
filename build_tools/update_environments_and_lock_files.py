"""Script to update CI environment files and associated lock files.

To run it you need to be in the root folder of the scikit-learn repo:
python build_tools/update_environments_and_lock_files.py

Two scenarios where this script can be useful:
- make sure that the latest versions of all the dependencies are used in the CI.
  There is a scheduled workflow that does this, see
  .github/workflows/update-lock-files.yml. This is still useful to run this
  script when the automated PR fails and for example some packages need to
  be pinned. You can add the pins to this script, run it, and open a PR with
  the changes.
- bump minimum dependencies in src/egttools/_min_dependencies.py. Running this
  script will update both the CI environment files and associated lock files.
  You can then open a PR with the changes.
- pin some packages to an older version by adding them to the
  default_package_constraints variable. This is useful when regressions are
  introduced in our dependencies, this has happened for example with pytest 7
  and coverage 6.3.

Environments are conda environment.yml or pip requirements.txt. Lock files are
conda-lock lock files or pip-compile requirements.txt.

pip requirements.txt are used when we install some dependencies (e.g. numpy and
scipy) with apt-get and the rest of the dependencies (e.g. pytest and joblib)
with pip.

To run this script you need:
- conda-lock. The version should match the one used in the CI in
  src/egttools/_min_dependencies.py
- pip-tools

To only update the environment and lock files for specific builds, you can use
the command line argument `--select-build` which will take a regex. For example,
to only update the documentation builds you can use:
`python build_tools/update_environments_and_lock_files.py --select-build doc`
"""

import json
import logging
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import click
from jinja2 import Environment
from packaging.version import Version

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

TRACE = logging.DEBUG - 5

common_dependencies_without_coverage = [
    "python",
    "numpy",
    "blas",
    "scipy",
    "matplotlib",
    "seaborn",
    "networkx",
    "pytest",
    "pytest-xdist",
    "pip",
    "ninja",
]

common_dependencies = common_dependencies_without_coverage + [
    "pytest-cov",
    "coverage",
]

docstring_test_dependencies = ["sphinx", "numpydoc"]

default_package_constraints = {}


def remove_from(alist, to_remove):
    return [each for each in alist if each not in to_remove]


build_metadata_list = [
    {
        "name": "pylatest_conda_forge_mkl_linux-64",
        "type": "conda",
        "tag": "main-ci",
        "folder": "build_tools/github",
        "platform": "linux-64",
        "channels": ["conda-forge"],
        "conda_dependencies": common_dependencies
                              + [
                                  "ccache",
                              ],
        "package_constraints": {
            "blas": "[build=mkl]",
        },
    },
    {
        "name": "pylatest_conda_forge_mkl_osx-64",
        "type": "conda",
        "tag": "main-ci",
        "folder": "build_tools/github",
        "platform": "osx-64",
        "channels": ["conda-forge"],
        "conda_dependencies": common_dependencies
                              + [
                                  "ccache",
                                  "compilers",
                                  "llvm-openmp",
                              ],
        "package_constraints": {
            "blas": "[build=mkl]",
        },
    },
]


def execute_command(command_list):
    logger.debug(" ".join(command_list))
    proc = subprocess.Popen(
        command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    out, err = proc.communicate()
    out, err = out.decode(errors="replace"), err.decode(errors="replace")

    if proc.returncode != 0:
        command_str = " ".join(command_list)
        raise RuntimeError(
            "Command exited with non-zero exit code.\n"
            "Exit code: {}\n"
            "Command:\n{}\n"
            "stdout:\n{}\n"
            "stderr:\n{}\n".format(proc.returncode, command_str, out, err)
        )
    logger.log(TRACE, out)
    return out


def get_package_with_constraint(package_name, build_metadata, uses_pip=False):
    build_package_constraints = build_metadata.get("package_constraints")
    if build_package_constraints is None:
        constraint = None
    else:
        constraint = build_package_constraints.get(package_name)

    constraint = constraint or default_package_constraints.get(package_name)

    if constraint is None:
        return package_name

    comment = ""
    if constraint == "min":
        constraint = execute_command(
            [sys.executable, "src/egttools/_min_dependencies.py", package_name]
        ).strip()
        comment = "  # min"

    if re.match(r"\d[.\d]*", constraint):
        equality = "==" if uses_pip else "="
        constraint = equality + constraint

    return f"{package_name}{constraint}{comment}"


environment = Environment(trim_blocks=True, lstrip_blocks=True)
environment.filters["get_package_with_constraint"] = get_package_with_constraint


def get_conda_environment_content(build_metadata):
    template = environment.from_string(
        """
# DO NOT EDIT: this file is generated from the specification found in the
# following script to centralize the configuration for CI builds:
# build_tools/update_environments_and_lock_files.py
channels:
  {% for channel in build_metadata['channels'] %}
  - {{ channel }}
  {% endfor %}
dependencies:
  {% for conda_dep in build_metadata['conda_dependencies'] %}
  - {{ conda_dep | get_package_with_constraint(build_metadata) }}
  {% endfor %}
  {% if build_metadata['pip_dependencies'] %}
  - pip
  - pip:
  {% for pip_dep in build_metadata.get('pip_dependencies', []) %}
    - {{ pip_dep | get_package_with_constraint(build_metadata, uses_pip=True) }}
  {% endfor %}
  {% endif %}""".strip()
    )
    return template.render(build_metadata=build_metadata)


def write_conda_environment(build_metadata):
    content = get_conda_environment_content(build_metadata)
    build_name = build_metadata["name"]
    folder_path = Path(build_metadata["folder"])
    output_path = folder_path / f"{build_name}_environment.yml"
    logger.debug(output_path)
    output_path.write_text(content)


def write_all_conda_environments(build_metadata_list):
    for build_metadata in build_metadata_list:
        write_conda_environment(build_metadata)


def conda_lock(environment_path, lock_file_path, platform):
    execute_command(
        [
            "conda-lock",
            "lock",
            "--mamba",
            "--kind",
            "explicit",
            "--platform",
            platform,
            "--file",
            str(environment_path),
            "--filename-template",
            str(lock_file_path),
        ]
    )


def create_conda_lock_file(build_metadata):
    build_name = build_metadata["name"]
    folder_path = Path(build_metadata["folder"])
    environment_path = folder_path / f"{build_name}_environment.yml"
    platform = build_metadata["platform"]
    lock_file_basename = build_name
    if not lock_file_basename.endswith(platform):
        lock_file_basename = f"{lock_file_basename}_{platform}"

    lock_file_path = folder_path / f"{lock_file_basename}_conda.lock"
    conda_lock(environment_path, lock_file_path, platform)


def write_all_conda_lock_files(build_metadata_list):
    for build_metadata in build_metadata_list:
        logger.info(f"# Locking dependencies for {build_metadata['name']}")
        create_conda_lock_file(build_metadata)


def get_pip_requirements_content(build_metadata):
    template = environment.from_string(
        """
# DO NOT EDIT: this file is generated from the specification found in the
# following script to centralize the configuration for CI builds:
# build_tools/update_environments_and_lock_files.py
{% for pip_dep in build_metadata['pip_dependencies'] %}
{{ pip_dep | get_package_with_constraint(build_metadata, uses_pip=True) }}
{% endfor %}""".strip()
    )
    return template.render(build_metadata=build_metadata)


def write_pip_requirements(build_metadata):
    build_name = build_metadata["name"]
    content = get_pip_requirements_content(build_metadata)
    folder_path = Path(build_metadata["folder"])
    output_path = folder_path / f"{build_name}_requirements.txt"
    logger.debug(output_path)
    output_path.write_text(content)


def write_all_pip_requirements(build_metadata_list):
    for build_metadata in build_metadata_list:
        write_pip_requirements(build_metadata)


def pip_compile(pip_compile_path, requirements_path, lock_file_path):
    execute_command(
        [
            str(pip_compile_path),
            "--upgrade",
            str(requirements_path),
            "-o",
            str(lock_file_path),
        ]
    )


def write_pip_lock_file(build_metadata):
    build_name = build_metadata["name"]
    python_version = build_metadata["python_version"]
    environment_name = f"pip-tools-python{python_version}"
    # To make sure that the Python used to create the pip lock file is the same
    # as the one used during the CI build where the lock file is used, we first
    # create a conda environment with the correct Python version and
    # pip-compile and run pip-compile in this environment

    execute_command(
        [
            "conda",
            "create",
            "-c",
            "conda-forge",
            "-n",
            f"pip-tools-python{python_version}",
            f"python={python_version}",
            "pip-tools",
            "-y",
        ]
    )

    json_output = execute_command(["conda", "info", "--json"])
    conda_info = json.loads(json_output)
    environment_folder = [
        each for each in conda_info["envs"] if each.endswith(environment_name)
    ][0]
    environment_path = Path(environment_folder)
    pip_compile_path = environment_path / "bin" / "pip-compile"

    folder_path = Path(build_metadata["folder"])
    requirement_path = folder_path / f"{build_name}_requirements.txt"
    lock_file_path = folder_path / f"{build_name}_lock.txt"
    pip_compile(pip_compile_path, requirement_path, lock_file_path)


def write_all_pip_lock_files(build_metadata_list):
    for build_metadata in build_metadata_list:
        logger.info(f"# Locking dependencies for {build_metadata['name']}")
        write_pip_lock_file(build_metadata)


def check_conda_lock_version():
    # Check that the installed conda-lock version is consistent with _min_dependencies.
    expected_conda_lock_version = execute_command(
        [sys.executable, "src/egttools/_min_dependencies.py", "conda-lock"]
    ).strip()

    installed_conda_lock_version = version("conda-lock")
    if installed_conda_lock_version != expected_conda_lock_version:
        raise RuntimeError(
            f"Expected conda-lock version: {expected_conda_lock_version}, got:"
            f" {installed_conda_lock_version}"
        )


def check_conda_version():
    # Avoid issues with glibc (https://github.com/conda/conda-lock/issues/292)
    # or osx (https://github.com/conda/conda-lock/issues/408) virtual package.
    # The glibc one has been fixed in conda 23.1.0 and the osx has been fixed
    # in conda 23.7.0.
    conda_info_output = execute_command(["conda", "info", "--json"])

    conda_info = json.loads(conda_info_output)
    conda_version = Version(conda_info["conda_version"])

    if Version("22.9.0") < conda_version < Version("23.7"):
        raise RuntimeError(
            f"conda version should be <= 22.9.0 or >= 23.7 got: {conda_version}"
        )


@click.command()
@click.option(
    "--select-build",
    default="",
    help=(
            "Regex to filter the builds we want to update environment and lock files. By"
            " default all the builds are selected."
    ),
)
@click.option(
    "--skip-build",
    default=None,
    help="Regex to skip some builds from the builds selected by --select-build",
)
@click.option(
    "--select-tag",
    default=None,
    help=(
            "Tag to filter the builds, e.g. 'main-ci' or 'scipy-dev'. "
            "This is an additional filtering on top of --select-build."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print commands executed by the script",
)
@click.option(
    "-vv",
    "--very-verbose",
    is_flag=True,
    help="Print output of commands executed by the script",
)
def main(select_build, skip_build, select_tag, verbose, very_verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)
    if very_verbose:
        logger.setLevel(TRACE)
        handler.setLevel(TRACE)
    check_conda_lock_version()
    check_conda_version()

    filtered_build_metadata_list = [
        each for each in build_metadata_list if re.search(select_build, each["name"])
    ]
    if select_tag is not None:
        filtered_build_metadata_list = [
            each for each in build_metadata_list if each["tag"] == select_tag
        ]
    if skip_build is not None:
        filtered_build_metadata_list = [
            each
            for each in filtered_build_metadata_list
            if not re.search(skip_build, each["name"])
        ]

    selected_build_info = "\n".join(
        f"  - {each['name']}, type: {each['type']}, tag: {each['tag']}"
        for each in filtered_build_metadata_list
    )
    selected_build_message = (
        f"# {len(filtered_build_metadata_list)} selected builds\n{selected_build_info}"
    )
    logger.info(selected_build_message)

    filtered_conda_build_metadata_list = [
        each for each in filtered_build_metadata_list if each["type"] == "conda"
    ]

    if filtered_conda_build_metadata_list:
        logger.info("# Writing conda environments")
        write_all_conda_environments(filtered_conda_build_metadata_list)
        logger.info("# Writing conda lock files")
        write_all_conda_lock_files(filtered_conda_build_metadata_list)

    filtered_pip_build_metadata_list = [
        each for each in filtered_build_metadata_list if each["type"] == "pip"
    ]
    if filtered_pip_build_metadata_list:
        logger.info("# Writing pip requirements")
        write_all_pip_requirements(filtered_pip_build_metadata_list)
        logger.info("# Writing pip lock files")
        write_all_pip_lock_files(filtered_pip_build_metadata_list)


if __name__ == "__main__":
    main()
