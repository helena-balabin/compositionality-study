# -*- coding: utf-8 -*-

"""Version information for :mod:`compositionality_study`.

Run with ``python -m compositionality_study.version``
"""

import os
from subprocess import CalledProcessError, check_output  # noqa: S404

__all__ = [
    "VERSION",
    "get_version",
    "get_git_hash",
]

VERSION = "0.0.1-dev"


def get_git_hash() -> str:
    """Get the :mod:`compositionality_study` git hash."""
    with open(os.devnull, "w") as devnull:
        try:
            ret = check_output(  # noqa: S603,S607
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=devnull,
            )
        except CalledProcessError:
            return "UNHASHED"
        else:
            return ret.strip().decode("utf-8")[:8]


def get_version(with_git_hash: bool = False):
    """Get the :mod:`compositionality_study` version string, including a git hash."""
    return f"{VERSION}-{get_git_hash()}" if with_git_hash else VERSION


if __name__ == "__main__":
    print(get_version(with_git_hash=True))  # noqa:T201
