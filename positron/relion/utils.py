#!/usr/bin/env python

"""
Module for RELION data management
"""

import os


def find_project_root(from_path: str, file_relative_path: str) -> str:
    """
    Searches for the Relion project root starting at from_path and iterate through parent directories
    till file_relative_path is found as a relative sub path or till filesystem root is found, at which
    point a RuntimeException is raise.

    :param from_path: starting search from this path
    :param file_relative_path: searching to find this relative path as a file
    """
    current_path = os.path.abspath(from_path)
    while True:
        trial_path = os.path.join(current_path, file_relative_path)
        if os.path.isfile(trial_path):
            return current_path
        if current_path == os.path.dirname(current_path):  # At filesystem root
            raise RuntimeError(f"Relion project directory could not be found from the subdirectory: {from_path}")
        current_path = os.path.dirname(current_path)