#!/usr/bin/env python

import argparse


def range_limited_float_type(mini=None, maxi=None):
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument (optional)
         maxi - maximum acceptable argument (optional)"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")

        if mini is not None and f < mini:
            raise argparse.ArgumentTypeError(f"must be greater than or equal to {mini}")
        if maxi is not None and f > maxi:
            raise argparse.ArgumentTypeError(f"must be less than or equal to {maxi}")

        return f

    # Return function handle to checking function
    return float_range_checker


def range_limited_int_type(mini=None, maxi=None):
    """Return function handle of an argument type function for
       ArgumentParser checking an integer range: mini <= arg <= maxi
         mini - minimum acceptable argument (optional)
         maxi - maximum acceptable argument (optional)"""

    # Define the function with default arguments
    def int_range_checker(arg):
        """New Type function for argparse - an integer within predefined range."""

        try:
            i = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be an integer")

        if mini is not None and i < mini:
            raise argparse.ArgumentTypeError(f"must be greater than or equal to {mini}")
        if maxi is not None and i > maxi:
            raise argparse.ArgumentTypeError(f"must be less than or equal to {maxi}")

        return i

    # Return function handle to checking function
    return int_range_checker