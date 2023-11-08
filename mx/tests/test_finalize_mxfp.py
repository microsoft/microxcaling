"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import sys
import argparse

from mx import (
    add_mx_args,
    get_mx_specs,
    finalize_mx_specs
)


def test_mx_args():
    parser = argparse.ArgumentParser()
    parser = add_mx_args(parser)
    args = parser.parse_args(args=[])
    specs = get_mx_specs(args)
    assert(specs == None)

    parser = argparse.ArgumentParser()
    parser = add_mx_args(parser)
    args = parser.parse_args(args=["--bfloat", "16"])
    specs = get_mx_specs(args)
    assert(specs != None and specs["bfloat"] == 16)


def test_finalize_mx_none():
    s = {}
    assert(finalize_mx_specs(s) == None)

    s = {}
    s["w_elem_format"] = None
    s["a_elem_format"] = None
    assert(finalize_mx_specs(s) == None)

    s = {}
    s["w_elem_format"] = "fp8_e4m3"
    s["a_elem_format"] = "int8"
    assert(finalize_mx_specs(s) != None)
