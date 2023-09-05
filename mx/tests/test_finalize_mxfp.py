"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import sys

from mx.specs import apply_mx_specs, finalize_mx_specs


def test_finalize_mx_none():
    s = apply_mx_specs({})
    assert(finalize_mx_specs(s) == None)

    s = apply_mx_specs({})
    s["w_elem_format"] = None
    s["a_elem_format"] = None
    assert(finalize_mx_specs(s) == None)

    s = apply_mx_specs({})
    s["w_elem_format"] = "fp8_e4m3"
    s["a_elem_format"] = "int8"
    assert(finalize_mx_specs(s) != None)
