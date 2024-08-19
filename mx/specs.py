"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Name: specs.py

This module defines MxSpecs and functions to parse it

Classes:
    MxSpecs

Functions:
    get_default_mx_specs
    get_backwards_mx_specs
    apply_mx_specs
    add_mx_args
    finalize_mx_specs
    get_mx_specs
    mx_assert_test

Usage Notes:
    Some spec options depend on others, like "a_elem_format_bp_os" depends on
    "a_elem_format". How do we create mx_specs and handle the dependencies?

    1. Create a dict and pass it to finalize_mx_specs:
    ```
        specs = {
            'a_elem_format': 8,
            'a_elem_format_bp_os': 4,
        }

        # This will set a_elem_format_bp_ex to 8, but won't touch
        # a_elem_format_bp_os which you already set.
        specs = finalize_mx_specs(specs)
    ```

    2. Use add_mx_args + get_mx_specs with argparse:
    ```
        parser = argparse.ArgumentParser()
        parser = add_mx_args(parser)

        # You can also add your own args here

        args = parser.parse_args()

        specs = get_mx_specs(args)
    ```
"""

import os, torch
import collections
import argparse
import json
import traceback

# Change this to True to enable an assert test
# in every MX user-facing function and class
_ASSERT_MODE = os.environ.get('MX_ASSERT', 'False')


class MxSpecs(collections.UserDict):
    """
    Class for handling quantization parameters.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor inheriting from UserDict/dict.
        Args:
            *args:        Passing a dict will initialize using those entries.
        """
        super(MxSpecs, self).__init__(*args, **kwargs)

        # All MX spec options are listed here. Each spec maps to
        # its default value and a help string for its command line arg.
        #
        # The type of each spec determines how it's parsed from command
        # line args. Currently we support int, str, and bool by default.
        # Custom parsing can be implemented in add_mx_args below.
        # Added new mode RNE (round to nearest, round ties to nearest even) for float32 to bfloat16 quantization
        defaults = {
            "scale_bits": 0,

            "w_elem_format": None,
            "a_elem_format": None,
            "w_elem_format_bp": None,
            "a_elem_format_bp": None,
            "a_elem_format_bp_ex": None,
            "a_elem_format_bp_os": None,
            "mx_flush_fp32_subnorms": False,

            "shared_exp_method": "max",
            "block_size": 0,

            "bfloat": 0,
            "fp": 0,

            "bfloat_subnorms": True,

            "quantize_backprop": True,

            "round": "nearest",
            "round_m": "nearest",
            "round_weight": "nearest",
            "round_output": "nearest",
            "round_grad_weight": "nearest",
            "round_grad_input": "nearest",
            "round_mx_output": "nearest",
            "round_mx_input_grad_input": "nearest",
            "round_mx_weight_grad_input": "nearest",
            "round_mx_grad_output_grad_input": "nearest",
            "round_mx_input_grad_weight": "nearest",
            "round_mx_grad_output_grad_weight": "nearest",

            "softmax_exp2": False,
            "vec_use_exp2": False,
            "vec_use_recip": False,

            "custom_cuda": False,
        }

        self.help_strings = {
            "scale_bits": "Bits (sign + magnitude) to use for shared exponent/scale",
            "w_elem_format": "Weight MX elem format, one of {fp8_e5m2, fp8_e4m3, "
                             "fp6_e3m2, fp6_e2m3, fp4_e2m1, int8, int4}",
            "a_elem_format": "Activation MX elem format. See w_elem_format",
            "w_elem_format_bp": "Backpass weight MX elem format. See w_elem_format",
            "a_elem_format_bp": "Backpass stashed activation MX elem format. See w_elem_format",
            "a_elem_format_bp_ex": "Backpass act (grad) MX elem format. See w_elem_format",
            "a_elem_format_bp_os": "Backpass act (grad) MX elem format. See w_elem_format",
            "mx_flush_fp32_subnorms": "MX quantization flushes blocks with "
                                      "subnormal shared scale to zero",

            "shared_exp_method": "Shared exponent calculation method. " "Options: max, none",
            "block_size": "mx shared exponent block size",

            "bfloat": 
                "BfloatX format (8exp + sign + mantissa). Only one of bfloat or fp can be used",
            "fp":
                "fpX format (5exp + sign + mantissa). Only one of bfloat or fp can be used",

            "bfloat_subnorms": "Bfloat/FP supports subnorms",

            "quantize_backprop": "Enable mx/bfloat quantization on backward pass",

            "round": "Global rounding mode. Choices: nearest, floor",
            "round_m": "ADAM optimizer m and v rounding mode",
            "round_weight": "Weight bfloat rounding mode (W in WAGE)",
            "round_output": "Activation bfloat rounding mode (A in WAGE)",
            "round_grad_weight": "Weight update rounding mode (G in WAGE)",
            "round_grad_input": "Error gradient rounding mode (E in WAGE)",
            "round_mx_output": "Forward pass mx rounding mode",
            "round_mx_input_grad_input": "",
            "round_mx_weight_grad_input": "",
            "round_mx_grad_output_grad_input": "",
            "round_mx_input_grad_weight": "",
            "round_mx_grad_output_grad_weight": "",

            "softmax_exp2": "Softmax uses 2^x instead of e^x",
            "vec_use_exp2": "Use 2^x to compute e^x",
            "vec_use_recip": "Use 1/x to compute division",

            "custom_cuda": "Enable custom CUDA kernels for quantization",
        }

        for k in defaults:
            if k not in self.data.keys():
                self.data[k] = defaults[k]

        for k in self.data.keys():
            assert(k in self.help_strings.keys())

    def safe_json(self, indent=None):
        """
        Return json of parameters.
        """
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(self.data, indent=indent, default=default)

    def __str__(self):
        return self.safe_json(indent=4)


def get_default_mx_specs():
    """
    mx_specs to disable quantization.
    """
    specs = MxSpecs()

    return specs


def get_backwards_mx_specs(specs):
    """Return a no-quantize spec if quantize_backprop is False"""
    bspecs = specs.copy()

    if bspecs["quantize_backprop"] == False:
        bspecs["w_elem_format"] = None
        bspecs["a_elem_format"] = None
        bspecs["w_elem_format_bp"] = None
        bspecs["a_elem_format_bp"] = None
        bspecs["a_elem_format_bp_os"] = None
        bspecs["a_elem_format_bp_ex"] = None
        bspecs["block_size"] = 0

        bspecs["bfloat"] = 0
        bspecs["fp"] = 0

    return bspecs


def apply_mx_specs(mx_specs, default_mx_specs=None):
    """
    Returns a MxSpecs object. Parameters defined in mx_specs are used.
    For parameters not defined, they are pulled from default_mx_specs. If no
    default_mx_specs is specified, then get_default_mx_specs is used.
    """
    if not default_mx_specs:
        default_mx_specs = get_default_mx_specs()

    if not mx_specs:
        return default_mx_specs

    # Apply new parameters from mx_specs
    for k in mx_specs:
        if mx_specs[k] != None:
            if k not in default_mx_specs:
                raise KeyError(f"Unknown key '{k}' passed to mx specs")
            default_mx_specs[k] = mx_specs[k]

    return default_mx_specs


def add_mx_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Automatically adds MX args to an parser based on its
    default type and value"""
    group = parser.add_argument_group("mx", "MX specs")

    group.add_argument(
        "--mx_dir", type=str, default=None, help="Path to mx library"
    )

    # All arguments are added as default None except bools.
    # A newly parsed mx_specs object must have finalize_mx_specs
    # called on it before it can be used!
    #
    # Examples:
    #  scale_bits: 0             -->  --scale_bits default None
    #  round: 'nearest'          -->  --round default None
    #  custom_cuda: False        -->  --custom_cuda store_true
    #  quantize_backprop: True   -->  --no_quantize_backprop store_true

    default_specs = get_default_mx_specs()
    for k, v in default_specs.items():
        help_str = default_specs.help_strings[k]
        help_str = "No help string" if help_str == "" else help_str

        # Make sure elem_format is type str
        if k.find("elem_format") != -1:
            group.add_argument("--" + k, type=str, default=v, help=help_str)

        # Default False bool -> store_true arg
        elif type(v) == bool and v is False:
            group.add_argument("--" + k, action="store_true", help=help_str)

        # Default True bool -> no_ prepended store_true arg
        elif type(v) == bool and v is True:
            group.add_argument("--no_" + k, action="store_true", help=help_str)

        else:
            group.add_argument("--" + k, type=type(v), default=None, help=help_str)

    group.add_argument("--skip_early_exit", action="store_true",
                       help="Don't early exit if no quantization is specified", default=False)

    return parser


def finalize_mx_specs(specs, early_exit=True):
    """Some specs depend on others.
    This function should be called after parsing arguments into
    an MX specs object before using the specs.
    """
    # Early exit, works for 0 and None
    if (
        not specs.get("w_elem_format", 0) 
        and not specs.get("a_elem_format", 0)
        and not specs.get("w_elem_format_bp", 0)
        and not specs.get("a_elem_format_bp", 0)
        and not specs.get("a_elem_format_bp_os", 0)
        and not specs.get("a_elem_format_bp_ex", 0)
        and not specs.get("bfloat", 0)
        and not specs.get("fp", 0)
        and early_exit
    ):
        return None
    
    if specs.get('custom_cuda'):
        assert torch.cuda.is_available(), f"'custom_cuda' is only supported on CUDA devices."

    # Handle specs wihch depend on another base spec
    def assign_if_none(f1, f2):
        if (f1 not in specs or specs[f1] is None) and f2 in specs:
            specs[f1] = specs[f2]

    assign_if_none("w_elem_format_bp", "w_elem_format")
    assign_if_none("a_elem_format_bp", "a_elem_format")
    assign_if_none("a_elem_format_bp_os", "a_elem_format")
    assign_if_none("a_elem_format_bp_ex", "a_elem_format")

    assign_if_none("round_m", "round")
    assign_if_none("round_output", "round")
    assign_if_none("round_grad_weight", "round")
    assign_if_none("round_grad_input", "round")
    assign_if_none("round_weight", "round")
    assign_if_none("round_mx_output", "round")

    assign_if_none("round_mx_input_grad_input", "round_grad_input")
    assign_if_none("round_mx_weight_grad_input", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_input", "round_grad_input")
    assign_if_none("round_mx_input_grad_weight", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_weight", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_input", "round_grad_input")

    specs = apply_mx_specs(specs, get_default_mx_specs())
    return specs


def get_mx_specs(parsed_args: argparse.Namespace):
    """Call this on the output of parser.parse_args()"""
    default_specs = get_default_mx_specs()

    parsed_specs = {}
    for k, v in default_specs.items():
        if type(v) == bool and v is True:
            arg_k = "no_" + k
            if hasattr(parsed_args, arg_k):
                parsed_specs[k] = not parsed_args.__getattribute__(arg_k)
        else:
            if hasattr(parsed_args, k):
                parsed_specs[k] = parsed_args.__getattribute__(k)

    if hasattr(parsed_args, "skip_early_exit"):
        early_exit = not parsed_args.skip_early_exit
    else:
        early_exit = True

    return finalize_mx_specs(parsed_specs, early_exit=early_exit)


def mx_assert_test(mx_specs):
    if _ASSERT_MODE == "True" and mx_specs is None:
        stack = traceback.extract_stack()
        f1 = stack[-2]  # failing MX func
        f2 = stack[-3]  # call site

        msg = (
            "MX assert test failed!\n"
            + f"mx_specs is None in function {f1.name}\n"
            + f"Called from {f2.filename}, line {f2.lineno}\n"
            + f"  {f2.line}"
        )
        raise ValueError(msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser = add_mx_args(parser)

    args = parser.parse_args([])
    specs = get_mx_specs(args)
    assert specs is None

    args = parser.parse_args([])
    args.bfloat = 4
    specs = get_mx_specs(args)
    assert specs["bfloat"] == 4
    defaults = get_default_mx_specs()
    for k, v in specs.items():
        if k != "bfloat":
            assert defaults[k] == v, (k, defaults[k], v)
    for k, v in defaults.items():
        if k != "bfloat":
            assert specs[k] == v, (k, v, specs[k])

    print("Passed!")
