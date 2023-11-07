# Run the FFN layer with MXFP6_e3m2
python ffn_mx.py --w_elem_format "fp6_e3m2" --a_elem_format "fp6_e3m2" --scale_bits 8 --block_size 32 --bfloat 16 --custom_cuda
