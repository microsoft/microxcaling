"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

from typing import List, Tuple, Optional, overload, Union, cast

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from .activations import sigmoid, tanh
from .linear import linear
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test
from .simd_ops import simd_add, simd_mul

class LSTM(torch.nn.LSTM):
    """
    If num_layers > 1, the input of the n'th layer is the
    hidden state of the previous layer with a dropout applied.

    If proj_size > 0, Hc changes from hidden_size to proj_size.
    The hidden state is multiplied by a learnable projection
    h_t = W_{hr}h_t right before the output.

    Notation:
        H    = hidden size
        H_in = input hidden size (input_size)
        H_c  = cell hidden size
        L    = sequence length
        N    = batch size
        D    = 2 if bidirectional else 1
   
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

    Inputs: input, (h_0, c_0)
        * **input**: (L, N, H_{in}) when batch_first=False or (N, L, H_{in})
          when batch_first=True
          Does not support torch.nn.utils.rnn.packed_sequence.
        * **h_0**: Initial hidden state (D * num_layers, N, H_{out})
          Defaults to zeros if (h_0, c_0) is not provided.
        * **c_0**: Initial cell state (D * num_layers, N, H_{cell})
          Defaults to zeros if (h_0, c_0) is not provided.

    Outputs: output, (h_n, c_n)
        * **output**: (L, N, D * H_{out}) when batch_first=False or (N, L, D * H_{out})
          when batch_first=True
        * **h_n**: Final hidden state (D * num_layers, N, H_{out})
        * **c_n**: Final cell state (D * num_layers, N, H_{cell})
    """

    def __init__(self, *args, mx_specs=None, name='', **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)

        # We don't have support for these configurations
        assert(self.batch_first == False)
        assert(self.proj_size == 0)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            assert 'a100' not in gpu_name.lower() and 'h100' not in gpu_name.lower(), "Quantization errors are high in LSTM for A100 and H100 GPUs."

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)
        self.mx_specs = apply_mx_specs(mx_specs)
        self.name = name

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)
 
    def _cell(self, xw, hc, layer, reverse=False):
        """ Applies a single LSTM cell, without input mamtuls
            xw:         pre-projected input
            hc:         tuple of hidden and cell states
            layer:      layer index
            reverse:    forward or reverse weight
            Produces the output hidden and cell states
        """
        if not reverse:
            W_h = getattr(self, 'weight_hh_l%d' % layer)
            b_h = getattr(self, 'bias_hh_l%d' % layer, None)
        else:
            W_h = getattr(self, 'weight_hh_l%d_reverse' % layer)
            b_h = getattr(self, 'bias_hh_l%d_reverse' % layer, None)

        # h,c (1, N, Hc)
        h,c = hc
        # gates (1, N, 4*Hc)
        linear_out = linear(h, W_h, bias=b_h, mx_specs=self.mx_specs,
                            name=self.name+'.HiddenLinear')
        gates = simd_add(xw, linear_out, mx_specs=self.mx_specs) 
        gates = gates.chunk(4,dim=-1)

        i = sigmoid(gates[0], mx_specs=self.mx_specs)
        f = sigmoid(gates[1], mx_specs=self.mx_specs)
        g = tanh(gates[2], mx_specs=self.mx_specs)
        o = sigmoid(gates[3], mx_specs=self.mx_specs)
        fc = simd_mul(f, c, mx_specs=self.mx_specs)
        ig = simd_mul(i, g, mx_specs=self.mx_specs)
        c = simd_add(fc, ig, mx_specs=self.mx_specs)
        tanh_c = tanh(c, mx_specs=self.mx_specs)
        h = simd_mul(o, tanh_c, mx_specs=self.mx_specs)
        return h,c

    def _proj_input(self, x, l, bidir=False):
        """ Applies the input matmul of an LSTM cell """
        y = linear(x,
                getattr(self, 'weight_ih_l%d'%l),
                bias=getattr(self, 'bias_ih_l%d'%l, None),
                mx_specs=self.mx_specs,
                name=self.name+'.InputLinear')

        y_r = None
        if bidir:
            y_r = linear(x,
                    getattr(self, 'weight_ih_l%d_reverse'%l),
                    bias=getattr(self, 'bias_ih_l%d_reverse'%l, None),
                    mx_specs=self.mx_specs,
                    name=self.name+'.InputReverseLinear')

        return y, y_r

    def _hx_slice(self, hx, s, e):
        """ Slices a (hidden state, cell state) tuple """
        h = hx[0][:, s:e]
        c = hx[1][:, s:e]
        return (h, c)
    
    def _hx_squeeze(self, hx, dim):
        """ Squeezes a (hidden state, cell state) tuple """
        return (hx[0].squeeze(dim), hx[1].squeeze(dim))

    def _hx_cat(self, hxs, dim):
        """ Concatenates a list of (hidden state, cell state) tuples """
        hs = torch.cat([hx[0] for hx in hxs], dim=dim)
        cs = torch.cat([hx[1] for hx in hxs], dim=dim)
        return hs, cs

    def _hidden_proj_packed(self, hx, inputs_w, batch_sizes, layer):
        """
        Computes hidden projections and step outputs for packed sequences.
        The logic is ported over from the PackedLayer struct here:
        https://github.com/zdevito/ATen/blob/master/aten/src/ATen/native/RNN.cpp
        """
        last_batch_size = batch_sizes[0]
        input_offset = 0
        hiddens = []
        step_outputs = []

        for i in range(len(batch_sizes)):
            batch_size = batch_sizes[i]
            step_input = inputs_w[input_offset:input_offset+batch_size]
            input_offset += batch_size
            dec = last_batch_size - batch_size
            if dec > 0:
                temp = self._hx_slice(hx, last_batch_size - dec, last_batch_size)
                hiddens.append(temp)
                hx = self._hx_slice(hx, 0, last_batch_size - dec)

            last_batch_size = batch_size
            hx = self._cell(step_input, hx, layer, reverse=False)
            step_outputs.append(hx[0])
        
        hiddens.append(hx)
        hiddens.reverse()
        hx = self._hx_cat(hiddens, dim=1)
        step_outputs = torch.cat(step_outputs, dim=1).squeeze(0)
        return hx, step_outputs
    
    def _hidden_proj_packed_r(self, hx_r, inputs_w, batch_sizes, layer):
        """
        Computes hidden projections and step outputs for packed sequences for
        the reversed sequence. Used for bidirectional LSTMs. The logic is
        ported over from the ReversedPackedLayer struct here:
        https://github.com/zdevito/ATen/blob/master/aten/src/ATen/native/RNN.cpp
        """
        last_batch_size = batch_sizes[-1]
        input_offset = len(inputs_w)
        hidden_r = self._hx_slice(hx_r, 0, batch_sizes[-1])
        step_outputs_r = []

        for i in range(len(batch_sizes) - 1, -1, -1):
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden_r = self._hx_cat([hidden_r, 
                    self._hx_slice(hx_r, last_batch_size, batch_size)], dim=1)
            step_input_r = inputs_w[input_offset-batch_size:input_offset]
            input_offset -= batch_size
            last_batch_size = batch_size
            hidden_r = self._cell(step_input_r, hidden_r, layer, reverse=True)
            step_outputs_r.append(hidden_r[0])
        
        step_outputs_r.reverse()
        step_outputs_r = torch.cat(step_outputs_r, dim=1).squeeze(0)
        return hidden_r, step_outputs_r

    def forward(self,
                input: PackedSequence,
                hx: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        """ RZ: torch LSTM accepts PackedSequence and Tensor inputs.
                We only accept Tensor and error out on PackedSequence.
        """
        if isinstance(input, PackedSequence):
            raise ValueError('mx LSTM does not support PackedSequence')
        else:
            raise Exception('Something is wrong!')

    def forward(self, input, hx=None):
        if self.mx_none:
            return super().forward(input, hx)
            
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)

        expected_hs = self.get_expected_hidden_size(input, batch_sizes)

        #--------------------------------------------------
        # LSTM
        #
        # input (L, N, Hin)
        # h_n (D*num_layers, N, Hout)
        # c_n (D*num_layers, N, H)
        # output (L, N, D*Hout)
        #
        # Note that L and N axes are swapped if batch_first=True
        #--------------------------------------------------
        D = 2 if self.bidirectional else 1
        H = hx[0].shape[-1]     # hidden size
        L = input.shape[0]      # seq length
        N = input.shape[1]      # batch size

        def _attr(s):
            return getattr(self, s)

        # Chunk the concat'd hidden/cell states
        # Each chunked hidden state is (1, N, H)
        hs, cs = hx
        assert(hs.shape[0] == D*self.num_layers)
        assert(cs.shape[0] == D*self.num_layers)
        hs = hs.unsqueeze(1).unbind(0)
        cs = cs.unsqueeze(1).unbind(0)

        # These will store the hidden outputs from the last
        # timestep in each layer
        final_hs = []
        final_cs = []

        for l in range(self.num_layers):
            # Initial hidden/cell states for this layer
            hx = (hs[D*l], cs[D*l])
            if self.bidirectional:
                hx_r = (hs[D*l+1], cs[D*l+1])

            #----------------------------------------------
            # Input projections
            if l == 0:
                # Single fused matmul for all input timesteps
                # inputs are (L, N, Hin)
                inputs_w, inputs_w_r = self._proj_input(
                        input, l, bidir=self.bidirectional)
            else:
                inputs_w, inputs_w_r = self._proj_input(
                        step_outputs, l, bidir=self.bidirectional)

            #----------------------------------------------
            # Hidden projections
            if isinstance(orig_input, PackedSequence):
                hx, step_outputs = self._hidden_proj_packed(hx, 
                    inputs_w, batch_sizes, l)

                if self.bidirectional:
                    hx_r, step_outputs_r = self._hidden_proj_packed_r(hx_r, 
                        inputs_w_r, batch_sizes, l)
            else:
                step_outputs = []
                step_outputs_r = []

                for i in range(L):
                    # inputs_w[i] and hx[0] is (1, N, Hout)
                    hx = self._cell(inputs_w[i], hx, l, reverse=False)
                    step_outputs.append(hx[0])

                    if self.bidirectional:
                        # Inputs and outputs in reverse order
                        hx_r = self._cell(inputs_w_r[-1-i], hx_r, l,
                            reverse=True)
                        step_outputs_r.insert(0, hx_r[0])
                
                step_outputs = torch.cat(step_outputs, dim=0)
                if self.bidirectional:
                    step_outputs_r = torch.cat(step_outputs_r, dim=0)

            if not self.bidirectional:
                final_hs.append(hx[0])
                final_cs.append(hx[1])
            else:
                final_hs.append(torch.cat((hx[0], hx_r[0]), dim=0))
                final_cs.append(torch.cat((hx[1], hx_r[1]), dim=0))

                # If bidir, each layer's output is the concat
                # of forward and backward outputs
                assert len(step_outputs) == len(step_outputs_r)
                step_outputs = torch.cat((step_outputs, step_outputs_r), dim=-1)
                

            #----------------------------------------------
            # Dropout
            if l != self.num_layers-1 and self.dropout > 0:
                step_outputs = self.dropout_layer(step_outputs)

        hidden = (torch.cat(final_hs, dim=0),
                  torch.cat(final_cs, dim=0))

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(step_outputs, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return step_outputs, self.permute_hidden(hidden, unsorted_indices)
