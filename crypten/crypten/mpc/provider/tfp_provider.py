#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import torch
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps, torch_stack
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
from crypten.cuda import CUDALongTensor
from .provider import TupleProvider


class TrustedFirstParty(TupleProvider):
    NAME = "TFP"

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        a = generate_random_ring_element(size0, device=device)
        b = generate_random_ring_element(size1, device=device)
        # a,b = CUDALongTensor(a), CUDALongTensor(b)
        c = getattr(torch, op)(a, b, *args, **kwargs)
        # print(op,size0,size1,c)
        a = ArithmeticSharedTensor(a, precision=0, src=0)
        b = ArithmeticSharedTensor(b, precision=0, src=0)
        c = ArithmeticSharedTensor(c, precision=0, src=0)

        return a, b, c

    def square(self, size, device=None):
        """Generate square double of given size"""
        r = generate_random_ring_element(size, device=device)
        r2 = r.mul(r)

        # Stack to vectorize scatter function
        stacked = torch_stack([r, r2])
        stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
        return stacked[0], stacked[1]

    def generate_binary_triple(self, size0, size1, device=None):
        """Generate xor triples of given size"""
        a = generate_kbit_random_tensor(size0, device=device)
        b = generate_kbit_random_tensor(size1, device=device)
        c = a & b

        a = BinarySharedTensor(a, src=0)
        b = BinarySharedTensor(b, src=0)
        c = BinarySharedTensor(c, src=0)

        return a, b, c

    def wrap_rng(self, size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        num_parties = comm.get().get_world_size()
        r = [
            generate_random_ring_element(size, device=device)
            for _ in range(num_parties)
        ]
        theta_r = count_wraps(r)

        shares = comm.get().scatter(r, 0)
        r = ArithmeticSharedTensor.from_shares(shares, precision=0)
        theta_r = ArithmeticSharedTensor(theta_r, precision=0, src=0)

        return r, theta_r

    def B2A_rng(self, size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        # generate random bit
        r = generate_kbit_random_tensor(size, bitlength=1, device=device)

        rA = ArithmeticSharedTensor(r, precision=0, src=0)
        rB = BinarySharedTensor(r, src=0)

        return rA, rB
