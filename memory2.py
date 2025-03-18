import random
import numpy
import matplotlib.pyplot as plt

def memory_naive(guide_len, depth, seq_count, elem_bytes=1):
    
    # Memory of the DPT table
    dpt = depth * guide_len * seq_count
    # Memory of the sequences
    seq = seq_count * depth

    return (dpt + seq) * elem_bytes

def memory_prefix_tree(guide_len, depth, seq_count, elem_bytes=1):
    
    L = [0] * depth
    L[0] = seq_count

    for layer in range(1, depth):
        L[layer] = max(1, L[layer - 1] // 4)

    elements = sum(L)
    
    # Memory of the tree DPT table
    dpt = elements * guide_len
    # Memory required by the tree
    seq = elements * 2

    # Max diagonal length
    max_diag = sum(L)

    return ((dpt + seq) * elem_bytes, max_diag)


GUIDE_LEN = 20
ELEM_BYTES = 4
REF_LEN = 24

print(f"guide_len: {GUIDE_LEN}, ref_len: {REF_LEN}, bytes per element: {ELEM_BYTES}")
for seq_count in range(32, 257, 16):
    if seq_count == 0:
        seq_count = 1

    mn = memory_naive(GUIDE_LEN, REF_LEN, seq_count, ELEM_BYTES)
    mt, max_diag = memory_prefix_tree(GUIDE_LEN, REF_LEN, seq_count, ELEM_BYTES) 
    savings = (1.0 - (mt / mn)) * 100.0

    print(f"\t[{seq_count:>3}] naive: {mn/1024:>6.2f}KB, tree: {mt/1024:>5.2f}KB (-{savings:.2f}%, max_diag: {max_diag})")
