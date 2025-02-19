
import random
import numpy
import matplotlib.pyplot as plt

numpy.set_printoptions(formatter={'all': lambda x: "{:>9.2f}".format(x)})

SIMULATION_RUNS = 1000

# Probability of sequence fork
p = 0.05

query_len_max = 32
query_len_min = 20

p_values = 12
p_max = 0.5
p_min = 0.1

def simulate_memory(query_len, p):

    L = [1] * query_len
    for i in range(1, query_len):

        total = L[i-1]
        for j in range(L[i-1]):
            if random.random() < p:
                total += 1

        L[i] = total

    # Memory required by the naive seq. aligment algo
    naive_memory = query_len * query_len * L[-1]

    # Memory required by us
    memory = sum([query_len * l for l in L])
    memory += sum(L) # We also have to store the offsets

    return L, naive_memory, memory



def simulate(query_len, p):
    
    ratios = []
    seq_lens = []
    naive_memories = []
    memories = []

    for _ in range(SIMULATION_RUNS):
        L, naive_memory, memory = simulate_memory(query_len, p)
        ratios.append(memory / naive_memory)
        naive_memories.append(naive_memory)
        memories.append(memory)
        seq_lens.append(L[-1])


    mean_ratio = sum(ratios) / len(ratios)
    mean_seq_len = sum(seq_lens) / len(seq_lens)
    mean_memory = sum(memories) / len(memories)
    mean_naive_memory = sum(naive_memories) / len(naive_memories)

    return mean_ratio, mean_seq_len, mean_memory, mean_naive_memory

def plot_mat(data, ax, title, cell_text_fn):
    
    y = numpy.arange(0, query_len_max - query_len_min + 1)
    x = numpy.arange(0, p_values)
    
    query_len_list = list(range(query_len_min, query_len_max+1))
    p_list = numpy.linspace(p_min, p_max, num=p_values)
    
    ax.matshow(data)
    ax.set_title(title)

    ax.set_ylabel(f"query len. (from {query_len_min} to {query_len_max})")
    ax.set_yticks(y)
    ax.set_yticklabels(query_len_list)
    ax.set_xlabel(f"p (from {p_min} to {p_max})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in p_list])

    mx, my = numpy.meshgrid(x, y)
    for col_val, row_val in zip(mx.flatten(), my.flatten()):
        ctext, color = cell_text_fn(col_val, row_val)
        ax.text(col_val, row_val, ctext, va='center', ha='center', color=color)


savings = numpy.zeros((query_len_max - query_len_min + 1, p_values))
naive_memory = numpy.zeros_like(savings)
memory = numpy.zeros_like(savings)
seqs = numpy.zeros_like(savings)

query_len_list = list(range(query_len_min, query_len_max+1))
p_list = numpy.linspace(p_min, p_max, num=p_values)

for q_i, query_len in enumerate(query_len_list):
    for p_i, p in enumerate(p_list):
        ratio, seq_len, mean_mem, naive_mem = simulate(query_len, p)
        savings[q_i, p_i] = (1.0 - ratio) * 100
        naive_memory[q_i, p_i] = naive_mem
        memory[q_i, p_i] = mean_mem
        seqs[q_i, p_i] = seq_len

y = numpy.arange(0, query_len_max - query_len_min + 1)
x = numpy.arange(0, p_values)

fig, axs = plt.subplots(2, 4)

def savings_fn(col, row):
    return f'{savings[row,col]:.1f}', 'black'

plot_mat(savings, axs[0,0], "Memory savings (%)", savings_fn)

def seqs_fn(col, row, s):
    value = s[row,col]

    ctext = f"{int(value)}"
    if value > 100:
        ctext = f"{value/1000:.1f}K"
    if value > 1000:
        ctext = f"{int(value/1000)}K"

    color = 'black'
    if int(value) == 0:
        color = 'red'

    return ctext, color

plot_mat(numpy.log10(seqs), axs[0,1], "Number of sequences", lambda col, row: seqs_fn(col, row, seqs))

shared_memory_limits = [48000, 96000]
shared_memory_colors = ['orange', 'red']
shared_memory_element_bytes = 0.25 # possibilities: [0.25, 0.5, 1, 2, 4]

def memory_fn(col, row, mem):
    value = mem[row,col] * shared_memory_element_bytes 
    
    ctext = 'A'
    color = 'black'

    if value > 48000:
        ctext = '48K'
        color = 'orange'
    if value > 96000:
        ctext = '96K'
        color = 'red'
    
    return ctext, color

plot_mat(numpy.log10(memory), axs[0,2], "Memory usage with limits", lambda col, row: memory_fn(col, row, memory))
plot_mat(numpy.log10(naive_memory), axs[0,3], "Naive memory usage with limits", lambda col, row: memory_fn(col, row, naive_memory))

# How many prefix-trie in shared memory?
block_sizes_48K = 48000.0 / (memory * shared_memory_element_bytes)
block_sizes_96K = 96000.0 / (memory * shared_memory_element_bytes)

def block_sizes_fn(col, row, bsm):
    value = bsm[row,col]
    color = 'white' if value > 1.0 else 'red'
    return f"{int(value)}", color

plot_mat(block_sizes_48K, axs[1,0], "Block size to 48K", lambda col, row: block_sizes_fn(col,row,block_sizes_48K))
plot_mat(block_sizes_96K, axs[1,2], "Block size to 96K", lambda col, row: block_sizes_fn(col,row,block_sizes_96K))

seq_48K = block_sizes_48K.astype(int) * seqs
seq_96K = block_sizes_96K.astype(int) * seqs

plot_mat(seq_48K, axs[1,1], "Total sequences to 48K", lambda col, row: seqs_fn(col,row,seq_48K))
plot_mat(seq_96K, axs[1,3], "Total sequences to 96K", lambda col, row: seqs_fn(col,row,seq_96K))

fig.suptitle(f"shared_memory_element_bytes: {shared_memory_element_bytes}")
plt.show()
