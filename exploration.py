

#
#
# A            layer_size
#
# 0 *          1
#   |
# 1 *          1
#   |
# 2 *---*      2
#   |   |
# 3 *-* *-*    4
#   | | | |
# 4 * * * *-*  5
#
# Span(A) = layer_size(4) = 5
#
#
# B            layer_size
#
# 0 *          1
#   |
# 1 *          1
#   |
# 2 *          1
#   |
# 3 *-*-*      3
#   | | |
# 4 * * *      3
#
# Span(B) = 3
#
#
# C            layer_size
#
# 0 *          1
#   |
# 1 *          1
#   |
# 2 *          1
#   |
# 3 *          1
#   |
# 4 *          1
#
# Span(C) = 1
#
#
# [1] Ordiniamo gli elementi del batch in ordine crescente di Span: C < B < A
# [2] Compattiamo tutti gli elementi del batch in oridine, inoltre salviamo l'id del branch
#     padre di ogni nodo nel layer precedente
#
#   layer_size |  end(prefix_sum) | beg(end - size) | PACKED     | GLOBAL_PARENT_OFFSET
#              |                  |                 |            |
#        C B A |           C B A  |           C B A |            |
#              |                  |                 |            |
#        1 1 1 |           1 2 3  |           0 1 2 | CBA        | - -   -
#        1 1 1 |           1 2 3  |           0 1 2 | CBA        | 0 1   2
#        1 1 2 |           1 2 4  |           0 1 2 | CBAA       | 0 1   22
#        1 3 4 |           1 4 8  |           0 1 4 | CBBBAAAA   | 0 111 0011
#        1 3 5 |           1 5 9  |           0 1 4 | CBBBAAAAA  | 0 123 45677
#
#                                                     012345678
#
# [3] Lineariziamo la loro memoria in base alla depth e creiamo la tabella di DP
#     per ogni regione compatta. Ogni warp ora puo lavorare in base al lato piu
#     lungo di ogni tabella (len(Q) = 32, buona lunghezza minima sempre presente).
#     Intrinsics dei warp possono essere usate per le dipendenze a questo livello.
#
#       CBA - CBA - CBAA - CBBBAAAA - CBBBAAAAA
#
#   Q1  111   222   3333   44444444   555555555
#   Q2  1X1   2X2   3XX3   4XXXXXX4   5XXXXXXX5
#   Q3  1X1   2X2   3XX3   4XXXXXX4   5XXXXXXX5
#   Q4  111   222   3333   44444444   555555555
#
#    W  |||   |||   ||||   --------   ---------
#    A  |||   |||   ||||   --------   ---------
#    R  |||   |||   ||||   --------   ---------
#    P  |||   |||   ||||   --------   ---------
#
#       123   123   1234   11111111   111111111
#       123   123   1234   22222222   222222222
#       123   123   1234   33333333   333333333
#       123   123   1234   44444444   444444444
#
#     Quando costruisco i batch e' opportuno unire abbastanza sample semplici (Span piccolo)
#     in modo da permettere ai warp di lavorare sempre in orizzontale, come nella quarta e
#     quinta tabella dell'esempio.
#
#     Potrei fare shuffle degli elementi del batch per avere gia un accesso in memoria ottimizzato.
#
#     1q1c 1q1b [1q1a] |
#     2q1c 2q1b [2q1a] 1q1c 1q1b [1q1a] |
#     3q1c 3q1b 3q1a 2q1c 2q1b (2q1a) 1q1c 1q1b 1q1a 1q1a |
#
#     Ogni elemento i a livello k (i, k), ha bisogno dell'elemento (i, k-1), (i - D[i], k-1) e (i - D[i], k - 2), se siamo prima
#     dell'anti-diagonale maggiore. D[i] e' la dimensione del layer precedente, che va calcolato.
#
#     Ogni riga puo essere processata in parallelo in modo indipendente
#
#     [2q3c 2q3b 2q3a 2q3a] ha bisogno di [1q3c 1q3b 1q3a 1q3a] (same size) e [1q2c 1q2b 1q2a], [2q2c 2q2b 2q2a] (diversa dimensione, virtuali).
#     Per le celle virtuali, il valore si puo calcolare con il GLOBAL_PARENT_OFFSET, che va sottratto ad ogni indirizzo di memoria.
#       -> PROBABILE CONFLITTO DI BANKS, AGGIUNGERE SHIFT-PADDING
#
#
# [4] Nel backtracking quando passo da un livello l al livello l-1 devo mappare la cella virtuale
#     alla cella reale usando il PARENT_OFFSET.
#
#    Quindi data c(i,j), dove i si muove lungo la direzione di Q e j invece attraversa le DP di livello avremo:
#
#
#       L4    L3    L2    L1    L0
#       AaAAA AaAA- Aa--- a---- a----   i   j
#       AaAAA AaAA- Aa--- a---- a----
#       AxAAA AxAA- Ax--- x---- x----   | /
#       AaAAA AaAA- Aa--- a---- a----   |/
#
#    In questo esempio le 'a' rappresentano una DP attraverso i layer del batch, quando passiamo dal layer 2 al layer 1 dobbiamo
#    mappare la cella 'x' (insieme al resto della DP) sulla slice del primo strand.
#
#       c(i, j-1) = layer_precedente(i, GLOBAL_PARENT_OFFSET[i])
#

if False:
    S = [
        'C', 'B', 'A',
        'C', 'B', 'A',
        'C', 'B', 'A', 'A',
        'C', 'B', 'B', 'B', 'A', 'A', 'A', 'A',
        'C', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A',
        'C', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A'
    ]

    L = [3, 3,  4,  8,  9, 9]
    Q = 'ABCDE'

    # Parent offset
    P = [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 2, 2, 3, 3, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 2, 2, 3, 3, 4, 4
    ]
else:
    S = [
        'A',
        'C', 'A',
        'C', 'D', 'S',
        'B', 'B', 'B',
    ]
    P = [
        0,
        0, 1,
        0, 1, 1,
        0, 0, 0
    ]
    Q = 'ACB'

    # Dimensione di ogni layer
    L = [1, 2, 3, 3]

# =========================================
# Easiest test case

SCORE_MATCH = 4
SCORE_INDEL = -1

def create_diag_sizes(L):
    result = []

    window_size = len(L)
    L_new = [0] * (window_size - 1) + L
    L_new = L_new + [0] * (window_size - 1)

    for i in range(0, len(L_new) - window_size + 1):
        result.append(sum(L_new[i:i+window_size]))

    return result


# Prefix-sum delle dimensioni di ogni layer
C = [sum(L[0:i]) for i in range(len(L))]

# Dimensione di ogni diagonale, creata tramite sliding-window
# di lunghezza 'len(L)' sulle dimensioni dei layer
D = create_diag_sizes(L)

# Offset in memoria di ogni tabella
TABLES = [c * len(Q) for c in C]

# Dynamic programming table
MEMORY = [0] * (len(Q) * sum(L))
M_BASE = float(L[-1] * len(Q) * len(L))
M_OURS = float(len(MEMORY))


print(f"Dimensione di ogni layer: {L}")
print(f"Prefix-sum delle dimensioni di ogni layer: {C}")
print(f"Dimensione di ogni diagonale: {D}")
print(f"Offset di ogni tabella in memoria: {TABLES}")
print(f"Dimensione memoria non ottimizzata: {M_BASE}")
print(f"Dimensione memoria: {M_OURS} (-{(1.0 - M_OURS/M_BASE)*100:.2f}%)")


def memory_table(t, i, j):
    """ Access in memory table with safeguards
    """

    # Magin root
    if t < 0 and i < 0:
        return 0

    # Margin left
    if t < 0:
        return -(i + 1)

    if i < 0:
        return -(t + 1)

    return MEMORY[TABLES[t] + i * L[t] + j]


def memory_table_set(t, i, j, value):
    MEMORY[TABLES[t] + i * L[t] + j] = value


def get_thread_position(tid, i_start, t_start):
    """ Returns the current table, i, and j position of the thread in the diagonal
    """

    i = i_start
    j = tid
    t = t_start

    # Finds the current layer by iteratively
    # This iterates for maximum 'len(Q)' times
    # NOTE: Use a binary search
    while j >= L[t]:
        j -= L[t]
        t += 1
        i -= 1

    return t, i, j


def threads(N, i_start, t_start, fn):
    for tid in range(N):
        t, i, j = get_thread_position(tid, i_start, t_start)
        fn(t, i, j)


def thread_function(t, i, j):
    """ Indipendent function executed in parallel over each element of a diagonal
    """

    u_cell = memory_table(t, i - 1, j)
    l_cell = memory_table(t - 1, i, j - P[C[t] + j])
    d_cell = memory_table(t - 1, i - 1, j - P[C[t] + j])

    # Check if the reference string and the query match
    q = Q[i]
    s = S[C[t] + j]
    matches = 1 if q == s else -1

    u_score = u_cell + SCORE_INDEL
    l_score = l_cell + SCORE_INDEL
    d_score = d_cell + matches * SCORE_MATCH

    score = max(max(u_score, l_score), d_score)
    memory_table_set(t, i, j, score)

    print(
        f"{t},{i},{j} -> u: {u_cell:>2}/{u_score:>2}, l: {l_cell:>2}/{l_score:>2}, d({q} {'==' if q == s else '!='} {s}): {d_cell:>2}/{d_score:>2} -> {score:>2}")


def argmax(xs):
    i = 0
    best = xs[0]
    for j, x in enumerate(xs):
        if x > best:
            best = x
            i = j
    return i

if __name__ == "__main__":

    diag_size = 0
    diag = 0

    # First set of anti-diagonals
    for i_start in range(0, len(Q)):
        t_start = 0

        diag_size += L[diag]
        diag += 1

        print(f"({t_start},{i_start}) with {diag_size} elements")
        threads(diag_size, i_start, t_start, thread_function)

    # Second set of anti-diagonals
    for t_start in range(1, len(L) - len(Q) + 1):
        i_start = len(Q) - 1

        # No change in diagonal size
        #diag_size -= L[diag - len(Q)]
        diag += 1

        print(f"({t_start},{i_start}) with {diag_size} elements")
        threads(diag_size, i_start, t_start, thread_function)

    # Third set of anti-diagonals
    for t_start in range(len(L) - len(Q) + 1, len(L)):
        i_start = len(Q) - 1

        diag_size -= L[diag - len(L)]
        diag += 1

        print(f"({t_start},{i_start}) with {diag_size} elements")
        threads(diag_size, i_start, t_start, thread_function)


    # Extract best aligments
    best = [ memory_table(len(L) - 1, len(Q) - 1, j) for j in range(L[-1])]
    print(f"best: {best}")

    # Reverse, find actual aligments
    cigarx = []
    for _ in range(L[-1]):
        cigarx.append(['N'] * len(L))

    frontier_index = 0
    frontier_old = [(len(L) - 1, len(Q) - 1, j, j, 0) for j,b in enumerate(best)]
    frontier_new = []
    while len(frontier_old) != 0 and frontier_index != len(L):

        # On element of the frontier for each thread
        for tidx, (t, i, j, output_id, element_id) in enumerate(frontier_old):
            
            cells_indices = [
                (t, i - 1, j),                  # 0, u
                (t - 1, i, j - P[C[t] + j]),    # 1, l
                (t - 1, i - 1, j - P[C[t] + j]) # 2, d
            ]

            cell_values = [
                memory_table(*cells_indices[0]),
                memory_table(*cells_indices[1]),
                memory_table(*cells_indices[2]),
            ]

            # Find best near cell
            cell_best_idx = argmax(cell_values)
            cell_best_indices = cells_indices[cell_best_idx]
            
            # Query and reference string values
            q = Q[i]
            s = S[C[t] + j]
            
            # Match or mismatch
            if cell_best_idx == 2:
                cigarx[output_id][len(L) - element_id - 1] = '=' if q == s else 'X'
            elif cell_best_idx == 1:
                cigarx[output_id][len(L) - element_id - 1] = 'I'
            else:
                cigarx[output_id][len(L) - element_id - 1] = 'D'

            # Add new cell position into frontier
            frontier_new.append((*cells_indices[cell_best_idx], output_id, element_id + 1))

            
        frontier_index += 1
        frontier_old = frontier_new
        frontier_new = []

        pass

print("best sequences:")
for i in range(L[-1]):
    print("".join(cigarx[i]))



