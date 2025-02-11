

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


#L = [ 3,  3,  5,  8]
#C = [ 3,  6, 11, 19]
#R = [ 0,  3,  6, 11]

#S = [ 
#    "1a1", "1b1", "1c1", 
#    "2a1", "2b1", "2c1", 
#    "3a1", "3a2", "3b1", "3c1", "3c2",
#    "4a1", "4a2", "4b1", "4b2", "4b3", "4c1", "4c2", "4c3"
#]

if False:
    S = [
        'C', 'B', 'A',
        'C', 'B', 'A',
        'C', 'B', 'A', 'A',
        'C', 'B', 'B', 'B', 'A', 'A', 'A', 'A',
        'C', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A'
    ]

    L = [3, 3,  4,  8,  9]
    Q = 'ABCDE'

    # Parent offset
    P = [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 2, 2, 3, 3, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 1
    ]
else:
    S = [ 
        'A', # layer: 0 
        'C', # layer: 1
        'C', # layer: 2
        'C'  # layer: 3
    ]
    P = [ 
        0, # layer: 0 
        0, # layer: 1 
        0, # layer: 2
        0, # layer: 3 
    ]
    Q = 'AACA'

    # Dimensione di ogni layer
    L = [ 1, 1, 1, 1 ]

# =========================================
# Easiest test case

SCORE_MATCH =  1
SCORE_INDEL = -1

def create_diag_sizes(L):
    result = []

    window_size = len(L)
    L_new = [0] * (window_size - 1) + L
    L_new = L_new + [0] * (window_size - 1) 

    for i in range(0, len(L_new) - window_size + 1):
        result.append(sum(L_new[i:i+window_size]))

    return result

# Prefix sum delle dimensioni di ogni layer
J = [ sum(L[0:i+1]) for i in range(len(L))]

# Dimensione di ogni diagonale, creata tramite sliding-window 
# (len = len(L)) sulle dimensioni dei layer
C = create_diag_sizes(L)

# Offset per accedere linearmente in memoria in base 
# all'indice dell'anti-diagonale
D = [ sum(C[0:i]) for i in range(len(C)) ]

# Dynamic programming table
M = [0] * (len(S) * len(Q))

print(f"Dimensione di ogni layer: {L}")
print(f"Prefix sum delle dimensioni di ogni layer: {J}")
print(f"Dimensione di ogni diagonale: {C}")
print(f"Offset per accedere linearmente in memoria: {D}")
print(f"memoria: {M}")

# Accedi alla memoria di una anti-diagonale con safeguards
def memory_get(mem, i, k, layer, diag):
    if k < 0 or i >= C[k] or i < 0:
        if k == -2: # Prima diagonale
            return 0
        # Margini
        return - (layer + 1) + (1 if diag else 0)
    return mem[D[k] + i]

# Setta la memoria di una anti-diagonale
def memory_set(mem, i, k, value):
    mem[D[k] + i] = value


def calc_curr_thread_layer(i, base, layer):

    # trova dimensione layer precedente
    prev_layer_size = 99
    curr_layer_size = J[0]
    curr_layer_index = 0 

    # Massimo len(Q) iterazioni
    while i + base >= curr_layer_size:
        curr_layer_index += 1
        curr_layer_size = C[curr_layer_index]
        prev_layer_size = L[curr_layer_index - 1]

    return prev_layer_size + P[i], layer - curr_layer_index

# Simula un gruppo di thread indipendenti
def threads(layer, size, base, fn):
    for i in range(size):
        offset, q = calc_curr_thread_layer(i, base, layer)
        fn(i, layer, offset, q)

# Processa un elemento dell'anti-diagonale se siamo nella prima
# parte della matrice, quindi gli indici sono piu semplici
def thread_fn_first(i, k, offset, q):

    u = memory_get(M, i, k - 1, k, False)
    l = memory_get(M, i - offset, k - 1, k, False)
    d = memory_get(M, i - offset, k - 2, k, True)

    u_score = u + SCORE_INDEL
    l_score = l + SCORE_INDEL

    matches = 1 if Q[q] == S[i] else -1
    d_score = d + matches * SCORE_MATCH

    score = max(max(u_score, l_score), d_score)
    memory_set(M, i, k, score)

    print(f"({i:>2},{k})-> u: {u:>2}/{u_score:>2}, l: {l:>2}/{l_score:>2}, d: {d:>2}/{d_score:>2} -> score: {score:>2}")


# Processa un elemento dell'anti-diagonale se siamo nella seconda
# parte della matrice, indici piu complessi relativi al bordo di destra
def thread_fn_second(i, k, offset, q):

    # Dove sono io partendo dalla fine?
    i_end = (C[k] - 1) - i

    # Dove sono relativo all'anti-diagonale precedente? Partendo dall'inizio
    a_beg = (C[k - 1] - 1) - i_end

    # Dove sono relativo all'anti-diagonale ancora prima? Partendo dall'inizio
    b_beg = (C[k - 2] - 1) - i_end

    u = memory_get(M, a_beg, k - 1, k, False)
    l = memory_get(M, a_beg - offset, k - 1, k, False)
    d = memory_get(M, b_beg - offset, k - 2, k, True)

    u_score = u + SCORE_INDEL
    l_score = l + SCORE_INDEL

    matches = 1 if Q[q - 1] == S[i] else -1
    d_score = d + matches * SCORE_MATCH

    score = max(max(u_score, l_score), d_score)
    memory_set(M, i, k, score)

    print(f"({i:>2},{k})-> u: {u:>2}/{u_score:>2}, l: {l:>2}/{l_score:>2}, d: {d:>2}/{d_score:>2} -> score: {score:>2}")


layer_count = len(L)

# Prima dell'anti-diagonale
for diag_index in range(layer_count):
    threads(
        diag_index,          # diagonale corrente  
        C[diag_index],       # dimensione della diagonale
        0,                   # dove inizia il confronto con la sequenza
        thread_fn_first      # funzione di ogni thread
    )

# Dopo l'anti-diagonale
for diag_index in range(layer_count, 2 * layer_count - 1):
    threads(
        diag_index, 
        C[diag_index], 
        J[layer_count - diag_index], 
        thread_fn_second
    )