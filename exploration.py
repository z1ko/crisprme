

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
#       CBA - CBA - CBAA - CBBBAAAA - CBBAAAAA
#
#   Q1  111   222   3333   44444444   55555555
#   Q2  1X1   2X2   3XX3   4XXXXXX4   5XXXXXX5
#   Q3  1X1   2X2   3XX3   4XXXXXX4   5XXXXXX5
#   Q4  111   222   3333   44444444   55555555
#
#    W  |||   |||   ||||   --------   --------
#    A  |||   |||   ||||   --------   --------
#    R  |||   |||   ||||   --------   --------
#    P  |||   |||   ||||   --------   --------
#
#       123   123   1234   11111111   11111111
#       123   123   1234   22222222   22222222
#       123   123   1234   33333333   33333333
#       123   123   1234   44444444   44444444
#
#     Quando costruisco i batch e' opportuno unire abbastanza sample semplici (Span piccolo)
#     in modo da permettere ai warp di lavorare sempre in orizzontale, come nella quarta e
#     quinta tabella dell'esempio.
#
#
# [4] Nel backtracking quando passo da un livello l al livello l-1 devo mappare la cella virtuale 
#     alla cella reale usando il PARENT_OFFSET.
#
#       --A --A --AA ----AAAA ---AAAAA
#       --- --0 --00 ----0011 ---01233
#                           |       ||
#                           |_______/|
#                           \________/
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

Q = 32
M = [ 3, 3, 4, 8, 9 ]

def chunk_shared_memory(Q, M):
    return Q * sum(M)

print("memory: ", chunk_shared_memory(Q, M))

