
#define DEBUG

using u32 = unsigned int;
using i32 = int;
using u16 = unsigned short;
using i16 = short;
using u8  = unsigned char;
using i8  = char;
using f32 = float;

#define SCORE_MATCH +1.0
#define SCORE_INDEL -1.0

#define QUERY_LEN 3

// Dynamic programming table allocated on the host side
extern __shared__ i32 SHD[];
__device__ u32 g_target_size;

// Local block constants
// NOTE: Can they go to constant memory?
__shared__ u32 L[QUERY_LEN];
__shared__ u32 Q[QUERY_LEN];
__shared__ u32 D[QUERY_LEN];
__shared__ u32 T[QUERY_LEN]; 
__shared__ u32 C[QUERY_LEN]; 

// Where a thread is inside the DP matrix
struct location { i32 t, i, j; };

// Returns the position inside the DP table of a thread by its index 
// TODO: Use binary search instead of a linear search
__device__ location get_thread_position(
  i32 tid, i32 i_beg, i32 t_beg, 
  const u32 *__restrict__ L
) {

  i32 i = i_beg;
  i32 j = tid;
  i32 t = t_beg;

  // Finds the current layer by iteratively subtracting
  // This iterates for maximum 'QUERY_LEN' times
  while (j >= L[t]) {
    j -= L[t];
    t += 1;
    i -= 1;
  }

  return { t, i, j };
}

// Read memory of the DP Table with safeguards for margins
__device__ i32 dpt_mem_get(i32 t, i32 i, i32 j) {

  // Magin root
  if (t < 0 && i < 0) return 0;
  // Margin left
  if (t < 0) return -(i + 1);
  // Margin up
  if (i < 0) return -(t + 1);

  return SHD[(g_target_size * 2) + T[t] + i * L[t] + j];
}

// Read parent offset
__device__ i32 parent_offset(i32 t, i32 j) {
  return SHD[g_target_size + C[t] + j];
}

// Read sequence element
__device__ u32 sequence(i32 t, i32 j) {
  return SHD[C[t] + j];
}

// Write to memory of the DP Table
__device__ void dpt_mem_set(i32 t, i32 i, i32 j, i32 value) {
  SHD[g_target_size * 2 + T[t] + i * L[t] + j] = value;
}

// Updates a single cell in the DP matrix
__device__ void dpt_update(i32 t, i32 i, i32 j) {

  i32 u_cell = dpt_mem_get(t    , i - 1, j                      );
  i32 l_cell = dpt_mem_get(t - 1, i    , j - parent_offset(t, j));
  i32 d_cell = dpt_mem_get(t - 1, i - 1, j - parent_offset(t, j));

  u32 s = sequence(t, j);
  u32 q = Q[i];
  i16 matches = (s == q) ? 1 : -1;

  i32 u_score = u_cell + SCORE_INDEL;
  i32 l_score = l_cell + SCORE_INDEL;
  i32 d_score = d_cell + SCORE_MATCH * matches;

  i32 score = max(d_score, max(l_score, u_score));
  dpt_mem_set(t, i, j, score);

#ifdef DEBUG
  char sym = (matches == 1) ? '=' : '!';
  printf("(%d, %d, %d) -> u: %2d/%2d, l: %2d/%2d, d(%d %c %d): %2d/%2d -> %2d\n", 
    t, i, j, u_cell, u_score, l_cell, l_score, s, sym, q, d_cell, d_score, score);
#endif

}

// NOTE: For now we assume just ONE block
extern "C" 
__global__ void mine_global_aligment(
  const u32 *__restrict__ target,         // Packed target sequences
  const u32 *__restrict__ parents,        // Packed parents
  const u32 *__restrict__ levels,         // Size of each target level
  const u32 *__restrict__ levels_cumsum,  // Cumsum of target levels
  const u32 *__restrict__ tables,         // Precomputed DP table indices
  const u32 *__restrict__ query,          // Query sequence
  u32 target_size                         // Size of packed sequence
) {

  g_target_size = target_size;
  if (threadIdx.x < QUERY_LEN) {
    L[threadIdx.x] = levels[threadIdx.x];
    T[threadIdx.x] = tables[threadIdx.x];
    C[threadIdx.x] = levels_cumsum[threadIdx.x];
    Q[threadIdx.x] = query[threadIdx.x];
  }

  // Load packed data into shared memory
  if (threadIdx.x < target_size) {
    SHD[threadIdx.x] = target[threadIdx.x];
    SHD[g_target_size + threadIdx.x] = parents[threadIdx.x];
  }

  __syncthreads();
  
  u32 concurrent_threads = 0;
  u32 diag = 0; 

  // Process the first set of anti-diagonals,
  // the ones starting from the left side of the DP table
  i32 t_beg = 0;
  for (i32 i_beg = 0; i_beg < QUERY_LEN; i_beg += 1) {

    // Calculate current diagonal size, i.e. the amount of concurrent threads
    concurrent_threads += L[diag];
    diag += 1; 

    // We try to maximize the amount of concurrent threads
    // by traversing the merged DP table the optimal way
    if (threadIdx.x < concurrent_threads) {
      auto [t, i, j] = get_thread_position(threadIdx.x, i_beg, t_beg, L);
      dpt_update(t, i, j);
    }

    // NOTE: We could let each warp proceed in autonomy by using
    // a circular buffer of work that gets filled immediately after
    // its dependant values are ready. 
    //
    // Atomic operations inside shared memory are much faster that
    // in global memory.
    //  
    __syncthreads();

    if (threadIdx.x == 0) {
      printf("\n");
    }
  }

  // Process the second set of anti-diagonals
  // the ones starting from the bottom of the DP table
  i32 i_beg = QUERY_LEN - 1;
  for (i32 t_beg = 1; t_beg < QUERY_LEN; t_beg += 1) {

    // Reduce amount of concurrent threads
    concurrent_threads -= L[diag - QUERY_LEN];
    diag += 1; 

    if (threadIdx.x < concurrent_threads) {
      auto [t, i, j] = get_thread_position(threadIdx.x, i_beg, t_beg, L);
      dpt_update(t, i, j);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      printf("\n");
    }
  }
}
