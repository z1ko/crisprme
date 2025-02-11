
#define DEBUG

using u32 = unsigned int;
using i32 = int;
using u16 = unsigned short;
using u8  = unsigned char;
using i8  = char;
using f32 = float;

#define SCORE_MATCH +1.0
#define SCORE_INDEL -1.0

#define QUERY_LEN 3

// Dynamic programming table allocated on the host side
extern __shared__ u32 DP[];

// Local block constants
// NOTE: Can they go to constant memory?
__shared__ u32 L[QUERY_LEN];
//__shared__ u32 C[QUERY_LEN];
__shared__ u32 D[QUERY_LEN];
//__shared__ u32 T[QUERY_LEN]; 

// Where a thread is inside the DP matrix
struct location { u32 t, i, j; };

// Returns the position inside the DP table of a thread by its index 
// TODO: Use binary search instead of a linear search
__device__ location get_thread_position(
  u32 tid, u32 i_beg, u32 t_beg, 
  const u32 *__restrict__ L
) {

  u32 i = i_beg;
  u32 j = tid;
  u32 t = t_beg;

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
__device__ u32 dpt_mem_get(i32 t, i32 i, i32 j) {

  // Magin root
  if (t < 0 && i < 0) return 0;
  // Margin left
  if (t < 0) return -(i + 1);
  // Margin up
  if (i < 0) return -(t + 1);

  // UNIMPLEMENTED
  return 0;
}

// Write to memory of the DP Table
__device__ u32 dpt_mem_set(u32 t, u32 i, u32 j, u32 value) {
  // UNIMPLEMENTED
}

// Updates a single cell in the DP matrix
__device__ void dpt_update(u32 t, u32 i, u32 j) {

  u32 u_cell = dpt_mem_get(t    , i - 1, j);
  u32 l_cell = dpt_mem_get(t - 1, i    , j);
  u32 d_cell = dpt_mem_get(t - 1, i - 1, j);

  // TODO 
  u16 matches = 0;

  u32 u_score = u_cell + SCORE_INDEL;
  u32 l_score = l_cell + SCORE_INDEL;
  u32 d_score = d_cell + SCORE_MATCH * matches;

  u32 score = max(d_score, max(l_score, u_score));
  dpt_mem_set(t, i, j, score);

#ifdef DEBUG
  printf("(%d, %d, %d) -> u: %2d, l: %2d, d: %2d\n", 
    t, i, j, u_cell, l_cell, d_cell);
#endif

}

// NOTE: For now we assume just ONE block
extern "C" 
__global__ void mine_global_aligment(
  //const u32 *__restrict__ target,         // Packed target sequences
  //const u32 *__restrict__ query,          // Query sequence
  const u32 *__restrict__ levels,         // Size of each target level
  //const u32 *__restrict__ levels_cumsum,  // Cumsum of target levels
  const u32 *__restrict__ diagonals      // Size of each thread diagonal
  //const u32 *__restrict__ tables          // Precomputed DP table indices
) {

  if (threadIdx.x < QUERY_LEN) {
    L[threadIdx.x] = levels[threadIdx.x];
    //C[threadIdx.x] = levels_cumsum[threadIdx.x];
    //T[threadIdx.x] = tables[threadIdx.x];
  }

  if (threadIdx.x < QUERY_LEN + 2) {
    D[threadIdx.x] = diagonals[threadIdx.x];
  }

  __syncthreads();
  
  u32 diag = 0; 

  // Process the first set of anti-diagonals,
  // the ones starting from the left side of the DP table
  u32 t_beg = 0;
  for (u32 i_beg = 0; i_beg < QUERY_LEN; i_beg += 1) {
    auto concurrent_threads = D[diag];
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
  u32 i_beg = QUERY_LEN - 1;
  for (u32 t_beg = 1; t_beg < QUERY_LEN; t_beg += 1) {
    auto concurrent_threads = D[diag];
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
