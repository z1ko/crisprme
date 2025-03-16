
#define DEBUG 0

using u32 = unsigned int;
using i32 = int;
using u16 = unsigned short;
using i16 = short;
using u8  = unsigned char;
using i8  = char;
using f32 = float;

#define SCORE_MATCH +1.0
#define SCORE_INDEL -2.0

#define QUERY_LEN 20

// Supported operations
enum oper_e : u16 {
  MATCH    = 0,
  MISMATCH = 1,
  DELETE   = 2,
  INSERT   = 3,
};

// Packed cell data of size 32 bits
struct cell_t {
  i16 score;
  oper_e op;
};

union dpt_t {
  cell_t cell;
  i32 offset;
  u32 seq;
};

// Dynamic programming table allocated on the host side
extern __shared__ dpt_t SHD[];
__device__ u32 g_target_size;

// Local block constants
// NOTE: Can they go to constant memory?
__shared__ u32 L[QUERY_LEN];
__shared__ u32 Q[QUERY_LEN];
__shared__ u32 D[QUERY_LEN];
__shared__ u32 T[QUERY_LEN]; 
__shared__ u32 C[QUERY_LEN]; 

// Where a thread is inside the DP matrix
struct backtrace_t {
  i32 t, i, j;
};

// Returns the position inside the DP table of a thread by its index 
// TODO: Use binary search instead of a linear search
__device__ backtrace_t get_thread_position(
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

// Pack a score and a backtrace arrow
__forceinline__ __device__ cell_t pack_cell(i16 score, oper_e op) {
  return cell_t { score, op };
}

// Read memory of the DP Table with safeguards for margins
__device__ cell_t dpt_mem_get(i32 t, i32 i, i32 j) {

  // Magin root
  if (t < 0 && i < 0) {
    return pack_cell(0, oper_e::MATCH);
  }

  // Margin left
  if (t < 0) {
    return pack_cell((i + 1) * SCORE_INDEL, oper_e::INSERT);
  }

  // Margin up
  if (i < 0) {
    return pack_cell((t + 1) * SCORE_INDEL, oper_e::DELETE);
  }

  return SHD[(g_target_size * 2) + T[t] + i * L[t] + j].cell;
}

// Read parent offset
__device__ i32 parent_offset(i32 t, i32 j) {
  return SHD[g_target_size + C[t] + j].offset;
}

// Read sequence element
__device__ u32 sequence(i32 t, i32 j) {
  return SHD[C[t] + j].seq;
}

// Write to memory of the DP Table
__device__ void dpt_mem_set(i32 t, i32 i, i32 j, cell_t value) {
  SHD[g_target_size * 2 + T[t] + i * L[t] + j].cell = value;
}

// Updates a single cell in the DP matrix
__device__ void dpt_update(i32 t, i32 i, i32 j) {

  // Possible parents
  const backtrace_t options[] = {
    {t - 1, i - 1, j - parent_offset(t, j)}, // diag
    {t - 1, i    , j - parent_offset(t, j)}, // left
    {t    , i - 1, j                      }, // up
  };

  u32 s = sequence(t, j);
  u32 q = Q[i];

  // Score for each operation
  const i16 scores[] = {
    SCORE_MATCH * (s == q) ? 1 : -1, // diag
    SCORE_INDEL, // left
    SCORE_INDEL, // up
  };

  // First option
  i16 op_score_best = -100; // Good enough value
  u8 op_idx_best = 0;

  // Find best backtrace
  for (u8 op_idx = 0; op_idx < 3; op_idx += 1) {
    const auto& opt = options[op_idx];

    // Get other cell, extract score and backtrace
    cell_t op_cell = dpt_mem_get(opt.t, opt.i, opt.j);
    i16 op_score = op_cell.score + scores[op_idx];

    // TODO: Resolve path divergence between threads
    if (op_score > op_score_best) {
      op_score_best = op_score;
      op_idx_best = op_idx;
    }
  }
  
  // Select corret operation
  // TODO: unify with prev. operations 
  oper_e op_type;
  switch (op_idx_best) {
    case 0: // diag
      op_type = (s == q) ? oper_e::MATCH : oper_e::MISMATCH;
      break;
    case 1: // left
      op_type = oper_e::DELETE;
      break;
    case 2: // up
      op_type = oper_e::INSERT;
      break;
  }

  // Store best score and operation
  cell_t result = pack_cell(op_score_best, op_type);
  dpt_mem_set(t, i, j, result);

#if DEBUG == 1

  char* op_str;
  switch (result.op) {
    case oper_e::MATCH:    op_str = "match";    break;
    case oper_e::MISMATCH: op_str = "mismatch"; break;
    case oper_e::DELETE:   op_str = "delete";   break;
    case oper_e::INSERT:   op_str = "insert";   break;
  }

  printf("(%d, %d, %d) -> (score: %2d, operation: %s)\n", 
    t, i, j, op_score_best, op_str);
#endif
}

// Get the best previous cell from the current one
__forceinline__ __device__ backtrace_t dpt_backtrace(i32 t, i32 i, i32 j, u8 *__restrict__ result) {

  // What are the backtrace indices? 
  const backtrace_t options[] = {
    {t    , i - 1, j                      }, // up
    {t - 1, i    , j - parent_offset(t, j)}, // left
    {t - 1, i - 1, j - parent_offset(t, j)}, // diag
  };

  // Get stored best backtrace
  cell_t cell = dpt_mem_get(t, i, j);

  // Backtrack
  u8 idx;
  switch (cell.op) {
    case oper_e::MATCH: // diag
      *result = '=';
      idx = 2;
      break;
    case oper_e::MISMATCH: // diag
      *result = 'X';
      idx = 2;
      break; 
    case oper_e::DELETE: // left
      *result = 'D';
      idx = 1;
      break;
    case oper_e::INSERT: // up
      *result = 'I';
      idx = 0;
      break;
  }

  return options[idx];
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
  u32* cigarx,                            // Result aligment CIGARXs
  i32* scores,                            // Result aligment scores
  u32 target_size                         // Size of packed sequence
) {

  // For now it is like this
  assert(sizeof(dpt_t) == sizeof(u32));

  g_target_size = target_size;
  if (threadIdx.x < QUERY_LEN) {
    L[threadIdx.x] = levels[threadIdx.x];
    T[threadIdx.x] = tables[threadIdx.x];
    C[threadIdx.x] = levels_cumsum[threadIdx.x];
    Q[threadIdx.x] = query[threadIdx.x];
  }

  // Load packed data into shared memory
  if (threadIdx.x < target_size) {
    SHD[threadIdx.x].seq = target[threadIdx.x];
    SHD[g_target_size + threadIdx.x].offset = parents[threadIdx.x];
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

#if DEBUG == 1
    if (threadIdx.x == 0) {
      printf("\n");
    }
#endif
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

#if DEBUG == 1
    if (threadIdx.x == 0) {
      printf("\n");
    }
#endif
  }

  // A thread for each sequence
  const auto seq_count = levels[QUERY_LEN - 1];
  if (threadIdx.x < seq_count) {

    backtrace_t state = { QUERY_LEN - 1, QUERY_LEN - 1, threadIdx.x };
    u8 cigar_seq_elem;

    // Store final aligment score
    scores[threadIdx.x] = dpt_mem_get(state.t, state.i, state.j).score;

#if DEBUG == 1
    printf("thread %d is now in backtracking, starting from (t: %d, i: %d, j: %d)\n", 
      threadIdx.x, state.t, state.i, state.j);
#endif

    // From the last layer go back and generate aligments.
    // The maximum aligment string lenght is QUERY_LEN * 2 - 1,
    // from last level to first without diagonal movements.

    bool complete = false;
    for (u8 seq_step = 0; seq_step < QUERY_LEN * 2; seq_step += 1) {
      if (!complete) {

        // Get next state
        state = dpt_backtrace(state.t, state.i, state.j, &cigar_seq_elem);

#if DEBUG == 1
        printf("new state for thread %d with output %c -> (t: %d, i: %d, j: %d)\n", 
          threadIdx.x, cigar_seq_elem, state.t, state.i, state.j);
#endif

        if (state.t == -1 && state.i == -1) {
#if DEBUG == 1
          printf("thread %d has reached end of its sequence\n", threadIdx.x);
#endif
          complete = true;
        }

        // Store directly to global memory
        cigarx[seq_step * seq_count + threadIdx.x] = cigar_seq_elem;
      }
    }
  }

  __syncthreads();
}
