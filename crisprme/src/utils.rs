use rand::{Rng, SeedableRng};

/// A generic sequence
pub type Seq = Vec<u8>;

/// Create a random reference sequence
pub fn generate_test_sequence(length: usize, nucleotides: &[u8], seed: u64) -> Seq {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let take_one = || nucleotides[rng.random_range(0..nucleotides.len())]; // Uniform random sequences
    std::iter::repeat_with(take_one)
        .take(length).collect()
}

/// Create a colletions of window slices of sequence
pub fn split_windows(sequence: &[u8], window_size: usize) -> Vec<&[u8]> {
    let windows = sequence.windows(window_size);
    windows.collect()
}

/// Linearize a collection of arrays
pub fn linearize_memory<T: Clone>(arrays: &[Vec<T>]) -> Vec<T> {
    arrays.concat()
}