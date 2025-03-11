use std::collections::HashSet;

#[derive(Debug)]
pub enum Node<const N: usize> {
    Inner {
        children: [Option<Box<Node<N>>>; N],
        subtrees: [usize; N],
    },
    Leaf {
        counter: usize,
    }
}

impl Node<4> {
    pub fn empty_inner() -> Self {
        Self::Inner {
            children: [const { Option::None }; 4],
            subtrees: [const {0}; 4],
        }
    }

    pub fn empty_leaf() -> Self {
        Self::Leaf { counter: 0 }
    }

    pub fn insert(&mut self, seq: &[u8]) -> bool {
        match self {
            Node::Leaf { counter } => {
                *counter += 1; // Increase counter for this sequence
                return false;
            },
            // Continue insertion
            Node::Inner { children, subtrees } => {

                // Get index by letter
                let curr_letter = seq[0];
                let child_idx = match curr_letter {
                    b'A' => 0,
                    b'C' => 1,
                    b'T' => 2,
                    b'G' => 3,
                    _ => unimplemented!()
                };

                let mut this_branch = false;
                if let None = children[child_idx] {
                    this_branch = true;

                    if seq.len() == 1 {
                        // Is this the last element? If so add leaf node
                        children[child_idx] = Some(Box::new(Node::Leaf { counter: 0 }));
                    } else {
                        // Normal node
                        children[child_idx] = Some(Box::new(Node::empty_inner()));
                    }
                }

                // Continue insertion
                let child_branch = children[child_idx].as_mut().unwrap()
                    .insert(&seq[1..]); 

                // If we generated some new branch during insertion increase number of subsequences 
                let branch = this_branch || child_branch;
                if branch {
                    // Update subtree size
                    subtrees[child_idx] += 1;
                }

                return branch;
            }
        }
    }

    pub fn pack(&self, offset: usize, curr_layer: usize, packed: &mut PackedTree<u8, usize>) -> usize {
        match self {
            // If we are at a Leaf there is nothing left to do here
            Node::Leaf { counter: _ } => { return 1; },
            // Pack childrens
            Node::Inner { children, subtrees: _ } => {
                
                // Add children to packed data
                let mut packed_total = 0;
                for child_idx in 0..4 {
                    if let Some(child) = &children[child_idx] {

                        // Get letter from id
                        let letter = match child_idx {
                            0 => b'A',
                            1 => b'C',
                            2 => b'T',
                            3 => b'G',
                            _ => unimplemented!()
                        };

                        packed.layers[curr_layer].push(letter);

                        // First layer has no parent
                        let offset_this = if curr_layer == 0 { 0 } else { offset };
                        // NOTE: WE MUST CHECK THAT THIS HAS ENOUGH BITS
                        packed.offset[curr_layer].push(offset_this);
                        
                        let offset_layer = packed.offset[curr_layer].len() - 1;
                        let child_elements = child.pack(offset_layer, curr_layer + 1, packed);
                        packed_total += child_elements;
                    }
                }
                return packed_total;
            }
        }
    }

    pub fn print_sequences(&self, offset: usize, seq_stack: &mut Vec<u8>) -> usize {
        match self {
            Node::Leaf { counter } => {
                let seq_string = std::str::from_utf8(&seq_stack).unwrap();
                println!("[{:>4}] {}: {}", offset, seq_string, counter);
                return 1;
            },
            Node::Inner { children, subtrees: _ } => {
                let mut curr_offset = offset;

                for child_idx in 0..4 {
                    if let Some(child) = &children[child_idx] {

                        // Get letter from id
                        let letter = match child_idx {
                            0 => b'A',
                            1 => b'C',
                            2 => b'T',
                            3 => b'G',
                            _ => unimplemented!()
                        };

                        seq_stack.push(letter);
                        curr_offset += child.print_sequences(curr_offset, seq_stack);
                        seq_stack.pop();
                    }
                }

                return curr_offset - offset;
            },
        }
    }

    pub fn sequences(&self, seq_stack: &mut Vec<u8>, result: &mut Vec<Vec<u8>>) {
        match self {
            Node::Leaf { counter: _ } => {
                result.push(seq_stack.clone());
            },
            Node::Inner { children, subtrees: _ } => {
                for child_idx in 0..4 {
                    if let Some(child) = &children[child_idx] {

                        // Get letter from id
                        let letter = match child_idx {
                            0 => b'A',
                            1 => b'C',
                            2 => b'T',
                            3 => b'G',
                            _ => unimplemented!()
                        };

                        seq_stack.push(letter);
                        child.sequences(seq_stack, result);
                        seq_stack.pop();
                    }
                }
            },
        }
    }
}

#[derive(Debug)]
pub struct Tree<const N: usize> {
    root: Node<N>,
    depth: usize
}

impl Tree<4> {
    pub fn new(depth: usize) -> Self {
        Self {
            root: Node::empty_inner(),
            depth
        }
    }

    pub fn insert(&mut self, seq: &[u8]) {
        self.root.insert(seq);
    }

    /// Returns a complete PackedTree ready to be processed
    pub fn pack(&self) -> GlobalPackedTree {
        let mut packed_tree = GlobalPackedTree(
            PackedTree {
                layers: vec![vec![]; self.depth],
                offset: vec![vec![]; self.depth],
                depth: self.depth
            }
        );

        let packed_elements = self.root.pack(0, 0, &mut packed_tree);
        assert!(packed_elements == self.span());

        return packed_tree;
    }

    /// Print how many annotations for each unique sequence
    pub fn print_sequences(&self) {
        println!("Tree - Sequences:");
        let mut seq_stack: Vec<u8> = Vec::with_capacity(self.depth);
        self.root.print_sequences(0, &mut seq_stack);
    }

    /// Generate vectors of sequences
    pub fn sequences(&self) -> Vec<Vec<u8>> {
        let mut seq_stack: Vec<u8> = Vec::with_capacity(self.depth);
        let mut result = Vec::new();
        self.root.sequences(&mut seq_stack, &mut result);
        result
    }

    /// How many unique sequences are stored in the tree
    pub fn span(&self) -> usize {
        if let Node::Inner { children: _, subtrees } = &self.root /* Always true */ {
            let mut sum = 0;
            for counter in subtrees {
                sum += counter;
            }
            return sum;
        }
        panic!();
    }

}

/// A Prefix tree implemented with linear memory and offsets
#[derive(Debug)]
pub struct PackedTree<E, O> {
    pub layers: Vec<Vec<E>>,
    pub offset: Vec<Vec<O>>,
    pub depth: usize,
}

impl<E, O> PackedTree<E, O>
{
    /// How many unique sequences are stored in the packed tree
    pub fn span(&self) -> usize {
        return self.layers[self.depth -1].len();
    }
}

impl PackedTree<u8, u32> {

    pub fn layer_sizes(&self) -> Vec<u32> {
        self.layers.iter()
            .map(|layer| layer.len().try_into().unwrap())
            .collect()
    }

    pub fn layer_sizes_cumsum(&self) -> Vec<u32> {
        let mut prefixsum: Vec<u32> = self.layer_sizes().iter()
            .scan(0, 
                |acc, x| {
                    *acc += x;
                    Some(*acc)
                }
            ).collect();

        prefixsum.insert(0, 0);
        prefixsum.pop();

        prefixsum
    }

    pub fn dp_table_offsets(&self) -> Vec<u32> {
        self.layer_sizes_cumsum().iter()
            .map(|s| *s * u32::try_from(self.depth).unwrap())
            .collect()
    }
}

// PackedTree that uses global offsets
#[derive(Debug, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub struct GlobalPackedTree(pub PackedTree<u8, usize>);

// PackedTree that uses local relative offsets
#[derive(Debug, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub struct LocalPackedTree<O>(pub PackedTree<u8, O>);

impl GlobalPackedTree {
    pub fn print(&self) {
        println!("global_packed_tree::elements:");
        for i in 0..self.depth {
            let letters: Vec<char> = self.layers[i].iter().map(|x| *x as char).collect(); 
            println!("[{:>4}]: {:?}", i, letters)
        }
        println!("global_packed_tree::offsets:");
        for i in 0..self.depth {
            println!("[{:>4}]: {:?}", i, self.offset[i])
        }
    }

    /// Returns an HashSet with all complete sequences present in the tree
    pub fn sequences(&self) -> HashSet<String> {
        let mut seq_set = HashSet::new();
        for s in 0..self.span() {

            let mut seq = vec![b'X'; self.depth];
            let mut offset = s;

            for l in 0..self.depth {
                seq[self.depth - 1 - l] = self.layers[self.depth - 1 - l][offset];
                offset = self.offset[self.depth - 1 - l][offset];
            }

            let seq_string = std::str::from_utf8(&seq).unwrap();
            seq_set.insert(String::from(seq_string));
        }
        return seq_set;
    }

    /// Returns a collection of sub PackedTree(s) over the entire PackedTree
    pub fn split_at_width<O: TryFrom<usize> + Clone>(&self, width: usize) -> Vec<LocalPackedTree<O>> 
    where
        <O as TryFrom<usize>>::Error: std::fmt::Debug
    { 

        // The algorithm is simple, we start from the last layer, then we calculate
        // the initial index of the parent layer and the length of the run.
        // Continue to iterate until we reach the first layer.    
        
        let mut results: Vec<LocalPackedTree<O>> = Vec::new();

        let mut beg = 0;
        let sequence_count = self.span();
        while beg < sequence_count {

            // Packed tree of this block
            let mut packed_tree = LocalPackedTree(
                PackedTree {
                    layers: vec![vec![]; self.depth],
                    offset: vec![vec![]; self.depth],
                    depth: self.depth
                }
            );

            let mut rlen = width;
            let mut rmin = beg;

            // Process each layer in sequence
            for layer in 0..self.depth {

                // Clip the last block if we are out of bounds
                rlen = rlen.min(sequence_count - rmin);

                // Inserts layer data
                packed_tree.layers[self.depth - 1 - layer].extend_from_slice(
                    &self.layers[self.depth - 1 - layer][rmin..rmin+rlen]);
                
                // Make offset local
                // TODO: Make this parallel
                
                let global_offsets = &self.offset[self.depth - 1 - layer][rmin..rmin+rlen];
                let mut local_offsets = Vec::from(global_offsets);

                rmin = global_offsets[0];
                let new_rlen = global_offsets[rlen - 1] - rmin + 1;

                for e in 0..rlen {
                    local_offsets[e] -= rmin;
                }

                let mut shift = 0;
                for e in 1..rlen {

                    let curr = global_offsets[e];
                    let prev = global_offsets[e - 1];
                    if curr == prev { 
                        shift += 1;
                    }

                    local_offsets[e] = shift;
                }

                // Convert to the specified offset memory type
                // Update local offsets inside the PackedTree structure
                packed_tree.offset[self.depth - 1 - layer] = local_offsets.iter()
                    .map(|offset| (*offset).try_into().unwrap())
                    .collect();

                // Calculate new rmin, rmax and rlen
                rlen = new_rlen;
            }

            results.push(packed_tree);
            beg += width;
        }

        return results;
    
    }
}

impl LocalPackedTree<u32> {
    pub fn print(&self) {
        println!("local_packed_tree::elements:");
        for i in 0..self.depth {
            let letters: Vec<char> = self.layers[i].iter().map(|x| *x as char).collect(); 
            println!("[{:>4}]: {:?}", i, letters)
        }
        println!("local_packed_tree::offsets:");
        for i in 0..self.depth {
            println!("[{:>4}]: {:?}", i, self.offset[i])
        }
    }

    /// Returns an HashSet with all complete sequences present in the tree
    pub fn sequences(&self) -> HashSet<String> {
        let mut seq_set = HashSet::new();
        for s in 0..self.span() {
            
            let mut seq = vec![b'X'; self.depth];
            let mut i = s as u32;

            seq[self.depth - 1] = self.layers[self.depth - 1][i as usize];
            let mut offset = self.offset[self.depth - 1][i as usize];

            for l in 1..self.depth {
                i -= offset;

                seq[self.depth - 1 - l] = self.layers[self.depth - 1 - l][i as usize];
                offset = self.offset[self.depth - 1 - l][i as usize];
            }

            let seq_string = std::str::from_utf8(&seq).unwrap();
            seq_set.insert(String::from(seq_string));
        }
        return seq_set;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use crate::utils::{self};
    use super::*;

    const REF_SIZE: usize = 300000;
    const ANCHOR_LEN: usize = 32;
    const RNG_SEED: u64 = 3666;

    /// Tests that the tree contains the correct sequences
    #[test] fn tree_content_correct() {

        // Create random reference sequence
        let reference = utils::generate_test_sequence(REF_SIZE, b"ACTG", RNG_SEED);
        let reference_windows = utils::split_windows(&reference, ANCHOR_LEN);

        // Insert windows into reference set
        let mut real_seq_set: HashSet<String> = HashSet::new();
        for seq in &reference_windows {
            let seq_string = std::str::from_utf8(&seq).unwrap();
            real_seq_set.insert(String::from(seq_string));
        }

        // Create prefix tree
        let mut tree: Tree<4> = Tree::new(ANCHOR_LEN);
        for seq in &reference_windows {
            tree.insert(seq);
        }

        // Extract sequences from tree
        let mut tree_seq_set: HashSet<String> = HashSet::new();
        for seq in &tree.sequences() {
            let seq_string = std::str::from_utf8(&seq).unwrap();
            tree_seq_set.insert(String::from(seq_string));
        }

        // Check that they contain the same elements
        assert!(tree_seq_set.eq(&real_seq_set));

    }

    /// Tests that the global packed tree contains the correct sequences
    #[test] fn packed_tree_content_correct() {

        // Create random reference sequence
        let reference = utils::generate_test_sequence(REF_SIZE, b"ACTG", RNG_SEED);
        let reference_windows = utils::split_windows(&reference, ANCHOR_LEN);

        // Insert windows into reference set
        let mut real_seq_set: HashSet<String> = HashSet::new();
        for seq in &reference_windows {
            let seq_string = std::str::from_utf8(&seq).unwrap();
            real_seq_set.insert(String::from(seq_string));
        }

        // Create prefix tree
        let mut tree: Tree<4> = Tree::new(ANCHOR_LEN);
        for seq in &reference_windows {
            tree.insert(seq);
        }

        // Create packed prefix tree
        let packed_tree = tree.pack();

        // How many strings?
        let sequence_count = packed_tree.layers[ANCHOR_LEN - 1].len();
        assert!(sequence_count == real_seq_set.len());

        // Check that they contain the same elements
        assert!(real_seq_set.eq(&packed_tree.sequences()));
    }

    /// Tests that a packed tree can be split
    #[test] fn packed_tree_split() {

        // Create random reference sequence
        let reference = utils::generate_test_sequence(REF_SIZE, b"ACTG", RNG_SEED);
        let reference_windows = utils::split_windows(&reference, ANCHOR_LEN);

        // Insert windows into reference set
        let mut real_seq_set: HashSet<String> = HashSet::new();
        for seq in &reference_windows {
            let seq_string = std::str::from_utf8(&seq).unwrap();
            real_seq_set.insert(String::from(seq_string));
        }

        // Create prefix tree
        let mut tree: Tree<4> = Tree::new(ANCHOR_LEN);
        for seq in &reference_windows {
            tree.insert(seq);
        }

        // Create packed prefix tree
        let packed_tree = tree.pack();

        // Split packed tree into multiple local trees and aggregate all sequences
        let split_packed_trees = packed_tree.split_at_width(64);
        let mut split_seq_set: HashSet<String> = HashSet::new();
        for split_tree in &split_packed_trees {
            let split_tree_seqs = split_tree.sequences();
            split_seq_set.extend(split_tree_seqs);
        }

        // Check that they contain the same elements
        assert!(real_seq_set.eq(&split_seq_set));
    }
}