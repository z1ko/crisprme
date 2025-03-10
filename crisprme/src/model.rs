
/*
#[derive(Debug)]
struct Node<const N: usize> {
    children: [Option<Box<Node<N>>>; N],
    counters: [usize; N],
}
*/

pub struct PackState {
    
    blocks: usize,
    blocks_width: usize,

    curr_block: usize,
    curr_offset: usize,
    curr_offset_base: usize,
    curr_offset_bias: usize,
}

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

    pub fn pack(&self, offset_base: usize, curr_layer: usize, packed: &mut PackedTree) -> usize {
        match self {
            // If we are at a Leaf there is nothing left to do here
            Node::Leaf { counter: _ } => { return 1; },
            // Pack childrens
            Node::Inner { children, subtrees } => {
                
                let mut offset_global = offset_base;
                let mut offset_bias = 0;

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
                        let offset = if curr_layer == 0 { 0 } else { offset_global };
                        // NOTE: WE MUST CHECK THAT THIS HAS ENOUGH BITS
                        packed.offset[curr_layer].push(offset);


                        let child_elements = child.pack(offset_global - offset_bias, curr_layer + 1, packed);
                        packed_total += child_elements;

                        offset_global += subtrees[child_idx];
                        offset_bias += 1;
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
    pub fn pack(&self) -> PackedTree {
        let mut packed_tree = PackedTree {
            layers: vec![vec![]; self.depth],
            offset: vec![vec![]; self.depth]
        };

        let packed_elements = self.root.pack(0, 0, &mut packed_tree);
        assert!(packed_elements == self.span());

        return packed_tree;
    }

    /// Returns a collection of PackedTree(s) over the entire Tree
    pub fn multipack(&self, max_width: usize) -> (Vec<PackedTree>, usize) { unimplemented!() }

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
pub struct PackedTree {
    pub layers: Vec<Vec<u8>>,
    pub offset: Vec<Vec<usize>>
}

impl PackedTree {
    pub fn print(&self, anchor_size: usize) {
        println!("PackedTree - Elements: ");
        for i in 0..anchor_size {
            let letters: Vec<char> = self.layers[i].iter().map(|x| *x as char).collect(); 
            println!("[{:>4}]: {:?}", i, letters)
        }
        println!("PackedTree - Offsets: ");
        for i in 0..anchor_size {
            println!("[{:>4}]: {:?}", i, self.offset[i])
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use crate::utils::{self};
    use super::*;

    /// Tests that the tree contains the correct sequences
    #[test] fn correct_sequences_content() {

        const REF_SIZE: usize = 100000;
        const ANCHOR_LEN: usize = 32;

        // Create random reference sequence
        let reference = utils::generate_test_sequence(REF_SIZE, b"ACTG");
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
}