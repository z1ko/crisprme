use crate::{Config, Gpu, KResult};
use std::time::{Duration, Instant};

/// Describe what to explore during autotuning
pub struct Search {
    pub configs: Vec<Config>,
}

impl Search {
    /// Generates all possible launch configurations
    pub fn configurations(&self) -> impl Iterator<Item = &Config> {
        self.configs.iter()
    }
}

/// Analyze kernel and return best launch configuration
pub fn benchmark<F>(gpu: &Gpu, search: Search, f: F) -> KResult<(Config, u128)>
where
    F: Fn(&Gpu, Config) -> Duration,
{
    Ok(search
        .configurations()
        .map(|c| (*c, f(gpu, *c).as_micros()))
        .min_by_key(|(_, t)| *t)
        .unwrap())
}
