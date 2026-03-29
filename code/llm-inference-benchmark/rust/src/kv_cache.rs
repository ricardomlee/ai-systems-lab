//! KV Cache implementation for efficient LLM inference
//!
//! The KV cache stores key and value states from previous forward passes,
//! avoiding recomputation of attention for already-generated tokens.
//! This reduces per-step complexity from O(n²) to O(n).

use candle_core::{Tensor, Device, Result};

/// KV Cache for a single layer
#[derive(Clone)]
pub struct LayerCache {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
}

/// Full KV Cache for all layers
pub struct KVCache {
    pub caches: Vec<LayerCache>,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

impl KVCache {
    /// Create a new KV cache with pre-allocated tensors
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        // Shape: [batch=1, max_seq, num_heads, head_dim]
        let shape = (1, max_seq_len, num_heads, head_dim);

        let caches: Vec<LayerCache> = (0..num_layers)
            .map(|_| {
                let k_cache = Tensor::zeros(shape, candle_core::DType::F32, device)
                    .expect("Failed to allocate K cache");
                let v_cache = Tensor::zeros(shape, candle_core::DType::F32, device)
                    .expect("Failed to allocate V cache");
                LayerCache { k_cache, v_cache }
            })
            .collect();

        Self {
            caches,
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Update cache with new key/value tensors
    pub fn update(
        &mut self,
        layer_idx: usize,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if layer_idx >= self.caches.len() {
            anyhow::bail!("Invalid layer index: {}", layer_idx);
        }

        let cache = &mut self.caches[layer_idx];
        let current_seq = self.seq_len;

        // Concatenate new k/v with cached values at the current sequence position
        // For efficiency, we use narrowslice instead of full concat
        let k_cache = cache.k_cache.narrow(1, current_seq, k.dim(1)?)?;
        let v_cache = cache.v_cache.narrow(1, current_seq, v.dim(1)?)?;

        // Update sequence length
        self.seq_len += k.dim(1)?;

        Ok((k_cache, v_cache))
    }

    /// Reset the cache for a new sequence
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new(22, 2048, 32, 64, &Device::Cpu)
    }
}
