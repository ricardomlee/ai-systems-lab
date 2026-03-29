//! Model loading and inference implementation

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType};
use candle_transformers::models::llama::{Llama, LlamaConfig};
use hf_hub::{Repo, RepoType, Api};
use tokenizers::Tokenizer;

use crate::kv_cache::KVCache;

/// LLM Model wrapper with KV cache support
pub struct Model {
    llama: Llama,
    tokenizer: Tokenizer,
    device: Device,
    kv_cache: Option<KVCache>,
    dtype: DType,
}

impl Model {
    /// Load model from Hugging Face Hub
    pub fn load(model_id: &str, use_fp16: bool) -> Result<Self> {
        let api = Api::new()
            .context("Failed to create Hugging Face API")?;

        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        // Download files
        println!("  Downloading config.json...");
        let config_path = repo.get("config.json")
            .context("Failed to download config.json")?;

        println!("  Downloading tokenizer.json...");
        let tokenizer_path = repo.get("tokenizer.json")
            .or_else(|_| repo.get("tokenizer_config.json"))
            .context("Failed to download tokenizer")?;

        println!("  Downloading model weights...");
        // Try safetensors first, then pytorch
        let weights_path = repo.get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .context("Failed to download model weights")?;

        // Load config
        let config_bytes = std::fs::read(config_path)?;
        let config: LlamaConfig = serde_json::from_slice(&config_bytes)
            .context("Failed to parse config")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .context("Failed to load tokenizer")?;

        // Set device and dtype
        let device = Device::new_cuda(0)
            .unwrap_or(Device::new_mps(0)
                .unwrap_or(Device::Cpu));

        let dtype = if use_fp16 { DType::F16 } else { DType::F32 };

        println!("  Loading weights into memory...");
        // Load model weights
        let weights = candle_nn::VarBuilder::from_pth(weights_path, dtype, &device)
            .context("Failed to load weights")?;

        let llama = Llama::load(&config, weights)
            .context("Failed to initialize Llama model")?;

        // Initialize KV cache
        let kv_cache = Some(KVCache::new(
            config.num_hidden_layers,
            2048,  // max_seq_len
            config.num_attention_heads,
            config.hidden_size / config.num_attention_heads,
            &device,
        ));

        println!("  Model loaded successfully!");
        println!("  Device: {:?}", device);
        println!("  Dtype: {:?}", dtype);

        Ok(Self {
            llama,
            tokenizer,
            device,
            kv_cache,
            dtype,
        })
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String> {
        // Encode prompt
        let mut tokens = self.tokenizer.encode(prompt, true)
            .context("Failed to encode prompt")?;
        let token_ids = tokens.get_ids().to_vec();

        let mut generated_tokens = Vec::with_capacity(max_tokens);
        let mut current_tokens = token_ids.clone();

        for i in 0..max_tokens {
            // Forward pass
            let input_len = current_tokens.len();
            let input_tensor = Tensor::new(&current_tokens[..], &self.device)?
                .unsqueeze(0)?;

            let logits = self.llama.forward(&input_tensor)?;

            // Get logits for next token
            let logits = logits.squeeze(0)?;
            let logits = logits.narrow(0, logits.dim(0)? - 1, 1)?;

            // Sample next token
            let next_token = if temperature == 0.0 {
                // Argmax sampling
                logits.argmax(1)?
            } else {
                // Temperature sampling
                let logits = &logits / temperature;
                let probs = candle_nn::ops::softmax(&logits, 1)?;
                probs.multinomial(1)?
            };

            let next_token_id = next_token.get::<u32>(0)? as u32;

            // Check for EOS
            if next_token_id == self.tokenizer.get_vocab(true).get("<eos>").copied().unwrap_or(2) {
                break;
            }

            generated_tokens.push(next_token_id);
            current_tokens = vec![next_token_id];
        }

        // Decode generated tokens
        let text = self.tokenizer.decode(&generated_tokens, true)
            .context("Failed to decode tokens")?;

        Ok(text)
    }

    /// Get model info
    pub fn info(&self) -> String {
        format!(
            "Model on {:?}, dtype: {:?}, KV cache: {}",
            self.device,
            self.dtype,
            if self.kv_cache.is_some() { "enabled" } else { "disabled" }
        )
    }
}
