//! LLM Inference in Rust - High-performance text generation
//!
//! This crate provides a fast LLM inference engine using Hugging Face's Candle library.
//! Expected speedup: 5-10x over Python baseline, up to 15x with optimizations.

mod model;
mod kv_cache;

use anyhow::Result;
use clap::Parser;
use std::time::Instant;

use model::Model;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model ID on Hugging Face Hub
    #[arg(short, long, default_value = "TinyLlama/TinyLlama-1.1B-Chat-v1.0")]
    model_id: String,

    /// Prompt for text generation
    #[arg(short, long, default_value = "Explain what makes code fast")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(short, long, default_value_t = 100)]
    max_tokens: usize,

    /// Temperature for sampling (0.0 = argmax)
    #[arg(short, long, default_value_t = 1.0)]
    temperature: f64,

    /// Use FP16 precision
    #[arg(long, default_value_t = false)]
    fp16: bool,

    /// Disable KV cache
    #[arg(long, default_value_t = false)]
    no_kv_cache: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("============================================================");
    println!("LLM Inference Benchmark - Rust");
    println!("============================================================");
    println!("Model: {}", args.model_id);
    println!("Prompt: {}", args.prompt);
    println!("Max tokens: {}", args.max_tokens);
    println!("Temperature: {}", args.temperature);
    println!("FP16: {}", args.fp16);
    println!("============================================================");

    // Load model
    println!("Loading model...");
    let mut model = Model::load(&args.model_id, args.fp16)?;

    // Warmup
    println!("Warming up...");
    let _ = model.generate(&args.prompt, 10, args.temperature)?;

    // Benchmark
    println!("Running inference...");
    let start = Instant::now();
    let output = model.generate(&args.prompt, args.max_tokens, args.temperature)?;
    let elapsed = start.elapsed();

    let tokens_per_sec = args.max_tokens as f64 / elapsed.as_secs_f64();
    let ms_per_token = (elapsed.as_secs_f64() * 1000.0) / args.max_tokens as f64;

    println!("\n============================================================");
    println!("Generated output:");
    println!("============================================================");
    println!("{}", output);
    println!("\n============================================================");
    println!("RESULTS");
    println!("============================================================");
    println!("Time: {:.2?}", elapsed);
    println!("Tokens/sec: {:.2}", tokens_per_sec);
    println!("Ms/token: {:.2}", ms_per_token);
    println!("============================================================");

    Ok(())
}
