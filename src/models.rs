use serde::{Deserialize, Serialize};

/// Quantization levels ordered from best quality to most compressed.
/// Used for dynamic quantization selection: try the best that fits.
pub const QUANT_HIERARCHY: &[&str] = &["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];

/// Bytes per parameter for each quantization level.
pub fn quant_bpp(quant: &str) -> f64 {
    match quant {
        "F32" => 4.0,
        "F16" | "BF16" => 2.0,
        "Q8_0" => 1.05,
        "Q6_K" => 0.80,
        "Q5_K_M" => 0.68,
        "Q4_K_M" | "Q4_0" => 0.58,
        "Q3_K_M" => 0.48,
        "Q2_K" => 0.37,
        _ => 0.58,
    }
}

/// Speed multiplier for quantization (lower quant = faster inference).
pub fn quant_speed_multiplier(quant: &str) -> f64 {
    match quant {
        "F16" | "BF16" => 0.6,
        "Q8_0" => 0.8,
        "Q6_K" => 0.95,
        "Q5_K_M" => 1.0,
        "Q4_K_M" | "Q4_0" => 1.15,
        "Q3_K_M" => 1.25,
        "Q2_K" => 1.35,
        _ => 1.0,
    }
}

/// Quality penalty for quantization (lower quant = lower quality).
pub fn quant_quality_penalty(quant: &str) -> f64 {
    match quant {
        "F16" | "BF16" => 0.0,
        "Q8_0" => 0.0,
        "Q6_K" => -1.0,
        "Q5_K_M" => -2.0,
        "Q4_K_M" | "Q4_0" => -5.0,
        "Q3_K_M" => -8.0,
        "Q2_K" => -12.0,
        _ => -5.0,
    }
}

/// Use-case category for scoring weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum UseCase {
    General,
    Coding,
    Reasoning,
    Chat,
    Multimodal,
    Embedding,
}

impl UseCase {
    pub fn label(&self) -> &'static str {
        match self {
            UseCase::General => "General",
            UseCase::Coding => "Coding",
            UseCase::Reasoning => "Reasoning",
            UseCase::Chat => "Chat",
            UseCase::Multimodal => "Multimodal",
            UseCase::Embedding => "Embedding",
        }
    }

    /// Infer use-case from the model's use_case field and name.
    pub fn from_model(model: &LlmModel) -> Self {
        let name = model.name.to_lowercase();
        let use_case = model.use_case.to_lowercase();

        if use_case.contains("embedding") || name.contains("embed") || name.contains("bge") {
            UseCase::Embedding
        } else if name.contains("code") || use_case.contains("code") {
            UseCase::Coding
        } else if use_case.contains("vision") || use_case.contains("multimodal") {
            UseCase::Multimodal
        } else if use_case.contains("reason") || use_case.contains("chain-of-thought") || name.contains("deepseek-r1") {
            UseCase::Reasoning
        } else if use_case.contains("chat") || use_case.contains("instruction") {
            UseCase::Chat
        } else {
            UseCase::General
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmModel {
    pub name: String,
    pub provider: String,
    pub parameter_count: String,
    #[serde(default)]
    pub parameters_raw: Option<u64>,
    pub min_ram_gb: f64,
    pub recommended_ram_gb: f64,
    pub min_vram_gb: Option<f64>,
    pub quantization: String,
    pub context_length: u32,
    pub use_case: String,
    #[serde(default)]
    pub is_moe: bool,
    #[serde(default)]
    pub num_experts: Option<u32>,
    #[serde(default)]
    pub active_experts: Option<u32>,
    #[serde(default)]
    pub active_parameters: Option<u64>,
}

impl LlmModel {
    /// Bytes-per-parameter for the model's quantization level.
    fn quant_bpp(&self) -> f64 {
        quant_bpp(&self.quantization)
    }

    /// Parameter count in billions, extracted from parameters_raw or parameter_count.
    pub fn params_b(&self) -> f64 {
        if let Some(raw) = self.parameters_raw {
            raw as f64 / 1_000_000_000.0
        } else {
            // Parse from string like "7B", "1.1B", "137M"
            let s = self.parameter_count.trim().to_uppercase();
            if let Some(num_str) = s.strip_suffix('B') {
                num_str.parse::<f64>().unwrap_or(7.0)
            } else if let Some(num_str) = s.strip_suffix('M') {
                num_str.parse::<f64>().unwrap_or(0.0) / 1000.0
            } else {
                7.0
            }
        }
    }

    /// Estimate memory required (GB) at a given quantization and context length.
    /// Formula: model_weights + KV_cache + runtime_overhead
    pub fn estimate_memory_gb(&self, quant: &str, ctx: u32) -> f64 {
        let bpp = quant_bpp(quant);
        let params = self.params_b();
        let model_mem = params * bpp;
        // KV cache: ~0.000008 GB per billion params per context token
        let kv_cache = 0.000008 * params * ctx as f64;
        // Runtime overhead (CUDA/Metal context, buffers)
        let overhead = 0.5;
        model_mem + kv_cache + overhead
    }

    /// Select the best quantization level that fits within a memory budget.
    /// Returns the quant name and estimated memory in GB, or None if nothing fits.
    pub fn best_quant_for_budget(&self, budget_gb: f64, ctx: u32) -> Option<(&'static str, f64)> {
        // Try best quality first
        for &q in QUANT_HIERARCHY {
            let mem = self.estimate_memory_gb(q, ctx);
            if mem <= budget_gb {
                return Some((q, mem));
            }
        }
        // Try halving context once
        let half_ctx = ctx / 2;
        if half_ctx >= 1024 {
            for &q in QUANT_HIERARCHY {
                let mem = self.estimate_memory_gb(q, half_ctx);
                if mem <= budget_gb {
                    return Some((q, mem));
                }
            }
        }
        None
    }

    /// For MoE models, compute estimated VRAM for active experts only.
    /// Returns None for dense models.
    pub fn moe_active_vram_gb(&self) -> Option<f64> {
        if !self.is_moe {
            return None;
        }
        let active_params = self.active_parameters? as f64;
        let bpp = self.quant_bpp();
        let size_gb = (active_params * bpp) / (1024.0 * 1024.0 * 1024.0);
        Some((size_gb * 1.1).max(0.5))
    }

    /// For MoE models, compute RAM needed for offloaded (inactive) experts.
    /// Returns None for dense models.
    pub fn moe_offloaded_ram_gb(&self) -> Option<f64> {
        if !self.is_moe {
            return None;
        }
        let active = self.active_parameters? as f64;
        let total = self.parameters_raw? as f64;
        let inactive = total - active;
        if inactive <= 0.0 {
            return Some(0.0);
        }
        let bpp = self.quant_bpp();
        Some((inactive * bpp) / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Intermediate struct matching the JSON schema from the scraper.
/// Extra fields are ignored when mapping to LlmModel.
#[derive(Deserialize)]
struct HfModelEntry {
    name: String,
    provider: String,
    parameter_count: String,
    #[serde(default)]
    parameters_raw: Option<u64>,
    min_ram_gb: f64,
    recommended_ram_gb: f64,
    min_vram_gb: Option<f64>,
    quantization: String,
    context_length: u32,
    use_case: String,
    #[serde(default)]
    is_moe: bool,
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    active_experts: Option<u32>,
    #[serde(default)]
    active_parameters: Option<u64>,
}

const HF_MODELS_JSON: &str = include_str!("../data/hf_models.json");

/// Known architectures for filtering HuggingFace models.
const KNOWN_ARCHITECTURES: &[&str] = &[
    "llama", "mistral", "mixtral", "qwen", "gemma", "phi", "gpt2", "gpt_neox",
    "falcon", "mpt", "bloom", "starcoder", "codegen", "deepseek", "internlm",
    "baichuan", "yi", "command", "olmo", "stablelm", "persimmon", "mamba",
    "rwkv", "recurrent_gemma", "cohere", "dbrx", "jamba", "arctic",
    "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM",
    "GemmaForCausalLM", "PhiForCausalLM", "GPTNeoXForCausalLM",
];

fn cache_path() -> Option<std::path::PathBuf> {
    dirs::cache_dir().map(|d| d.join("llmfit").join("models.json"))
}

fn load_cached_models() -> Vec<LlmModel> {
    let Some(path) = cache_path() else { return vec![] };
    let Ok(data) = std::fs::read_to_string(&path) else { return vec![] };
    serde_json::from_str(&data).unwrap_or_default()
}

pub struct ModelDatabase {
    models: Vec<LlmModel>,
}

impl ModelDatabase {
    pub fn new() -> Self {
        Self::load(false)
    }

    pub fn load(offline: bool) -> Self {
        let entries: Vec<HfModelEntry> =
            serde_json::from_str(HF_MODELS_JSON).expect("Failed to parse embedded hf_models.json");

        let mut models: Vec<LlmModel> = entries
            .into_iter()
            .map(|e| LlmModel {
                name: e.name,
                provider: e.provider,
                parameter_count: e.parameter_count,
                parameters_raw: e.parameters_raw,
                min_ram_gb: e.min_ram_gb,
                recommended_ram_gb: e.recommended_ram_gb,
                min_vram_gb: e.min_vram_gb,
                quantization: e.quantization,
                context_length: e.context_length,
                use_case: e.use_case,
                is_moe: e.is_moe,
                num_experts: e.num_experts,
                active_experts: e.active_experts,
                active_parameters: e.active_parameters,
            })
            .collect();

        if !offline {
            let cached = load_cached_models();
            if !cached.is_empty() {
                // Cached entries override embedded ones by name
                let mut name_set: std::collections::HashSet<String> = std::collections::HashSet::new();
                let mut merged = Vec::new();
                for m in cached {
                    name_set.insert(m.name.clone());
                    merged.push(m);
                }
                for m in models {
                    if !name_set.contains(&m.name) {
                        merged.push(m);
                    }
                }
                models = merged;
            }
        }

        ModelDatabase { models }
    }

    pub fn get_all_models(&self) -> &Vec<LlmModel> {
        &self.models
    }

    pub fn find_model(&self, query: &str) -> Vec<&LlmModel> {
        let query_lower = query.to_lowercase();
        self.models
            .iter()
            .filter(|m| {
                m.name.to_lowercase().contains(&query_lower)
                    || m.provider.to_lowercase().contains(&query_lower)
                    || m.parameter_count.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    pub fn models_fitting_system(&self, available_ram_gb: f64, has_gpu: bool, vram_gb: Option<f64>) -> Vec<&LlmModel> {
        self.models
            .iter()
            .filter(|m| {
                // Check RAM requirement
                let ram_ok = m.min_ram_gb <= available_ram_gb;
                
                // If model requires GPU and system has GPU, check VRAM
                if let Some(min_vram) = m.min_vram_gb {
                    if has_gpu {
                        if let Some(system_vram) = vram_gb {
                            ram_ok && min_vram <= system_vram
                        } else {
                            // GPU detected but VRAM unknown, allow but warn
                            ram_ok
                        }
                    } else {
                        // Model prefers GPU but can run on CPU with enough RAM
                        ram_ok && available_ram_gb >= m.recommended_ram_gb
                    }
                } else {
                    ram_ok
                }
            })
            .collect()
    }
}

/// Fetch models from HuggingFace API and save to cache.
pub fn update_from_huggingface() {
    use std::collections::HashSet;

    println!("Fetching models from HuggingFace...");

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client");

    let mut all_models: Vec<LlmModel> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    let limit = 100;
    let max_pages = 5; // 500 models max

    for page in 0..max_pages {
        let offset = page * limit;
        let url = format!(
            "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit={}&offset={}",
            limit, offset
        );

        let resp = match client.get(&url).send() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to fetch page {}: {}", page + 1, e);
                break;
            }
        };

        if !resp.status().is_success() {
            eprintln!("API returned status {} on page {}", resp.status(), page + 1);
            break;
        }

        let entries: Vec<serde_json::Value> = match resp.json() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to parse response: {}", e);
                break;
            }
        };

        if entries.is_empty() {
            break;
        }

        for entry in &entries {
            if let Some(model) = hf_entry_to_model(entry) {
                if !seen_names.contains(&model.name) {
                    seen_names.insert(model.name.clone());
                    all_models.push(model);
                }
            }
        }

        print!("\r  Fetched {} models so far...", all_models.len());
    }
    println!();

    if all_models.is_empty() {
        println!("No models fetched.");
        return;
    }

    // Load existing embedded DB to count new models
    let embedded: Vec<HfModelEntry> =
        serde_json::from_str(HF_MODELS_JSON).unwrap_or_default();
    let embedded_names: HashSet<String> = embedded.iter().map(|e| e.name.clone()).collect();
    let new_count = all_models.iter().filter(|m| !embedded_names.contains(&m.name)).count();

    // Save to cache
    let Some(path) = cache_path() else {
        eprintln!("Could not determine cache directory");
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let total = {
        // Merge: cached override embedded
        let mut name_set: HashSet<String> = HashSet::new();
        let mut merged = Vec::new();
        for m in &all_models {
            name_set.insert(m.name.clone());
            merged.push(m.name.clone());
        }
        for e in &embedded {
            if !name_set.contains(&e.name) {
                merged.push(e.name.clone());
            }
        }
        merged.len()
    };

    match serde_json::to_string_pretty(&all_models) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                eprintln!("Failed to write cache: {}", e);
                return;
            }
        }
        Err(e) => {
            eprintln!("Failed to serialize models: {}", e);
            return;
        }
    }

    println!("Updated: {} models fetched, {} new, {} total", all_models.len(), new_count, total);
}

/// Extract parameter count from model name like "Qwen2.5-7B-Instruct" -> 7_000_000_000
fn extract_params_from_name(name: &str) -> Option<u64> {
    // Look for patterns like "7B", "0.6B", "70B", "1.5B", "405B", "110M"
    let name_upper = name.to_uppercase();
    // Split by common separators
    for part in name_upper.split(|c: char| c == '-' || c == '_' || c == '/') {
        if let Some(num_str) = part.strip_suffix('B') {
            if let Ok(num) = num_str.parse::<f64>() {
                if num > 0.0 && num < 10000.0 {
                    return Some((num * 1_000_000_000.0) as u64);
                }
            }
        } else if let Some(num_str) = part.strip_suffix('M') {
            if let Ok(num) = num_str.parse::<f64>() {
                if num > 0.0 && num < 10000.0 {
                    return Some((num * 1_000_000.0) as u64);
                }
            }
        }
    }
    None
}

/// Convert a HuggingFace API model entry to our LlmModel format.
fn hf_entry_to_model(entry: &serde_json::Value) -> Option<LlmModel> {
    let model_id = entry.get("modelId")?.as_str()?;
    let downloads = entry.get("downloads").and_then(|d| d.as_u64()).unwrap_or(0);

    // Filter: >1000 downloads
    if downloads <= 1000 {
        return None;
    }

    // Check for known architecture via tags
    let tags = entry.get("tags").and_then(|t| t.as_array());
    let has_known_arch = if let Some(tags) = tags {
        tags.iter().any(|t| {
            let s = t.as_str().unwrap_or("").to_lowercase();
            KNOWN_ARCHITECTURES.iter().any(|k| s.contains(&k.to_lowercase()))
        })
    } else {
        false
    };

    // Also accept if library_name is "transformers" (most text-gen models)
    let is_transformers = entry.get("library_name")
        .and_then(|v| v.as_str())
        .map(|s| s == "transformers")
        .unwrap_or(false);

    if !has_known_arch && !is_transformers {
        return None;
    }

    // Try to extract parameter count from model name (e.g., "Qwen2.5-7B-Instruct" -> 7B)
    let params_raw = extract_params_from_name(model_id)?;
    if params_raw == 0 {
        return None;
    }

    let params_b = params_raw as f64 / 1_000_000_000.0;
    let parameter_count = if params_b >= 1.0 {
        format!("{:.1}B", params_b)
    } else {
        format!("{:.0}M", params_b * 1000.0)
    };

    // Compute memory requirements using same formulas as existing codebase
    // min_vram_gb: model at Q4_K_M quantization
    let bpp_q4 = quant_bpp("Q4_K_M");
    let min_vram_gb = params_b * bpp_q4 + 0.5; // model weights + overhead

    // min_ram_gb: need more headroom for CPU inference
    let min_ram_gb = min_vram_gb * 1.2;
    let recommended_ram_gb = min_ram_gb * 1.5;

    // Extract provider from model_id (org/model -> org)
    let provider = model_id.split('/').next().unwrap_or("unknown").to_string();

    // Detect use case from tags/name
    let name_lower = model_id.to_lowercase();
    let use_case = if name_lower.contains("code") || name_lower.contains("starcoder") {
        "Code generation".to_string()
    } else if name_lower.contains("chat") || name_lower.contains("instruct") {
        "Chat / instruction following".to_string()
    } else if name_lower.contains("vision") || name_lower.contains("vl") {
        "Vision / multimodal".to_string()
    } else {
        "General text generation".to_string()
    };

    // Default context length (conservative estimate)
    let context_length = 4096;

    Some(LlmModel {
        name: model_id.to_string(),
        provider,
        parameter_count,
        parameters_raw: Some(params_raw),
        min_ram_gb,
        recommended_ram_gb,
        min_vram_gb: Some(min_vram_gb),
        quantization: "Q4_K_M".to_string(),
        context_length,
        use_case,
        is_moe: name_lower.contains("moe") || name_lower.contains("mixtral") || name_lower.contains("dbrx"),
        num_experts: None,
        active_experts: None,
        active_parameters: None,
    })
}
