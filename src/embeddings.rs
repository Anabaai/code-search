use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu; // Use CPU for portability and simplicity
        let api = Api::new()?;
        let repo = api.repo(Repo::new(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            RepoType::Model,
        ));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], verify_dtype(&device), &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let tokens = self.tokenizer.encode_batch(texts.to_vec(), true).map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let mask = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(mask.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        
        // Mean pooling with attention mask
        // embeddings: [B, Seq, Hidden]
        // attention_mask: [B, Seq]
        let (_b, _seq, hidden_size) = embeddings.dims3()?;
        
        // Expand mask to [B, Seq, Hidden]
        let mask_expanded = attention_mask
            .unsqueeze(2)?
            .broadcast_as((_b, _seq, hidden_size))?
            .to_dtype(candle_core::DType::F32)?;
            
        let masked_embeddings = embeddings.mul(&mask_expanded)?;
        let sum_embeddings = masked_embeddings.sum(1)?;
        let sum_mask = mask_expanded.sum(1)?;
        
        // Avoid division by zero by clamping mask sum
        let sum_mask = sum_mask.clamp(1e-9, f32::MAX)?;
        
        let pooled_embeddings = (sum_embeddings / sum_mask)?;
        let normalized_embeddings = normalize_l2(&pooled_embeddings)?;
        
        let embeddings_vec: Vec<Vec<f32>> = normalized_embeddings.to_vec2()?;
        Ok(embeddings_vec)
    }
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(v.broadcast_div(&norm)?)
}

fn verify_dtype(_device: &Device) -> candle_core::DType {
    // Default to F32 for CPU
    candle_core::DType::F32
}
