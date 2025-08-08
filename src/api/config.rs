use std::fs::read_to_string;
use std::path::Path;

use serde::Deserialize;
use serde_json::from_str;

use crate::error::MyError;

/// data参数
#[derive(Deserialize)]
pub struct DataConfig {
    pub sampling_rate: usize, // 22050,
    pub filter_length: usize, // 1024,
    pub hop_length:    usize, // 256,
    pub win_length:    usize, // 1024,
    pub n_speakers:    usize, // 0
}

/// model参数
#[derive(Deserialize)]
pub struct ModelConfig {
    pub zero_g:                   bool,            // true
    pub inter_channels:           usize,           // 192
    pub hidden_channels:          usize,           // 192
    pub filter_channels:          usize,           // 768
    pub n_heads:                  usize,           // 2
    pub n_layers:                 usize,           // 6
    pub kernel_size:              usize,           // 3
    pub p_dropout:                f64,             // 0.1
    pub resblock:                 String,          // "1"
    pub resblock_kernel_sizes:    [usize; 3],      // [3, 7, 11]
    pub resblock_dilation_sizes:  [[usize; 3]; 3], // [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    pub upsample_rates:           [usize; 4],      // [8, 8, 2, 2]
    pub upsample_initial_channel: usize,           // 512
    pub upsample_kernel_sizes:    [usize; 4],      // [16, 16, 4, 4]
    pub gin_channels:             usize,           // 256
}

/// 存储config.json
#[derive(Deserialize)]
pub struct HParams {
    #[serde(rename = "_version_")] // 将`version`映射到json的`_version_`
    pub version: String, // "v2"
    pub data:    DataConfig,
    pub model:   ModelConfig,
}

impl HParams {
    /// 读取config.json创建对象
    pub fn new(file: &Path) -> Result<Self, MyError> {
        match read_to_string(file) {
            Ok(cfg) => from_str::<HParams>(&cfg).map_err(|e| MyError::StringToJsonError{error: e}),
            Err(e) => Err(MyError::ReadToStringError{file: file.to_str().unwrap().to_string(), error: e}),
        }
    }
}
