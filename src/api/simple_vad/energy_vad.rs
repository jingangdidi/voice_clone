/// Voice Activity Detection based on signal energy
pub struct EnergyVAD {
    sample_rate:      u32,
    frame_length_ms:  u32, // 25
    frame_shift_ms:   u32, // 20
    energy_threshold: f32, // 0.05
    pre_emphasis:     f32, // 0.95
}

impl EnergyVAD {
    /// Create a new EnergyVAD instance
    pub fn new(
        sample_rate:      u32,
        frame_length_ms:  u32,
        frame_shift_ms:   u32,
        energy_threshold: f32,
        pre_emphasis:     f32,
    ) -> Self {
        EnergyVAD {
            sample_rate,
            frame_length_ms,
            frame_shift_ms,
            energy_threshold,
            pre_emphasis,
        }
    }

    /// Process audio waveform and return VAD mask
    fn vad_mask(&self, content: &Vec<f32>) -> Vec<bool> {
        // Pre-emphasis filtering
        let mut pre_emphasized = vec![content[0]];
        for i in 1..content.len() {
            pre_emphasized.push(
                content[i] - self.pre_emphasis * content[i - 1]
            );
        }

        // Convert frame parameters to samples
        let frame_length = ((self.frame_length_ms * self.sample_rate) / 1000) as usize;
        let frame_shift = ((self.frame_shift_ms * self.sample_rate) / 1000) as usize;

        // Calculate number of frames
        let num_samples = content.len();
        let num_frames = if num_samples >= frame_length {
            (num_samples - frame_length + frame_shift) / frame_shift
        } else {
            0
        };

        // Compute energy for each frame
        let mut energy = vec![0.0; num_frames];
        for i in 0..num_frames {
            let start = i * frame_shift;
            let end = start + frame_length;
            if end > num_samples {
                break;
            }
            energy[i] = pre_emphasized[start..end]
                .iter()
                .map(|&x| x * x)
                .sum();
        }

        // Compute VAD mask
        energy.into_iter().map(|e| if e > self.energy_threshold { true } else { false }).collect()
    }

    /// Apply VAD mask to original audio (handles stereo input)
    pub fn apply_vad(&self, content: &Vec<f32>) -> Vec<(usize, usize)> {
        // Convert frame shift to samples
        let shift = ((self.frame_shift_ms * self.sample_rate) / 1000) as usize;

        // Process mono waveform (first channel)
        let vad = self.vad_mask(content);

        // get raw data active region
        let total_length = content.len();
        let mut result: Vec<(usize, usize)> = Vec::new(); // 存储active的起始和终止索引位置
        let mut merge_start = 0; // 多个区间是连续的则合并
        let mut merge_end = 0; // 多个区间是连续的则合并
        let mut start = 0;
        let mut end = 0;
        for (i, v) in vad.iter().enumerate() {
            if *v {
                start = i * shift;
                end = std::cmp::min((i + 1) * shift, total_length);
                if start == merge_end { // 是连续的，当前start和end合并到merge区间
                    merge_end = end;
                } else { // 不是连续的，保存merge区间，再用当前start和end重置merge区间
                    if merge_end > 0 {
                        result.push((merge_start, merge_end));
                    }
                    merge_start = start;
                    merge_end = end
                }
            }
        }
        // 保存最后一个区间
        if merge_start == start && merge_end == end{
            result.push((merge_start, merge_end));
        }
        result
    }
}
