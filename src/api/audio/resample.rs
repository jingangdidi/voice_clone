/// python的resampy包kaiser_best算法，filter参数可以指定是否使用best，false则使用fast
/// y_hat = resampy.resample(y, 44100, 22050, filter='kaiser_best', axis=-1)
/// https://github.com/bmcfee/resampy/blob/main/resampy/core.py
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32, bese_filter: bool) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let sample_ratio = to_rate as f32 / from_rate as f32;
    let output_length = (samples.len() as f32 * sample_ratio) as usize;
    let y: Vec<f32> = vec![0.0; output_length];

    let (precision, mut interp_win): (usize, Vec<f32>) = {
        let (precision, kaiser_best_half_window): (usize, &str) = if bese_filter {
            (8192, include_str!("../../../extract_kaiser_filter_data_from_npz/kaiser_best_half_window.txt"))
        } else {
            (512, include_str!("../../../extract_kaiser_filter_data_from_npz/kaiser_fast_half_window.txt"))
        };
        (
            precision,
            kaiser_best_half_window.split("\n").filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    trimmed.parse::<f32>().ok()
                }
            }).collect()
        )
    };

    if sample_ratio < 1.0 {
        interp_win = interp_win.iter().map(|s| s * sample_ratio).collect();
    }

    // 计算相邻元素的差分
    /*
    a = np.array([1, 2, 3])
    res1 = np.diff(a) # [1, 1]
    res2 = np.diff(a, append=0) # [1, 1, -3]
    */
    let mut interp_delta = Vec::with_capacity(interp_win.len());
    for i in 1..interp_win.len() {
        interp_delta.push(interp_win[i] - interp_win[i-1]);
    }
    interp_delta.push(0.0);

    let scale = sample_ratio.min(1.0);
    let time_increment = 1.0 / sample_ratio;
    let t_out: Vec<f32> = (0..output_length).map(|s| s as f32 * time_increment).collect();

    resample_loop(
        samples,
        &t_out,
        &interp_win,
        &interp_delta,
        precision,
        scale,
        y,
    )
}

fn resample_loop(x: &[f32], t_out: &[f32], interp_win: &[f32], interp_delta: &[f32], num_table: usize, scale: f32, mut y: Vec<f32>) -> Vec<f32> {
    let index_step = (scale * num_table as f32) as usize;
    let mut time_register: f32;

    let mut n: usize;
    let mut frac: f32;
    let mut index_frac: f32;
    let mut offset: usize;
    let mut eta: f32;
    let mut weight: f32;
    let mut i_max: usize;
    let mut k_max: usize;

    let nwin = interp_win.len();
    let n_orig = x.len();
    let n_out = t_out.len();

    for t in 0..n_out {
        time_register = t_out[t];

        // Grab the top bits as an index to the input buffer
        n = time_register as usize;

        // Grab the fractional component of the time index
        frac = scale * (time_register - time_register.trunc());

        // Offset into the filter
        index_frac = frac * num_table as f32;
        offset = index_frac as usize;

        // Interpolation factor
        eta = index_frac - index_frac.trunc();

        // Compute the left wing of the filter response
        i_max = (n + 1).min((nwin - offset) / index_step);
        for i in 0..i_max {
            weight = interp_win[offset + i * index_step] + eta * interp_delta[offset + i * index_step];
            y[t] += weight * x[n - i];
        }

        // Invert P
        frac = scale - frac;

        // Offset into the filter
        index_frac = frac * num_table as f32;
        offset = index_frac as usize;

        // Interpolation factor
        eta = index_frac - index_frac.trunc();

        // Compute the right wing of the filter response
        k_max = (n_orig - n - 1).min((nwin - offset) / index_step);
        for k in 0..k_max {
            weight = interp_win[offset + k * index_step] + eta * interp_delta[offset + k * index_step];
            y[t] += weight * x[n + k + 1];
        }
    }
    y
}
