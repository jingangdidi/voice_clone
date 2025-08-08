use crate::error::MyError;

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

pub struct Stft {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    //planner: FftPlanner<f32>,
}

impl Stft {
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        window_function: WindowFunction,
        window_periodic: bool,
    ) -> Self {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(n_fft);
        Self {
            n_fft,
            hop_length,
            forward,
            //planner,
            window: window_function.new(n_fft, window_periodic),
        }
    }

    pub fn forward(&self, input: Vec<f32>) -> Result<Vec<Vec<f32>>, MyError> {
        let signal_length = input.len();

        // For center padding (reflection)
        //let pad_length = self.n_fft / 2;
        let pad_length = (self.n_fft - self.hop_length) / 2;
        let padded_length = signal_length + 2 * pad_length;

        // Calculate the number of frames
        let num_frames = (padded_length - self.n_fft) / self.hop_length + 1;
        let n_freqs = self.n_fft / 2 + 1;

        let mut output: Vec<Vec<f32>> = (0..n_freqs).map(|_| vec![0.0; num_frames]).collect(); // 把复数的实数部分和虚数部分取平方后相加再取根号，[n_freqs, num_frames]
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); self.n_fft];

        let padded_channel = pad_reflect(input, pad_length);
        for frame in 0..num_frames {
            let start = frame * self.hop_length;
            for i in 0..self.n_fft {
                buffer[i] = Complex::new(padded_channel[start + i] * self.window[i], 0.0);
            }
            self.forward.process(buffer.as_mut_slice());
            for (freq, &value) in buffer.iter().take(n_freqs).enumerate() {
                // +-------------------------+
                // | pub struct Complex<T> { |
                // |     pub re: T,          | // 实数部分
                // |     pub im: T,          | // 虚数部分
                // | }                       |
                // +-------------------------+
                // python源代码：spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
                output[freq][frame] = (value.re.powi(2) + value.im.powi(2) + 1e-6).sqrt();
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    //Rectangular,
    Hann,
    //Hamming,
    //Blackman,
    //Gaussian(f32), // Standard deviation
    //Triangular,
    //Bartlett,
    //FlatTop,
}

impl WindowFunction {
    pub fn new(&self, size: usize, periodic: bool) -> Vec<f32> {
        let mut window = Vec::with_capacity(size);
        let m = if periodic { size + 1 } else { size };

        for n in 0..size {
            let x = n as f32 / (m - 1) as f32;
            let value = match self {
                //WindowFunction::Rectangular => 1.0,
                WindowFunction::Hann => 0.5 * (1.0 - (2.0 * PI * x).cos()),
                //WindowFunction::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
                //WindowFunction::Blackman => 0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos(),
                //WindowFunction::Gaussian(sigma) => -0.5 * (1.0 / *sigma * (x - 0.5)).powi(2).exp(),
                //WindowFunction::Triangular => 1.0 - (2.0 * x - 1.0).abs(),
                //WindowFunction::Bartlett => if x < 0.5 {
                //    2.0 * x
                //} else {
                //    2.0 - 2.0 * x
                //},
                //WindowFunction::FlatTop => 0.21557895 - 0.41663158 * (2.0 * PI * x).cos()
                //        + 0.277263158 * (4.0 * PI * x).cos()
                //        - 0.083578947 * (6.0 * PI * x).cos()
                //        + 0.006947368 * (8.0 * PI * x).cos(),
            };
            window.push(value);
        }
        window
    }
}

fn pad_reflect(signal: Vec<f32>, pad_length: usize) -> Vec<f32> {
    let signal_length = signal.len();
    let mut padded = [vec![0.0; pad_length], signal.clone(), vec![0.0; pad_length]].concat();

    // Reflect at the beginning
    for i in 0..pad_length {
        padded[pad_length - 1 - i] = signal[i + 1];
    }

    // Reflect at the end
    for i in 0..pad_length {
        padded[pad_length + signal_length + i] = signal[signal_length - 2 - i];
    }
    padded
}
