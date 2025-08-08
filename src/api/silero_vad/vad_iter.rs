use crate::{
    api::silero_vad::{
        silero,
        utils,
    },
    error::MyError,
};

//const DEBUG_SPEECH_PROB: bool = true;

#[derive(Debug)]
pub struct VadIter {
    silero: silero::Silero,
    params: Params,
    state: State,
}

impl VadIter {
    pub fn new(silero: silero::Silero, params: utils::VadParams) -> Self {
        Self {
            silero,
            params: Params::from(params),
            state: State::new(),
        }
    }

    pub fn process(&mut self, samples: &[f32], verbose: bool) -> Result<(), MyError> {
        self.reset_states();
        //for audio_frame in samples.chunks_exact(self.params.frame_size_samples) {
        let iter = samples.chunks_exact(self.params.frame_size_samples);
        let mut last_part = iter.remainder().to_vec();
        for audio_frame in iter {
            let speech_prob: f32 = self.silero.calc_level(audio_frame)?;
            self.state.update(&self.params, speech_prob, verbose);
        }

        // 最后一部分长度不够则在后面用0补齐
        let last_len = last_part.len();
        if last_len > 0 {
            last_part.extend_from_slice(&vec![0.0; self.params.frame_size_samples-last_len]);
            let speech_prob: f32 = self.silero.calc_level(&last_part)?;
            self.state.update(&self.params, speech_prob, verbose);
        }

        self.state.check_for_last_speech(samples.len(), self.params.speech_pad_samples);

        let speeches_len = self.state.speeches.len();
        for i in 0..speeches_len {
            if i == 0 {
                self.state.speeches[i].start = self.state.speeches[i].start.saturating_sub(self.params.speech_pad_samples as i64);
            }
            if i != speeches_len - 1 {
                let silence_duration = self.state.speeches[i+1].start - self.state.speeches[i].end;
                if silence_duration < 2 * self.params.speech_pad_samples as i64 {
                    self.state.speeches[i].end = silence_duration / 2;
                    self.state.speeches[i+1].start = self.state.speeches[i+1].start.saturating_sub(silence_duration / 2);
                } else {
                    self.state.speeches[i].end = (samples.len() as i64).min(self.state.speeches[i].end + self.params.speech_pad_samples as i64);
                    self.state.speeches[i+1].start = self.state.speeches[i+1].start.saturating_sub(self.params.speech_pad_samples as i64);
                }
            } else {
                self.state.speeches[i].end = (samples.len() as i64).min(self.state.speeches[i].end + self.params.speech_pad_samples as i64);
            }
        }

        // 下面这段改写自whisper_timestamped包的get_vad_segments函数
        // https://github.com/linto-ai/whisper-timestamped/blob/master/whisper_timestamped/transcribe.py
        let dilatation: i64 = 16000 / 2; // 即：round(dilatation * sample_rate), dilatation=0.5
        let mut speeches: Vec<utils::TimeStamp> = vec![];
        for speech in &self.state.speeches {
            let new_speech = utils::TimeStamp{
                start: if speech.start > dilatation {
                    speech.start - dilatation
                } else {
                    0
                },
                end: (samples.len() as i64).min(speech.end + dilatation),
            };
            let current_len = speeches.len();
            if current_len > 0 && speeches[current_len-1].end >= new_speech.start {
                speeches[current_len-1].end = new_speech.end;
            } else {
                speeches.push(new_speech);
            }
        }
        self.state.speeches = speeches;

        //println!("\n{:?}\n", self.state.speeches);

        Ok(())
    }

    //pub fn speeches(&self) -> &[utils::TimeStamp] {
    pub fn speeches(&self) -> Vec<(usize, usize)> {
        self.state.speeches.iter().map(|s| (s.start as usize, s.end as usize)).collect()
    }
}

impl VadIter {
    fn reset_states(&mut self) {
        self.silero.reset();
        self.state = State::new()
    }
}

#[allow(unused)]
#[derive(Debug)]
struct Params {
    frame_size: usize,
    threshold: f32,
    min_silence_duration_ms: usize,
    speech_pad_ms: usize,
    min_speech_duration_ms: usize,
    max_speech_duration_s: f32,
    sample_rate: usize,
    sr_per_ms: usize,
    frame_size_samples: usize,
    min_speech_samples: usize,
    speech_pad_samples: usize,
    max_speech_samples: f32,
    min_silence_samples: usize,
    min_silence_samples_at_max_speech: usize,
}

impl From<utils::VadParams> for Params {
    fn from(value: utils::VadParams) -> Self {
        let frame_size = value.frame_size;
        let threshold = value.threshold;
        let min_silence_duration_ms = value.min_silence_duration_ms;
        let speech_pad_ms = value.speech_pad_ms;
        let min_speech_duration_ms = value.min_speech_duration_ms;
        let max_speech_duration_s = value.max_speech_duration_s;
        let sample_rate = value.sample_rate;
        let sr_per_ms = sample_rate / 1000;
        let frame_size_samples = frame_size * sr_per_ms;
        let min_speech_samples = sr_per_ms * min_speech_duration_ms;
        let speech_pad_samples = sr_per_ms * speech_pad_ms;
        let max_speech_samples = sample_rate as f32 * max_speech_duration_s
            - frame_size_samples as f32
            - 2.0 * speech_pad_samples as f32;
        let min_silence_samples = sr_per_ms * min_silence_duration_ms;
        let min_silence_samples_at_max_speech = sr_per_ms * 98;
        Self {
            frame_size,
            threshold,
            min_silence_duration_ms,
            speech_pad_ms,
            min_speech_duration_ms,
            max_speech_duration_s,
            sample_rate,
            sr_per_ms,
            frame_size_samples,
            min_speech_samples,
            speech_pad_samples,
            max_speech_samples,
            min_silence_samples,
            min_silence_samples_at_max_speech,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    current_sample: usize,
    temp_end: usize,
    next_start: usize,
    prev_end: usize,
    triggered: bool,
    current_speech: utils::TimeStamp,
    speeches: Vec<utils::TimeStamp>,
}

impl State {
    fn new() -> Self {
        Default::default()
    }

    fn update(&mut self, params: &Params, speech_prob: f32, verbose: bool) {
        self.current_sample += params.frame_size_samples;
        if speech_prob >= params.threshold {
            if self.temp_end != 0 {
                self.temp_end = 0;
                if self.next_start < self.prev_end {
                    self.next_start = self
                        .current_sample
                        .saturating_sub(params.frame_size_samples);
                }
            }
            if !self.triggered {
                self.debug(speech_prob, params, "start", verbose);
                self.triggered = true;
                self.current_speech.start =
                    self.current_sample as i64 - params.frame_size_samples as i64;
                //println!("start: {}", self.current_speech.start);
                return;
            }
            //return;
        }
        if self.triggered
            && (self.current_sample as i64 - self.current_speech.start) as f32
                > params.max_speech_samples
        {
            if self.prev_end > 0 {
                self.current_speech.end = self.prev_end as _;
                self.take_speech();
                if self.next_start < self.prev_end {
                    self.triggered = false
                } else {
                    self.current_speech.start = self.next_start as _;
                }
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
            } else {
                //self.current_speech.end = self.current_sample as _;
                self.current_speech.end = self.current_sample as i64 - params.frame_size_samples as i64;
                self.take_speech();
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.triggered = false;
                return;
            }
            //return;
        }
        /*
        if speech_prob >= (params.threshold - 0.15) && (speech_prob < params.threshold) {
            if self.triggered {
                self.debug(speech_prob, params, "speaking", verbose)
            } else {
                self.debug(speech_prob, params, "silence", verbose)
            }
        }
        */
        if self.triggered && speech_prob < (params.threshold - 0.15) {
            self.debug(speech_prob, params, "end", verbose);
            if self.temp_end == 0 {
                //self.temp_end = self.current_sample;
                self.temp_end = self.current_sample.saturating_sub(params.frame_size_samples);
            }
            //if self.current_sample.saturating_sub(self.temp_end) > params.min_silence_samples_at_max_speech {
            if self.current_sample.saturating_sub(params.frame_size_samples).saturating_sub(self.temp_end) > params.min_silence_samples_at_max_speech {
                self.prev_end = self.temp_end;
            }
            //if self.current_sample.saturating_sub(self.temp_end) >= params.min_silence_samples {
            if self.current_sample.saturating_sub(params.frame_size_samples).saturating_sub(self.temp_end) >= params.min_silence_samples {
                self.current_speech.end = self.temp_end as _;
                if self.current_speech.end - self.current_speech.start
                    > params.min_speech_samples as _
                {
                    self.take_speech();
                    self.prev_end = 0;
                    self.next_start = 0;
                    self.temp_end = 0;
                    self.triggered = false;
                }
            }
        }
    }

    fn take_speech(&mut self) {
        self.speeches.push(std::mem::take(&mut self.current_speech)); // current speech becomes TimeStamp::default() due to take()
    }

    fn check_for_last_speech(&mut self, last_sample: usize, speech_pad_samples: usize) {
        if self.current_speech.start > 0 {
            //self.current_speech.end = last_sample as _;
            self.current_speech.end = last_sample.saturating_sub(speech_pad_samples) as i64; // 这里家去增加的pad
            self.take_speech();
            self.prev_end = 0;
            self.next_start = 0;
            self.temp_end = 0;
            self.triggered = false;
        }
    }

    fn debug(&self, speech_prob: f32, params: &Params, title: &str, verbose: bool) {
        if verbose {
            let speech = self.current_sample as f32
                - params.frame_size_samples as f32
                - if title == "end" {
                    params.speech_pad_samples
                } else {
                    0
                } as f32; // minus window_size_samples to get precise start time point.
            println!(
                "[{:10}: {:.3} s ({:.3}) {:8}]",
                title,
                speech / params.sample_rate as f32,
                speech_prob,
                self.current_sample - params.frame_size_samples,
            );
        }
    }
}
