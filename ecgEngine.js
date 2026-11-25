/**
 * Wearable Single-Lead ECG Synthesizer, Sensor & Activity Profiles, Artifact Injector,
 * Adaptive Denoiser, FFT Spectral Analyzer, Fiducial & HRV Feature Extractor,
 * and 60-Second Multi-Window Holter Session Engine.
 * Sampling rate: 250 Hz.
 */

(function (global) {
  const ns = (global.CardioXAI = global.CardioXAI || {});

const FS = 250; // Hz
const WINDOW_SECONDS = 10;
const WINDOW_SAMPLES = FS * WINDOW_SECONDS;

const SENSOR_PROFILES = {
  WATCH_I: {
    key: 'WATCH_I',
    label: 'Smartwatch Lead I',
    badge: '250 Hz · Watch Lead I',
    gain: 0.94,
    pScale: 0.9,
    noiseMult: 1.15,
    description: 'Wrist-worn dry electrodes; higher motion & EMG susceptibility.'
  },
  PATCH_II: {
    key: 'PATCH_II',
    label: 'Chest Patch Lead II',
    badge: '250 Hz · Patch Lead II',
    gain: 1.12,
    pScale: 1.25,
    noiseMult: 0.65,
    description: 'Adhesive sternal patch; prominent P-waves and low motion drift.'
  },
  HOLTER_V5: {
    key: 'HOLTER_V5',
    label: 'Holter V5',
    badge: '250 Hz · Holter V5',
    gain: 1.22,
    pScale: 1.0,
    noiseMult: 0.5,
    description: 'Clinical ambulatory lead; high QRS amplitude and clean baseline.'
  },
  RING_HYBRID: {
    key: 'RING_HYBRID',
    label: 'Smart-Ring Bi-Finger',
    badge: '250 Hz · Smart Ring',
    gain: 0.82,
    pScale: 0.78,
    noiseMult: 1.32,
    description: 'Ultra-compact finger dry-contact lead; elevated EMG & baseline sensitivity.'
  }
};

const ACTIVITY_PROFILES = {
  RESTING: {
    key: 'RESTING',
    label: 'Resting',
    motion: 8,
    baseline: 12,
    emg: 6,
    powerline60: false,
    description: 'Seated resting state with minimal skeletal muscle noise.'
  },
  WALKING: {
    key: 'WALKING',
    label: 'Walking',
    motion: 36,
    baseline: 30,
    emg: 22,
    powerline60: false,
    description: '2 Hz stride-coupled motion bursts and moderate baseline sway.'
  },
  JOGGING: {
    key: 'JOGGING',
    label: 'Jogging',
    motion: 74,
    baseline: 58,
    emg: 52,
    powerline60: false,
    description: 'High-amplitude motion artifacts and elevated pectoral/arm EMG.'
  },
  SLEEP: {
    key: 'SLEEP',
    label: 'Sleep / Mains',
    motion: 4,
    baseline: 20,
    emg: 5,
    powerline60: true,
    description: 'Low motion overnight recording with 60 Hz bedside mains coupling.'
  }
};

const RHYTHM_PRESETS = {
  NSR: {
    code: 'NSR',
    name: 'Normal Sinus Rhythm',
    defaultBpm: 72,
    severity: 'normal',
    description: 'Regular sinus P-waves preceding narrow QRS complexes with physiologic respiratory sinus arrhythmia.'
  },
  AFIB: {
    code: 'AFIB',
    name: 'Atrial Fibrillation',
    defaultBpm: 108,
    severity: 'warning',
    description: 'Irregularly irregular ventricular response, absent P-waves, and chaotic low-amplitude fibrillatory baseline.'
  },
  PVC: {
    code: 'PVC',
    name: 'Premature Ventricular Contraction',
    defaultBpm: 74,
    severity: 'warning',
    description: 'Sinus rhythm interrupted by premature wide, high-amplitude discordant QRS complexes and compensatory pauses.'
  },
  PAC: {
    code: 'PAC',
    name: 'Premature Atrial Contraction',
    defaultBpm: 76,
    severity: 'caution',
    description: 'Early narrow-complex beats triggered by ectopic atrial foci with altered/inverted P-wave morphology.'
  },
  VTACH: {
    code: 'VTACH',
    name: 'Ventricular Tachycardia',
    defaultBpm: 168,
    severity: 'critical',
    description: 'Rapid monomorphic wide-complex tachycardia originating from ventricular focus with AV dissociation.'
  },
  LBBB: {
    code: 'LBBB',
    name: 'Left Bundle Branch Block',
    defaultBpm: 68,
    severity: 'caution',
    description: 'Prolonged QRS (>120 ms) with notched monophasic R-wave plateau and secondary discordant ST-T depression.'
  },
  BIGEMINY: {
    code: 'BIGEMINY',
    name: 'Ventricular Bigeminy',
    defaultBpm: 78,
    severity: 'warning',
    description: 'Coupled 1:1 alternating normal sinus beats and premature wide ventricular ectopic beats with post-ectopic pauses.'
  },
  PAROXYSMAL_AF: {
    code: 'PAROXYSMAL_AF',
    name: 'Paroxysmal AFib Onset',
    defaultBpm: 94,
    severity: 'warning',
    description: 'Dynamic transition from organized sinus rhythm (0–4.5 s) into acute irregularly irregular Atrial Fibrillation.'
  }
};

const SAMPLE_CASES = {
  CASE_PVC_BIGEMINY: {
    key: 'CASE_PVC_BIGEMINY',
    label: 'Case 01 · Wearable PVC Bigeminy',
    preset: 'BIGEMINY',
    bpm: 76,
    motion: 14,
    baseline: 18,
    emg: 10,
    seed: 777
  },
  CASE_PAF_ONSET: {
    key: 'CASE_PAF_ONSET',
    label: 'Case 02 · Paroxysmal AFib Transition',
    preset: 'PAROXYSMAL_AF',
    bpm: 96,
    motion: 12,
    baseline: 14,
    emg: 8,
    seed: 512
  },
  CASE_JOG_NSR: {
    key: 'CASE_JOG_NSR',
    label: 'Case 03 · Jogging Motion + Sinus',
    preset: 'NSR',
    bpm: 134,
    motion: 68,
    baseline: 48,
    emg: 42,
    seed: 303
  },
  CASE_VT_RUN: {
    key: 'CASE_VT_RUN',
    label: 'Case 04 · Sustained VTach Alert',
    preset: 'VTACH',
    bpm: 172,
    motion: 10,
    baseline: 10,
    emg: 8,
    seed: 909
  }
};

function gaussian(x, mu, sigma, amp) {
  const z = (x - mu) / sigma;
  return amp * Math.exp(-0.5 * z * z);
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generates an individual heartbeat waveform centered around R-peak at offset 0 (seconds).
 */
function renderBeatSample(dt, beatType, counterfactuals, profile = SENSOR_PROFILES.WATCH_I) {
  const restoreP = counterfactuals.restoreP;
  const narrowQrs = counterfactuals.narrowQrs;
  const g = profile.gain;
  const pG = profile.pScale;

  let val = 0;

  if (beatType === 'VTACH' && !narrowQrs) {
    val += gaussian(dt, -0.02, 0.045, 0.92 * g);
    val += gaussian(dt, 0.045, 0.04, -0.58 * g);
    val += gaussian(dt, 0.16, 0.065, -0.25 * g);
    if (restoreP) {
      val += gaussian(dt, -0.165, 0.022, 0.14 * pG);
    }
    return val;
  }

  if (beatType === 'PVC' && !narrowQrs) {
    val += gaussian(dt, -0.06, 0.02, -0.22 * g);
    val += gaussian(dt, -0.005, 0.032, 1.38 * g);
    val += gaussian(dt, 0.055, 0.034, -0.62 * g);
    val += gaussian(dt, 0.21, 0.055, -0.34 * g);
    if (restoreP) {
      val += gaussian(dt, -0.16, 0.02, 0.13 * pG);
    }
    return val;
  }

  if (beatType === 'LBBB' && !narrowQrs) {
    val += gaussian(dt, -0.165, 0.022, 0.13 * pG);
    val += gaussian(dt, -0.028, 0.018, 0.68 * g);
    val += gaussian(dt, 0.018, 0.022, 0.84 * g);
    val += gaussian(dt, 0.068, 0.02, -0.18 * g);
    val += gaussian(dt, 0.21, 0.055, -0.22 * g);
    return val;
  }

  const hasNormalP =
    beatType === 'NSR' ||
    restoreP ||
    (beatType === 'PVC' && narrowQrs) ||
    (beatType === 'VTACH' && narrowQrs) ||
    (beatType === 'LBBB' && narrowQrs);

  if (hasNormalP) {
    val += gaussian(dt, -0.165, 0.022, 0.14 * pG);
  } else if (beatType === 'PAC') {
    val += gaussian(dt, -0.12, 0.018, -0.11 * pG);
  }

  val += gaussian(dt, -0.032, 0.009, -0.14 * g);
  val += gaussian(dt, 0.0, 0.011, 1.15 * g);
  val += gaussian(dt, 0.032, 0.011, -0.26 * g);
  val += gaussian(dt, 0.22, 0.044, 0.28 * g);

  return val;
}

/**
 * Returns fiducial wave offsets (ms relative to R-peak) for Microscope auto-annotation.
 */
function getBeatFiducials(beat, counterfactuals = {}) {
  if (!beat) return [];
  const t = beat.type;
  const narrow = !!counterfactuals.narrowQrs;
  const restoreP = !!counterfactuals.restoreP;
  const markers = [];

  if ((t === 'NSR' || t === 'LBBB' || restoreP) && t !== 'AFIB') {
    markers.push({ label: 'P', offsetMs: -165, color: '#38bdf8' });
  } else if (t === 'PAC') {
    markers.push({ label: "P'", offsetMs: -120, color: '#a78bfa' });
  }

  if (t === 'PVC' && !narrow) {
    markers.push({ label: 'Q', offsetMs: -60, color: '#fbbf24' });
    markers.push({ label: 'R*', offsetMs: -5, color: '#f43f5e' });
    markers.push({ label: 'S*', offsetMs: 55, color: '#fbbf24' });
    markers.push({ label: 'T*', offsetMs: 210, color: '#fb7185' });
  } else if (t === 'VTACH' && !narrow) {
    markers.push({ label: 'R*', offsetMs: -20, color: '#f43f5e' });
    markers.push({ label: 'S*', offsetMs: 45, color: '#fbbf24' });
    markers.push({ label: 'T*', offsetMs: 160, color: '#fb7185' });
  } else if (t === 'LBBB' && !narrow) {
    markers.push({ label: 'R1', offsetMs: -28, color: '#a78bfa' });
    markers.push({ label: 'R2', offsetMs: 18, color: '#f43f5e' });
    markers.push({ label: 'T-', offsetMs: 210, color: '#fb7185' });
  } else {
    markers.push({ label: 'Q', offsetMs: -32, color: '#94a3b8' });
    markers.push({ label: 'R', offsetMs: 0, color: '#34d399' });
    markers.push({ label: 'S', offsetMs: 32, color: '#94a3b8' });
    markers.push({ label: 'T', offsetMs: 220, color: '#34d399' });
  }

  return markers;
}

/**
 * Generates a clean Normal Sinus reference beat array for Microscope template comparison.
 */
function generateSinusTemplateBeat(preSamples, postSamples, sensorProfile = 'WATCH_I') {
  const profile = SENSOR_PROFILES[sensorProfile] || SENSOR_PROFILES.WATCH_I;
  const total = preSamples + postSamples;
  const arr = new Float32Array(total);
  const noCf = { restoreP: false, regularizeRR: false, narrowQrs: false };
  for (let k = 0; k < total; k++) {
    const dt = (k - preSamples) / FS;
    arr[k] = renderBeatSample(dt, 'NSR', noCf, profile);
  }
  return arr;
}

/**
 * Generates a 10-second synthetic wearable ECG record with beat metadata,
 * wearable artifacts, and adaptive denoising.
 */
function generateSyntheticECG({
  preset = 'NSR',
  sensorProfile = 'WATCH_I',
  bpm = 72,
  motionNoise = 10,
  baselineWander = 15,
  emgNoise = 8,
  powerline60 = false,
  denoise = true,
  counterfactuals = { restoreP: false, regularizeRR: false, narrowQrs: false },
  seed = 101
}) {
  const profile = SENSOR_PROFILES[sensorProfile] || SENSOR_PROFILES.WATCH_I;
  const rng = mulberry32(seed);
  const clean = new Float32Array(WINDOW_SAMPLES);
  const rawNoisy = new Float32Array(WINDOW_SAMPLES);

  const baseRR = 60 / Math.max(40, Math.min(200, bpm));
  const beats = [];

  let tCursor = 0.48;
  let beatIdx = 0;

  while (tCursor < WINDOW_SECONDS - 0.35) {
    let localType = preset;
    let rr = baseRR;

    if (preset === 'PAROXYSMAL_AF') {
      if (tCursor < 4.5) {
        localType = 'NSR';
        rr = baseRR * 1.12 * (1 + 0.03 * Math.sin(tCursor * 1.2));
      } else {
        localType = 'AFIB';
        if (!counterfactuals.regularizeRR) {
          const jitter = (rng() - 0.48) * 0.52 * (baseRR * 0.85);
          rr = Math.max(0.36, baseRR * 0.84 + jitter);
        } else {
          rr = baseRR * (1 + (rng() - 0.5) * 0.015);
        }
      }
    } else if (preset === 'BIGEMINY') {
      if (!counterfactuals.regularizeRR) {
        if (beatIdx % 2 === 1) {
          localType = 'PVC';
          rr = baseRR * 1.38;
        } else {
          localType = 'NSR';
          rr = baseRR * 0.68;
        }
      } else {
        localType = beatIdx % 2 === 1 ? 'PVC' : 'NSR';
        rr = baseRR * (1 + (rng() - 0.5) * 0.015);
      }
    } else if (!counterfactuals.regularizeRR) {
      if (preset === 'AFIB') {
        const jitter = (rng() - 0.45) * 0.48 * baseRR;
        rr = Math.max(0.36, baseRR + jitter);
      } else if (preset === 'PVC') {
        if (beatIdx % 4 === 2) {
          localType = 'PVC';
          rr = baseRR * 1.42;
        } else if (beatIdx % 4 === 1) {
          localType = 'NSR';
          rr = baseRR * 0.66;
        } else {
          localType = 'NSR';
          rr = baseRR * (1 + (rng() - 0.5) * 0.04);
        }
      } else if (preset === 'PAC') {
        if (beatIdx % 4 === 2) {
          localType = 'PAC';
          rr = baseRR * 1.18;
        } else if (beatIdx % 4 === 1) {
          localType = 'NSR';
          rr = baseRR * 0.72;
        } else {
          localType = 'NSR';
          rr = baseRR * (1 + (rng() - 0.5) * 0.04);
        }
      } else if (preset === 'NSR' || preset === 'LBBB') {
        rr = baseRR * (1 + 0.035 * Math.sin(tCursor * 1.3) + (rng() - 0.5) * 0.025);
      } else if (preset === 'VTACH') {
        rr = baseRR * (1 + (rng() - 0.5) * 0.03);
      }
    } else {
      if (preset === 'PVC' && beatIdx % 4 === 2) localType = 'PVC';
      else if (preset === 'PAC' && beatIdx % 4 === 2) localType = 'PAC';
      else if (preset === 'BIGEMINY' && beatIdx % 2 === 1) localType = 'PVC';
      else if (preset === 'PAROXYSMAL_AF') localType = tCursor < 4.5 ? 'NSR' : 'AFIB';
      rr = baseRR * (1 + (rng() - 0.5) * 0.015);
    }

    beats.push({
      index: beatIdx + 1,
      timeSec: tCursor,
      sampleIdx: Math.round(tCursor * FS),
      type: localType,
      qrsWidthMs:
        (localType === 'PVC' || localType === 'VTACH' || localType === 'LBBB') && !counterfactuals.narrowQrs
          ? localType === 'VTACH'
            ? 156
            : localType === 'PVC'
            ? 148
            : 136
          : 86,
      prIntervalMs:
        (localType === 'AFIB' || localType === 'PVC' || localType === 'VTACH') && !counterfactuals.restoreP
          ? null
          : localType === 'PAC' && !counterfactuals.restoreP
          ? 112
          : 158
    });

    tCursor += rr;
    beatIdx++;
  }

  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    const t = i / FS;
    let y = 0;

    for (let b = 0; b < beats.length; b++) {
      const dt = t - beats[b].timeSec;
      if (Math.abs(dt) < 0.42) {
        y += renderBeatSample(dt, beats[b].type, counterfactuals, profile);
      }
    }

    const inAfibZone = preset === 'AFIB' || (preset === 'PAROXYSMAL_AF' && t >= 4.5);
    if (inAfibZone && !counterfactuals.restoreP) {
      y +=
        0.042 * Math.sin(2 * Math.PI * 5.8 * t + 0.4) +
        0.028 * Math.sin(2 * Math.PI * 8.1 * t + 1.9);
    }

    clean[i] = y;
  }

  const nm = profile.noiseMult;
  const motionAmp = (motionNoise / 100) * 0.42 * nm;
  const baselineAmp = (baselineWander / 100) * 0.36 * nm;
  const emgAmp = (emgNoise / 100) * 0.14 * nm;
  const motionCenter1 = 2.6 + rng() * 2.0;
  const motionCenter2 = 6.5 + rng() * 2.0;

  let noiseEnergy = 0;
  let signalEnergy = 0;

  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    const t = i / FS;
    let artifact = 0;

    artifact += baselineAmp * (0.68 * Math.sin(2 * Math.PI * 0.28 * t + 0.5) + 0.32 * Math.sin(2 * Math.PI * 0.11 * t));

    if (motionAmp > 0.005) {
      const burst1 = Math.exp(-Math.pow((t - motionCenter1) / 0.32, 2));
      const burst2 = Math.exp(-Math.pow((t - motionCenter2) / 0.25, 2));
      artifact += motionAmp * (burst1 * Math.sin(2 * Math.PI * 3.1 * t) + 0.8 * burst2 * Math.cos(2 * Math.PI * 2.4 * t));
    }

    artifact += emgAmp * ((rng() + rng() + rng() - 1.5) * 1.15);

    if (powerline60) {
      artifact += 0.095 * Math.sin(2 * Math.PI * 60 * t);
    }

    rawNoisy[i] = clean[i] + artifact;
    signalEnergy += clean[i] * clean[i];
    noiseEnergy += artifact * artifact;
  }

  const processed = denoise ? applyAdaptiveDenoiser(rawNoisy) : Float32Array.from(rawNoisy);

  let residualNoiseEnergy = 0;
  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    const err = processed[i] - clean[i];
    residualNoiseEnergy += err * err;
  }
  const activeNoiseEnergy = denoise ? Math.max(residualNoiseEnergy, 1e-6) : Math.max(noiseEnergy, 1e-6);
  const snrDb = Math.min(42, Math.max(1.5, 10 * Math.log10(Math.max(signalEnergy, 1e-4) / activeNoiseEnergy)));

  return {
    fs: FS,
    sensorProfile: profile,
    clean,
    raw: rawNoisy,
    signal: processed,
    beats,
    snrDb,
    isCustomUpload: false
  };
}

/**
 * Zero-phase baseline wander subtraction + 60 Hz notch-attenuating 5-tap symmetric filter.
 */
function applyAdaptiveDenoiser(input) {
  const n = input.length;
  const detrended = new Float32Array(n);
  const output = new Float32Array(n);

  const halfBase = 52;
  const prefix = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + input[i];

  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - halfBase);
    const hi = Math.min(n - 1, i + halfBase);
    const baseline = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1);
    detrended[i] = input[i] - baseline;
  }

  // 5-tap symmetric low-pass filter
  for (let i = 0; i < n; i++) {
    const s2 = detrended[Math.max(0, i - 2)];
    const s1 = detrended[Math.max(0, i - 1)];
    const s0 = detrended[i];
    const p1 = detrended[Math.min(n - 1, i + 1)];
    const p2 = detrended[Math.min(n - 1, i + 2)];
    output[i] = 0.1 * s2 + 0.2 * s1 + 0.4 * s0 + 0.2 * p1 + 0.1 * p2;
  }

  return output;
}

/**
 * Applies/removes adaptive denoiser on imported/custom records dynamically.
 */
function reprocessCustomRecord(baseRecord, { denoise = true, sensorProfile = 'WATCH_I' }) {
  const profile = SENSOR_PROFILES[sensorProfile] || SENSOR_PROFILES.WATCH_I;
  const raw = baseRecord.raw;
  const signal = denoise ? applyAdaptiveDenoiser(raw) : Float32Array.from(raw);
  let sigEnergy = 0;
  let diffEnergy = 0;
  for (let i = 2; i < signal.length; i++) {
    sigEnergy += signal[i] * signal[i];
    const d2 = signal[i] - 2 * signal[i - 1] + signal[i - 2];
    diffEnergy += d2 * d2;
  }
  const estSnr = Math.min(36, Math.max(8, 10 * Math.log10(Math.max(sigEnergy, 1e-4) / Math.max(diffEnergy * 0.25, 1e-6))));
  return {
    ...baseRecord,
    sensorProfile: profile,
    signal,
    snrDb: denoise ? Math.max(estSnr, 22.4) : Math.max(10.5, estSnr - 4.2)
  };
}

/**
 * Computes 0–75 Hz Power Spectral Density (dB) + band energy shares comparing Raw vs Filtered ECG.
 */
function computePowerSpectrum(processed, raw, fs = FS) {
  const N = 512;
  const start = Math.min(250, Math.max(0, processed.length - N));
  const freqs = [];
  const rawDb = [];
  const cleanDb = [];

  const hann = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    hann[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
  }

  let pBase = 0;
  let pQrs = 0;
  let pEmg = 0;
  let hum60Db = -65;

  const maxBin = Math.floor((75 / fs) * N);
  for (let k = 0; k <= maxBin; k++) {
    const freq = (k * fs) / N;
    freqs.push(freq);
    const omega = (2 * Math.PI * k) / N;

    let reRaw = 0;
    let imRaw = 0;
    let reCln = 0;
    let imCln = 0;

    for (let n = 0; n < N; n++) {
      const idx = start + n;
      const w = hann[n];
      const cosW = Math.cos(omega * n);
      const sinW = Math.sin(omega * n);
      const rVal = (raw[idx] || 0) * w;
      const cVal = (processed[idx] || 0) * w;

      reRaw += rVal * cosW;
      imRaw -= rVal * sinW;
      reCln += cVal * cosW;
      imCln -= cVal * sinW;
    }

    const magRaw = Math.sqrt(reRaw * reRaw + imRaw * imRaw) / N;
    const magCln = Math.sqrt(reCln * reCln + imCln * imCln) / N;
    const dbRaw = Math.max(-65, 20 * Math.log10(magRaw + 1e-5));
    const dbCln = Math.max(-65, 20 * Math.log10(magCln + 1e-5));
    rawDb.push(dbRaw);
    cleanDb.push(dbCln);

    const pwr = magCln * magCln;
    if (freq < 2.0) pBase += pwr;
    else if (freq >= 5 && freq <= 20) pQrs += pwr;
    else if (freq > 25) pEmg += pwr;
    if (Math.abs(freq - 60) < 1.0) hum60Db = Math.max(hum60Db, dbCln);
  }

  const tot = pBase + pQrs + pEmg || 1e-9;
  return {
    freqs,
    rawDb,
    cleanDb,
    bandShares: {
      baselinePct: Math.round((pBase / tot) * 100),
      qrsPct: Math.round((pQrs / tot) * 100),
      emgPct: Math.round((pEmg / tot) * 100),
      hum60Db: Number(hum60Db.toFixed(1))
    }
  };
}

/**
 * Real-time R-peak detector + time/frequency HRV & Morphology feature extractor.
 */
function analyzeSignalTelemetry(signal, groundTruthBeats = null, fs = FS) {
  const n = signal.length;

  const env = new Float32Array(n);
  let maxEnv = 0;
  for (let i = 2; i < n - 2; i++) {
    const d = signal[i + 1] - signal[i - 1];
    env[i] = d * d;
    if (env[i] > maxEnv) maxEnv = env[i];
  }

  const win = Math.max(4, Math.round(0.08 * fs));
  const mwi = new Float32Array(n);
  let running = 0;
  let maxMwi = 0;
  for (let i = 0; i < n; i++) {
    running += env[i];
    if (i >= win) running -= env[i - win];
    mwi[i] = running / win;
    if (mwi[i] > maxMwi) maxMwi = mwi[i];
  }

  const detectedBeats = [];
  if (groundTruthBeats && groundTruthBeats.length > 0) {
    for (const b of groundTruthBeats) {
      detectedBeats.push({ ...b });
    }
  } else {
    const threshold = Math.max(0.02, maxMwi * 0.24);
    const refractory = Math.round(0.28 * fs);
    let lastPeak = -refractory;

    for (let i = 4; i < n - 4; i++) {
      if (mwi[i] > threshold && i - lastPeak > refractory) {
        let bestIdx = i;
        let bestVal = Math.abs(signal[i]);
        const lo = Math.max(0, i - 20);
        const hi = Math.min(n - 1, i + 20);
        for (let k = lo; k <= hi; k++) {
          if (Math.abs(signal[k]) > bestVal) {
            bestVal = Math.abs(signal[k]);
            bestIdx = k;
          }
        }
        const qrsThresh = bestVal * 0.14;
        let qStart = bestIdx;
        let qEnd = bestIdx;
        while (qStart > Math.max(0, bestIdx - 28) && Math.abs(signal[qStart]) > qrsThresh) qStart--;
        while (qEnd < Math.min(n - 1, bestIdx + 28) && Math.abs(signal[qEnd]) > qrsThresh) qEnd++;
        const estQrsMs = Math.max(72, Math.min(180, Math.round(((qEnd - qStart + 8) / fs) * 1000)));

        detectedBeats.push({
          index: detectedBeats.length + 1,
          sampleIdx: bestIdx,
          timeSec: bestIdx / fs,
          type: estQrsMs >= 120 ? 'PVC' : 'DETECTED',
          qrsWidthMs: estQrsMs,
          prIntervalMs: estQrsMs >= 120 ? null : 155
        });
        lastPeak = bestIdx;
        i = bestIdx + Math.round(0.2 * fs);
      }
    }
  }

  const rrIntervals = [];
  for (let i = 1; i < detectedBeats.length; i++) {
    const rrMs = ((detectedBeats[i].sampleIdx - detectedBeats[i - 1].sampleIdx) / fs) * 1000;
    rrIntervals.push(rrMs);
    detectedBeats[i].rrMs = Math.round(rrMs);
  }
  if (detectedBeats.length > 0) {
    detectedBeats[0].rrMs = rrIntervals.length > 0 ? Math.round(rrIntervals[0]) : 833;
  }

  const meanRR =
    rrIntervals.length > 0 ? rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length : 833;
  const heartRate = Math.round(60000 / Math.max(250, meanRR));

  let sdnn = 0;
  let rmssd = 0;
  let nn50 = 0;
  if (rrIntervals.length > 1) {
    const variance =
      rrIntervals.reduce((acc, r) => acc + Math.pow(r - meanRR, 2), 0) / rrIntervals.length;
    sdnn = Math.round(Math.sqrt(variance));

    let diffSq = 0;
    for (let i = 1; i < rrIntervals.length; i++) {
      const diff = Math.abs(rrIntervals[i] - rrIntervals[i - 1]);
      if (diff > 50) nn50++;
      diffSq += diff * diff;
    }
    rmssd = Math.round(Math.sqrt(diffSq / (rrIntervals.length - 1)));
  }

  const pnn50 = rrIntervals.length > 1 ? Math.round((nn50 / (rrIntervals.length - 1)) * 100) : 0;
  // Poincaré SD1 and SD2 (ms)
  const sd1 = Math.round(rmssd / Math.SQRT2);
  const sd2 = Math.round(Math.sqrt(Math.max(0, 2 * sdnn * sdnn - 0.5 * rmssd * rmssd)));

  // Frequency-domain HRV approximation via RR tachogram spectral split
  let lfPower = 35;
  let hfPower = 45;
  if (rrIntervals.length >= 3) {
    const rapidVar = rmssd * rmssd;
    const slowVar = Math.max(10, sdnn * sdnn - 0.35 * rapidVar);
    hfPower = Math.max(12, Math.round(rapidVar * 0.65));
    lfPower = Math.max(15, Math.round(slowVar * 0.72));
  }
  const vlfPower = Math.max(8, Math.round((lfPower + hfPower) * 0.22));
  const lfHfRatio = Number((lfPower / Math.max(1, hfPower)).toFixed(2));

  const meanQrs =
    detectedBeats.length > 0
      ? Math.round(detectedBeats.reduce((a, b) => a + b.qrsWidthMs, 0) / detectedBeats.length)
      : 90;

  const validPrs = detectedBeats.filter((b) => b.prIntervalMs != null);
  const meanPr =
    validPrs.length > 0
      ? Math.round(validPrs.reduce((a, b) => a + b.prIntervalMs, 0) / validPrs.length)
      : null;

  let hfNoise = 0;
  for (let i = 2; i < n; i++) {
    hfNoise += Math.abs(signal[i] - 2 * signal[i - 1] + signal[i - 2]);
  }
  const avgHf = hfNoise / n;
  const sqi = Math.max(42, Math.min(99, Math.round(100 - avgHf * 260)));

  const ectopicCount = detectedBeats.filter((b) => b.type === 'PVC' || b.type === 'PAC' || b.type === 'VTACH').length;

  return {
    beats: detectedBeats,
    rrIntervals,
    meanRR: Math.round(meanRR),
    heartRate,
    sdnn,
    rmssd,
    pnn50,
    sd1,
    sd2,
    vlfPower,
    lfPower,
    hfPower,
    lfHfRatio,
    meanQrs,
    meanPr,
    sqi,
    ectopicCount,
    irregularityRatio: meanRR > 0 ? sdnn / meanRR : 0
  };
}

/**
 * Generates a 60-second (6 x 10s epochs) wearable Holter monitoring session around the active preset.
 */
function generateHolterSession(activePreset = 'NSR', baseBpm = 72) {
  const sequenceMap = {
    NSR: ['NSR', 'NSR', 'PAC', 'NSR', 'NSR', 'NSR'],
    AFIB: ['PAROXYSMAL_AF', 'AFIB', 'AFIB', 'AFIB', 'AFIB', 'PAC'],
    PVC: ['NSR', 'PVC', 'BIGEMINY', 'PVC', 'NSR', 'PVC'],
    PAC: ['NSR', 'PAC', 'PAC', 'NSR', 'PAC', 'NSR'],
    VTACH: ['PVC', 'BIGEMINY', 'VTACH', 'VTACH', 'PVC', 'NSR'],
    LBBB: ['LBBB', 'LBBB', 'LBBB', 'LBBB', 'LBBB', 'LBBB'],
    BIGEMINY: ['NSR', 'PVC', 'BIGEMINY', 'BIGEMINY', 'PVC', 'NSR'],
    PAROXYSMAL_AF: ['NSR', 'PAROXYSMAL_AF', 'AFIB', 'AFIB', 'PAC', 'NSR']
  };

  const seq = sequenceMap[activePreset] || sequenceMap.NSR;
  return seq.map((epPreset, idx) => {
    const info = RHYTHM_PRESETS[epPreset] || RHYTHM_PRESETS.NSR;
    const epBpm = epPreset === activePreset ? baseBpm : info.defaultBpm;
    const rec = generateSyntheticECG({
      preset: epPreset,
      bpm: epBpm,
      motionNoise: 10 + (idx % 3) * 6,
      baselineWander: 14,
      emgNoise: 8,
      denoise: true,
      seed: 200 + idx * 37
    });
    const tel = analyzeSignalTelemetry(rec.signal, rec.beats, FS);
    return {
      epochIndex: idx,
      label: `${idx * 10}–${(idx + 1) * 10} s`,
      preset: epPreset,
      shortName: epPreset === 'PAROXYSMAL_AF' ? 'PAF Onset' : epPreset === 'BIGEMINY' ? 'Bigeminy' : epPreset,
      bpm: tel.heartRate,
      sdnn: tel.sdnn,
      ectopicCount: tel.ectopicCount,
      severity: info.severity,
      seed: 200 + idx * 37
    };
  });
}

/**
 * Generates an anonymized sample CSV string from any preset or sample case.
 */
function generateSampleCSVText(preset = 'PVC', opts = {}) {
  const rec = generateSyntheticECG({
    preset,
    bpm: opts.bpm || RHYTHM_PRESETS[preset]?.defaultBpm || 74,
    motionNoise: opts.motion ?? 5,
    baselineWander: opts.baseline ?? 8,
    emgNoise: opts.emg ?? 4,
    denoise: false,
    seed: opts.seed || 777
  });
  const lines = ['time_sec,lead_i_mv'];
  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    lines.push(`${(i / FS).toFixed(4)},${rec.raw[i].toFixed(5)}`);
  }
  return lines.join('\n');
}

/**
 * Parses uploaded CSV or JSON single-lead ECG traces into a normalized 2500-sample record.
 */
function parseUploadedECG(text, { denoise = true, sensorProfile = 'WATCH_I', label = 'Imported CSV/JSON' } = {}) {
  let samples = [];
  const trimmed = text.trim();

  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    const parsed = JSON.parse(trimmed);
    const arr = Array.isArray(parsed) ? parsed : parsed.signal || parsed.ecg || parsed.samples || [];
    samples = arr.map(Number).filter((v) => Number.isFinite(v));
  } else {
    const lines = trimmed.split(/\r?\n/);
    for (const line of lines) {
      if (!line.trim()) continue;
      const parts = line.split(/[,\t;\s]+/).filter(Boolean);
      const val = Number(parts[parts.length - 1]);
      if (Number.isFinite(val)) samples.push(val);
    }
  }

  if (samples.length < 250) {
    throw new Error('Uploaded file must contain at least 250 numeric ECG samples (1 second at 250 Hz).');
  }

  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const raw = new Float32Array(WINDOW_SAMPLES);
  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    raw[i] = samples[i % samples.length] - mean;
  }

  return reprocessCustomRecord(
    {
      fs: FS,
      sensorProfile: SENSOR_PROFILES[sensorProfile] || SENSOR_PROFILES.WATCH_I,
      clean: raw,
      raw,
      signal: raw,
      beats: null,
      snrDb: 24.0,
      isCustomUpload: true,
      sourceLabel: label
    },
    { denoise, sensorProfile }
  );
}

  Object.assign(ns, {
    FS,
    WINDOW_SECONDS,
    WINDOW_SAMPLES,
    SENSOR_PROFILES,
    ACTIVITY_PROFILES,
    RHYTHM_PRESETS,
    SAMPLE_CASES,
    renderBeatSample,
    getBeatFiducials,
    generateSinusTemplateBeat,
    generateSyntheticECG,
    applyAdaptiveDenoiser,
    reprocessCustomRecord,
    computePowerSpectrum,
    analyzeSignalTelemetry,
    generateHolterSession,
    generateSampleCSVText,
    parseUploadedECG
  });
})(window);
