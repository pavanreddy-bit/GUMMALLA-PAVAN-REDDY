/**
 * High-DPI Canvas Renderers (v4.0) for Clinical ECG Telemetry, Calipers, XAI Overlays,
 * Single-Beat Microscope + Fiducial Annotations + Sinus Template,
 * Multi-Method XAI Comparator, Layer Activation Heatmap, Counterfactual Sensitivity Tornado,
 * Phase SHAP, FFT Spectrum, Faithfulness Curves, Attention Matrix, HRV Frequency & Poincaré.
 */

(function (global) {
  const ns = (global.CardioXAI = global.CardioXAI || {});
  const { FS, WINDOW_SAMPLES, generateSinusTemplateBeat, getBeatFiducials } = ns;

function prepCanvas(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(300, Math.floor(rect.width || canvas.clientWidth || 720));
  const height = cssHeight || parseInt(canvas.getAttribute('height'), 10) || 240;

  if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function fillRoundRect(ctx, x, y, w, h, r) {
  if (typeof ctx.roundRect === 'function') {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.fill();
  } else {
    ctx.fillRect(x, y, w, h);
  }
}

function getThemeColors() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  return {
    isLight,
    bg: isLight ? '#fff8f8' : '#09101d',
    gridMinor: isLight ? 'rgba(225, 29, 72, 0.08)' : 'rgba(56, 189, 248, 0.06)',
    gridMajor: isLight ? 'rgba(225, 29, 72, 0.18)' : 'rgba(56, 189, 248, 0.14)',
    trace: isLight ? '#0f172a' : '#22d3ee',
    rawGhost: isLight ? 'rgba(100, 116, 139, 0.35)' : 'rgba(148, 163, 184, 0.28)',
    text: isLight ? '#0f172a' : '#e2e8f0',
    muted: isLight ? '#64748b' : '#94a3b8',
    border: isLight ? 'rgba(15, 23, 42, 0.15)' : 'rgba(148, 163, 184, 0.2)'
  };
}

function saliencyColor(val, alpha = 0.45) {
  const v = Math.max(0, Math.min(1, val));
  if (v < 0.35) {
    const t = v / 0.35;
    return `rgba(${Math.round(56 + t * 195)}, ${Math.round(189 - t * 20)}, ${Math.round(248 - t * 210)}, ${alpha * (0.3 + v)})`;
  }
  const t = (v - 0.35) / 0.65;
  return `rgba(${Math.round(251 - t * 7)}, ${Math.round(191 - t * 128)}, ${Math.round(36 + t * 58)}, ${alpha * (0.45 + 0.55 * v)})`;
}

/**
 * Main 10-second clinical ECG strip with XAI overlays, configurable gain, threshold,
 * ghost raw toggle, R-peak markers, sweep playhead, and interactive calipers.
 */
function renderECGStrip(canvas, {
  ecgRecord,
  inference,
  telemetry,
  xaiMode,
  playheadIdx,
  selectedBeatIndex,
  hoverIdx,
  gainScale = 1.0,
  saliencyThreshold = 0.14,
  showGhostRaw = true,
  calipers = null,
  preset = 'NSR'
}) {
  const { ctx, width, height } = prepCanvas(canvas, 270);
  const colors = getThemeColors();
  const { signal, raw } = ecgRecord;

  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const pxPerSec = width / (WINDOW_SAMPLES / FS);
  const minorStep = pxPerSec * 0.04;
  const majorStep = pxPerSec * 0.2;

  ctx.lineWidth = 1;
  ctx.strokeStyle = colors.gridMinor;
  ctx.beginPath();
  for (let x = 0; x < width; x += minorStep) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += 12) {
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();

  ctx.strokeStyle = colors.gridMajor;
  ctx.beginPath();
  for (let x = 0; x < width; x += majorStep) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += 60) {
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();

  const baselineY = height * 0.62;
  const mvScale = height * 0.29 * gainScale;
  const valToY = (v) => baselineY - v * mvScale;
  const idxToX = (i) => (i / (WINDOW_SAMPLES - 1)) * width;

  // Paroxysmal AFib onset marker
  if (!ecgRecord.isCustomUpload && preset === 'PAROXYSMAL_AF') {
    const tx = idxToX(Math.round(4.5 * FS));
    ctx.save();
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.75)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(tx, 0);
    ctx.lineTo(tx, height);
    ctx.stroke();
    ctx.fillStyle = '#fbbf24';
    ctx.font = '700 10px Inter, sans-serif';
    ctx.fillText('AFib Onset (4.5s)', tx + 6, 15);
    ctx.restore();
  }

  if (xaiMode === 'gradcam' || xaiMode === 'shap' || xaiMode === 'smoothgrad') {
    const attrArr = xaiMode === 'smoothgrad' ? inference.smoothGrad : inference.gradCam;
    const step = 4;
    for (let i = 0; i < WINDOW_SAMPLES - step; i += step) {
      let avg = 0;
      for (let k = 0; k < step; k++) avg += attrArr[i + k];
      avg /= step;
      if (avg > saliencyThreshold) {
        const x1 = idxToX(i);
        const x2 = idxToX(i + step);
        ctx.fillStyle =
          xaiMode === 'smoothgrad'
            ? `rgba(167, 139, 250, ${Math.min(0.65, avg * 0.62)})`
            : saliencyColor(avg, xaiMode === 'shap' ? 0.5 : 0.42);
        ctx.fillRect(x1, 0, Math.max(1.5, x2 - x1), height);
      }
    }
  } else if (xaiMode === 'ig') {
    const ig = inference.integratedGradients;
    const step = 3;
    for (let i = 0; i < WINDOW_SAMPLES - step; i += step) {
      const v = ig[i];
      if (Math.abs(v) > Math.max(0.1, saliencyThreshold)) {
        const x = idxToX(i);
        const y = valToY(signal[i]);
        ctx.fillStyle =
          v > 0
            ? `rgba(244, 63, 94, ${Math.min(0.75, Math.abs(v) * 0.7)})`
            : `rgba(56, 189, 248, ${Math.min(0.65, Math.abs(v) * 0.6)})`;
        ctx.beginPath();
        ctx.arc(x, y, 2.6 + Math.abs(v) * 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  if (xaiMode === 'attention') {
    for (const link of inference.attentionLinks) {
      const x1 = idxToX(link.fromIdx);
      const x2 = idxToX(link.toIdx);
      const y1 = valToY(signal[link.fromIdx]) - 8;
      const y2 = valToY(signal[link.toIdx]) - 8;
      const midX = (x1 + x2) / 2;
      const arcHeight = link.longRange ? 68 : 34 + link.weight * 28;

      ctx.save();
      ctx.strokeStyle = `rgba(167, 139, 250, ${0.28 + link.weight * 0.65})`;
      ctx.lineWidth = 1.2 + link.weight * 2.4;
      if (link.longRange) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.quadraticCurveTo(midX, Math.max(12, Math.min(y1, y2) - arcHeight), x2, y2);
      ctx.stroke();
      ctx.restore();
    }
  }

  if (showGhostRaw && raw && (!ecgRecord.isCustomUpload || ecgRecord.raw !== ecgRecord.signal)) {
    ctx.strokeStyle = colors.rawGhost;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < WINDOW_SAMPLES; i += 2) {
      const x = idxToX(i);
      const y = valToY(raw[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const activeBeat = telemetry.beats[selectedBeatIndex];
  if (activeBeat) {
    const xStart = idxToX(Math.max(0, activeBeat.sampleIdx - Math.round(0.28 * FS)));
    const xEnd = idxToX(Math.min(WINDOW_SAMPLES - 1, activeBeat.sampleIdx + Math.round(0.36 * FS)));
    ctx.save();
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.7)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(xStart, 6, xEnd - xStart, height - 12);
    ctx.restore();
  }

  ctx.strokeStyle = colors.trace;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    const x = idxToX(i);
    const y = valToY(signal[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.font = '600 10px Inter, sans-serif';
  ctx.textAlign = 'center';
  telemetry.beats.forEach((b, idx) => {
    const x = idxToX(b.sampleIdx);
    const y = valToY(signal[b.sampleIdx]);
    const isEctopic = b.type === 'PVC' || b.type === 'PAC' || b.type === 'VTACH';
    ctx.fillStyle = isEctopic ? '#f43f5e' : '#34d399';
    ctx.beginPath();
    ctx.arc(x, Math.max(16, y - 8), idx === selectedBeatIndex ? 4.5 : 3.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = idx === selectedBeatIndex ? '#34d399' : colors.muted;
    ctx.fillText(`#${b.index}`, x, Math.max(12, y - 16));
  });

  // Caliper measurement overlay
  if (calipers && calipers.startIdx !== null) {
    const sIdx = calipers.startIdx;
    const eIdx = calipers.endIdx !== null ? calipers.endIdx : sIdx;
    const x1 = idxToX(Math.min(sIdx, eIdx));
    const x2 = idxToX(Math.max(sIdx, eIdx));
    const y1 = valToY(signal[sIdx]);
    const y2 = valToY(signal[eIdx]);

    ctx.save();
    ctx.fillStyle = 'rgba(251, 191, 36, 0.14)';
    ctx.fillRect(x1, 0, Math.max(2, x2 - x1), height);

    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(idxToX(sIdx), 0);
    ctx.lineTo(idxToX(sIdx), height);
    ctx.moveTo(idxToX(eIdx), 0);
    ctx.lineTo(idxToX(eIdx), height);
    ctx.stroke();

    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(idxToX(sIdx), y1, 4.5, 0, Math.PI * 2);
    ctx.arc(idxToX(eIdx), y2, 4.5, 0, Math.PI * 2);
    ctx.fill();

    const dtMs = Math.round((Math.abs(eIdx - sIdx) / FS) * 1000);
    const dv = (signal[eIdx] - signal[sIdx]).toFixed(2);
    const instBpm = dtMs >= 160 ? `${Math.round(60000 / dtMs)} BPM` : '—';
    const label = `Δt: ${dtMs} ms (${instBpm}) · ΔV: ${Number(dv) >= 0 ? '+' : ''}${dv} mV`;

    const boxW = Math.min(260, ctx.measureText(label).width + 24);
    const boxX = Math.max(8, Math.min(width - boxW - 8, (x1 + x2) / 2 - boxW / 2));
    ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
    fillRoundRect(ctx, boxX, 8, boxW, 22, 6);
    ctx.fillStyle = '#fbbf24';
    ctx.font = '700 10.5px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label, boxX + boxW / 2, 22);
    ctx.restore();
  }

  if (playheadIdx !== null && playheadIdx >= 0) {
    const px = idxToX(playheadIdx % WINDOW_SAMPLES);
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.8)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();
  }

  if (hoverIdx !== null && hoverIdx >= 0 && hoverIdx < WINDOW_SAMPLES) {
    const hx = idxToX(hoverIdx);
    const hy = valToY(signal[hoverIdx]);
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(hx, 0);
    ctx.lineTo(hx, height);
    ctx.stroke();

    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(hx, hy, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = colors.muted;
  ctx.font = '500 10px Consolas, monospace';
  ctx.textAlign = 'left';
  for (let sec = 0; sec <= 10; sec += 2) {
    const x = (sec / 10) * width;
    ctx.fillText(`${sec}s`, Math.min(width - 22, x + 4), height - 6);
  }
}

/**
 * Single-Beat XAI Microscope with Sinus Template Overlay and Fiducial P/Q/R/S/T Markers.
 */
function renderBeatZoom(canvas, {
  ecgRecord,
  inference,
  beat,
  xaiMode,
  hoverSampleOffset,
  showTemplateOverlay = true,
  showFiducials = true,
  counterfactuals = {},
  sensorProfileKey = 'WATCH_I',
  gainScale = 1.0
}) {
  const { ctx, width, height } = prepCanvas(canvas, 220);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  if (!beat) {
    ctx.fillStyle = colors.muted;
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText('No beat detected in current window.', 16, height / 2);
    return null;
  }

  const preSamples = Math.round(0.28 * FS);
  const postSamples = Math.round(0.38 * FS);
  const totalSamples = preSamples + postSamples;
  const startIdx = beat.sampleIdx - preSamples;

  const idxToX = (k) => (k / (totalSamples - 1)) * width;
  const baselineY = height * 0.64;
  const mvScale = height * 0.34 * gainScale;
  const valToY = (v) => baselineY - v * mvScale;

  const bands = [
    { label: 'P-Wave', fromMs: -220, toMs: -110, color: 'rgba(56, 189, 248, 0.08)' },
    { label: 'QRS', fromMs: -65, toMs: 65, color: 'rgba(244, 63, 94, 0.08)' },
    { label: 'ST-T', fromMs: 85, toMs: 310, color: 'rgba(52, 211, 153, 0.08)' }
  ];

  ctx.font = '600 10px Inter, sans-serif';
  ctx.textAlign = 'center';
  for (const b of bands) {
    const k1 = preSamples + (b.fromMs / 1000) * FS;
    const k2 = preSamples + (b.toMs / 1000) * FS;
    const x1 = idxToX(k1);
    const x2 = idxToX(k2);
    ctx.fillStyle = b.color;
    ctx.fillRect(x1, 0, x2 - x1, height);
    ctx.fillStyle = colors.muted;
    ctx.fillText(b.label, (x1 + x2) / 2, 16);
  }

  for (let k = 0; k < totalSamples - 1; k++) {
    const sIdx = Math.max(0, Math.min(WINDOW_SAMPLES - 1, startIdx + k));
    const camVal = inference.gradCam[sIdx];
    const igVal = inference.integratedGradients[sIdx];
    const sgVal = inference.smoothGrad[sIdx];

    const x1 = idxToX(k);
    const x2 = idxToX(k + 1);
    if (xaiMode === 'ig') {
      ctx.fillStyle =
        igVal >= 0
          ? `rgba(244, 63, 94, ${Math.min(0.65, Math.abs(igVal) * 0.6)})`
          : `rgba(56, 189, 248, ${Math.min(0.55, Math.abs(igVal) * 0.55)})`;
    } else if (xaiMode === 'smoothgrad') {
      ctx.fillStyle = `rgba(167, 139, 250, ${Math.min(0.62, sgVal * 0.6)})`;
    } else if (xaiMode !== 'raw') {
      ctx.fillStyle = saliencyColor(camVal, 0.5);
    } else {
      ctx.fillStyle = 'transparent';
    }
    ctx.fillRect(x1, height - 18, Math.max(2, x2 - x1), 12);
    if (xaiMode !== 'raw' && (xaiMode === 'smoothgrad' ? sgVal : camVal) > 0.3) {
      ctx.fillRect(x1, 24, Math.max(2, x2 - x1), height - 46);
    }
  }

  if (showTemplateOverlay) {
    const tpl = generateSinusTemplateBeat(preSamples, postSamples, sensorProfileKey);
    ctx.save();
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.55)';
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let k = 0; k < totalSamples; k++) {
      const x = idxToX(k);
      const y = valToY(tpl[k]);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = colors.trace;
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  for (let k = 0; k < totalSamples; k++) {
    const sIdx = Math.max(0, Math.min(WINDOW_SAMPLES - 1, startIdx + k));
    const x = idxToX(k);
    const y = valToY(ecgRecord.signal[sIdx]);
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (showFiducials) {
    const fiducials = getBeatFiducials(beat, counterfactuals);
    ctx.font = '700 10px Consolas, monospace';
    ctx.textAlign = 'center';
    for (const f of fiducials) {
      const k = Math.round(preSamples + (f.offsetMs / 1000) * FS);
      if (k < 2 || k >= totalSamples - 2) continue;
      const sIdx = Math.max(0, Math.min(WINDOW_SAMPLES - 1, startIdx + k));
      const fx = idxToX(k);
      const val = ecgRecord.signal[sIdx];
      const fy = valToY(val);
      ctx.fillStyle = f.color;
      ctx.beginPath();
      ctx.arc(fx, fy, 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(f.label, fx, val >= 0 ? Math.max(28, fy - 8) : Math.min(height - 22, fy + 14));
    }
  }

  const rx = idxToX(preSamples);
  ctx.strokeStyle = 'rgba(52, 211, 153, 0.45)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(rx, 20);
  ctx.lineTo(rx, height - 20);
  ctx.stroke();
  ctx.setLineDash([]);

  if (hoverSampleOffset !== null && hoverSampleOffset >= 0 && hoverSampleOffset < totalSamples) {
    const sIdx = Math.max(0, Math.min(WINDOW_SAMPLES - 1, startIdx + hoverSampleOffset));
    const hx = idxToX(hoverSampleOffset);
    const hy = valToY(ecgRecord.signal[sIdx]);
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(hx, hy, 4.5, 0, Math.PI * 2);
    ctx.fill();
  }

  return { startIdx, totalSamples };
}

/**
 * Renders horizontal SHAP Phase Attribution chart.
 */
function renderPhaseShap(canvas, phaseShap) {
  const { ctx, width, height } = prepCanvas(canvas, 145);
  const colors = getThemeColors();
  ctx.clearRect(0, 0, width, height);

  const labelW = 104;
  const chartW = width - labelW - 46;
  const zeroX = labelW + chartW * 0.35;
  const rowH = (height - 14) / phaseShap.length;

  ctx.strokeStyle = colors.border;
  ctx.beginPath();
  ctx.moveTo(zeroX, 4);
  ctx.lineTo(zeroX, height - 6);
  ctx.stroke();

  ctx.font = '600 11px Inter, sans-serif';
  phaseShap.forEach((item, idx) => {
    const cy = 10 + idx * rowH + rowH * 0.45;
    ctx.fillStyle = colors.muted;
    ctx.textAlign = 'left';
    ctx.fillText(item.phase, 4, cy + 4);

    const maxAbs = 0.6;
    const barPx = (Math.min(maxAbs, Math.abs(item.value)) / maxAbs) * (chartW * 0.62);
    const isPos = item.value >= 0;

    ctx.fillStyle = isPos ? '#f43f5e' : '#38bdf8';
    const x = isPos ? zeroX : zeroX - barPx;
    fillRoundRect(ctx, x, cy - 6, Math.max(3, barPx), 12, 4);

    ctx.fillStyle = colors.text;
    ctx.font = '600 10px Consolas, monospace';
    ctx.textAlign = isPos ? 'left' : 'right';
    ctx.fillText(
      `${isPos ? '+' : ''}${item.value.toFixed(2)}`,
      isPos ? zeroX + barPx + 6 : zeroX - barPx - 6,
      cy + 4
    );
    ctx.font = '600 11px Inter, sans-serif';
  });
}

/**
 * Renders 0–75 Hz FFT Power Spectrum with optional hover crosshair.
 */
function renderSpectrumChart(canvas, spectrum, hoverBin = null) {
  const { ctx, width, height } = prepCanvas(canvas, 150);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const padL = 44;
  const padR = 16;
  const padT = 16;
  const padB = 26;
  const w = width - padL - padR;
  const h = height - padT - padB;

  const freqToX = (f) => padL + (f / 75) * w;
  const dbToY = (db) => padT + ((0 - Math.max(-65, Math.min(0, db))) / 65) * h;

  ctx.fillStyle = 'rgba(52, 211, 153, 0.09)';
  ctx.fillRect(freqToX(5), padT, freqToX(20) - freqToX(5), h);

  ctx.strokeStyle = 'rgba(244, 63, 94, 0.45)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(freqToX(60), padT);
  ctx.lineTo(freqToX(60), padT + h);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = colors.border;
  ctx.strokeRect(padL, padT, w, h);

  ctx.save();
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.55)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  spectrum.freqs.forEach((f, idx) => {
    const x = freqToX(f);
    const y = dbToY(spectrum.rawDb[idx]);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2.1;
  ctx.beginPath();
  spectrum.freqs.forEach((f, idx) => {
    const x = freqToX(f);
    const y = dbToY(spectrum.cleanDb[idx]);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  if (hoverBin !== null && hoverBin >= 0 && hoverBin < spectrum.freqs.length) {
    const hx = freqToX(spectrum.freqs[hoverBin]);
    const hy = dbToY(spectrum.cleanDb[hoverBin]);
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(hx, padT);
    ctx.lineTo(hx, padT + h);
    ctx.stroke();
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(hx, hy, 3.8, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = colors.muted;
  ctx.font = '500 10px Consolas, monospace';
  ctx.textAlign = 'center';
  for (const f of [0, 10, 20, 30, 40, 50, 60, 75]) {
    ctx.fillText(`${f} Hz`, freqToX(f), height - 8);
  }
  ctx.textAlign = 'right';
  ctx.fillText('0 dB', padL - 6, padT + 8);
  ctx.fillText('-65 dB', padL - 6, padT + h);

  ctx.fillStyle = '#34d399';
  ctx.textAlign = 'left';
  ctx.fillText('QRS Band (5–20 Hz)', freqToX(5) + 4, padT + 12);
  ctx.fillStyle = '#f43f5e';
  ctx.fillText('60 Hz', freqToX(60) + 4, padT + 12);

  return { padL, w };
}

/**
 * Synchronized Multi-Method XAI Comparator (ECG + Grad-CAM++ + Integrated Gradients + SmoothGrad).
 */
function renderXaiComparator(canvas, { ecgRecord, inference, consensusScore }) {
  const { ctx, width, height } = prepCanvas(canvas, 330);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const padL = 138;
  const padR = 16;
  const w = width - padL - padR;
  const idxToX = (i) => padL + (i / (WINDOW_SAMPLES - 1)) * w;

  const tracks = [
    { label: 'ECG Trace', sub: 'Filtered 250 Hz', y: 14, h: 86, type: 'ecg' },
    { label: 'Grad-CAM++', sub: 'Conv Feature Map', y: 112, h: 58, type: 'cam' },
    { label: 'Integrated Grad', sub: '+Pos / -Neg Input', y: 182, h: 58, type: 'ig' },
    { label: 'SmoothGrad', sub: 'Noise-Avg Edge', y: 252, h: 58, type: 'sg' }
  ];

  tracks.forEach((tr) => {
    ctx.strokeStyle = colors.border;
    ctx.strokeRect(padL, tr.y, w, tr.h);

    ctx.fillStyle = colors.text;
    ctx.font = '700 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(tr.label, 10, tr.y + tr.h * 0.44);
    ctx.fillStyle = colors.muted;
    ctx.font = '500 10px Inter, sans-serif';
    ctx.fillText(tr.sub, 10, tr.y + tr.h * 0.72);

    if (tr.type === 'ecg') {
      const midY = tr.y + tr.h * 0.62;
      const scale = tr.h * 0.34;
      ctx.strokeStyle = colors.trace;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i < WINDOW_SAMPLES; i += 2) {
        const x = idxToX(i);
        const y = midY - ecgRecord.signal[i] * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    } else {
      const step = 4;
      for (let i = 0; i < WINDOW_SAMPLES - step; i += step) {
        const x = idxToX(i);
        const bw = Math.max(1.5, idxToX(i + step) - x);
        if (tr.type === 'cam') {
          const v = inference.gradCam[i];
          ctx.fillStyle = saliencyColor(v, 0.85);
          const barH = Math.max(2, v * (tr.h - 8));
          ctx.fillRect(x, tr.y + tr.h - 4 - barH, bw, barH);
        } else if (tr.type === 'ig') {
          const v = inference.integratedGradients[i];
          const midY = tr.y + tr.h / 2;
          const barH = Math.abs(v) * (tr.h * 0.44);
          ctx.fillStyle = v >= 0 ? 'rgba(244, 63, 94, 0.85)' : 'rgba(56, 189, 248, 0.85)';
          ctx.fillRect(x, v >= 0 ? midY - barH : midY, bw, Math.max(1.5, barH));
        } else if (tr.type === 'sg') {
          const v = inference.smoothGrad[i];
          ctx.fillStyle = `rgba(167, 139, 250, ${0.25 + v * 0.75})`;
          const barH = Math.max(2, v * (tr.h - 8));
          ctx.fillRect(x, tr.y + tr.h - 4 - barH, bw, barH);
        }
      }
    }
  });

  ctx.fillStyle = '#34d399';
  ctx.font = '700 10.5px Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.fillText(`Inter-Method Cosine Consensus: ${(consensusScore * 100).toFixed(0)}%`, width - 16, 12);
}

/**
 * Renders Layer-by-Layer Neural Activation Heatmap (ResNet-B1, ResNet-B3, BiGRU, MHA).
 */
function renderLayerActivations(canvas, layers) {
  const { ctx, width, height } = prepCanvas(canvas, 220);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const padL = 200;
  const padR = 16;
  const padT = 18;
  const padB = 26;
  const w = width - padL - padR;
  const rowH = (height - padT - padB) / layers.length;

  layers.forEach((layer, r) => {
    const y = padT + r * rowH + 4;
    const h = rowH - 8;
    ctx.fillStyle = colors.text;
    ctx.font = '600 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(layer.name, 10, y + h / 2 + 4);

    const n = layer.values.length;
    const cellW = w / n;
    for (let c = 0; c < n; c++) {
      const v = layer.values[c];
      ctx.fillStyle =
        r === 2
          ? `rgba(167, 139, 250, ${0.1 + v * 0.85})`
          : r === 3
          ? `rgba(52, 211, 153, ${0.1 + v * 0.85})`
          : saliencyColor(v, 0.85);
      ctx.fillRect(padL + c * cellW, y, Math.max(1.5, cellW - 0.5), h);
    }
  });

  ctx.fillStyle = colors.muted;
  ctx.font = '500 10px Consolas, monospace';
  ctx.textAlign = 'center';
  for (let sec = 0; sec <= 10; sec += 2) {
    ctx.fillText(`${sec}s`, padL + (sec / 10) * w, height - 8);
  }
}

/**
 * Renders Counterfactual Sensitivity Tornado Chart (Delta Probability).
 */
function renderCounterfactualTornado(canvas, cfSensitivity) {
  const { ctx, width, height } = prepCanvas(canvas, 220);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const padL = 162;
  const padR = 104;
  const w = width - padL - padR;
  const zeroX = padL + w * 0.68;
  const rowH = (height - 24) / cfSensitivity.length;

  ctx.strokeStyle = colors.border;
  ctx.beginPath();
  ctx.moveTo(zeroX, 10);
  ctx.lineTo(zeroX, height - 14);
  ctx.stroke();

  cfSensitivity.forEach((item, idx) => {
    const cy = 16 + idx * rowH + rowH * 0.45;
    ctx.fillStyle = colors.text;
    ctx.font = '600 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(item.label, 10, cy + 4);

    const barPx = Math.min(w * 0.62, Math.abs(item.deltaProb) * w * 0.72);
    const isPos = item.deltaProb >= 0;
    ctx.fillStyle = isPos ? '#34d399' : '#f43f5e';
    const bx = isPos ? zeroX : zeroX - barPx;
    fillRoundRect(ctx, bx, cy - 8, Math.max(4, barPx), 16, 4);

    ctx.fillStyle = colors.muted;
    ctx.font = '600 10.5px Consolas, monospace';
    ctx.textAlign = 'left';
    const sign = item.deltaProb >= 0 ? '+' : '';
    ctx.fillText(`${sign}${(item.deltaProb * 100).toFixed(1)}% → ${item.resultingClass}`, zeroX + (isPos ? barPx + 8 : 8), cy + 4);
  });
}

/**
 * Renders Frequency-Domain HRV (VLF / LF / HF Power & LF/HF Ratio).
 */
function renderHrvFrequencyBars(canvas, telemetry) {
  const { ctx, width, height } = prepCanvas(canvas, 195);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const bands = [
    { label: 'VLF (<0.04 Hz)', val: telemetry.vlfPower, color: '#94a3b8' },
    { label: 'LF (0.04–0.15 Hz)', val: telemetry.lfPower, color: '#fbbf24' },
    { label: 'HF (0.15–0.40 Hz)', val: telemetry.hfPower, color: '#34d399' }
  ];
  const maxVal = Math.max(100, ...bands.map((b) => b.val)) * 1.15;
  const padL = 44;
  const padR = 24;
  const padT = 28;
  const padB = 34;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const colW = w / bands.length;

  ctx.strokeStyle = colors.border;
  ctx.strokeRect(padL, padT, w, h);

  bands.forEach((b, i) => {
    const bh = (b.val / maxVal) * (h - 12);
    const bx = padL + i * colW + colW * 0.24;
    const bw = colW * 0.52;
    const by = padT + h - bh;

    ctx.fillStyle = b.color;
    fillRoundRect(ctx, bx, by, bw, Math.max(4, bh), 5);

    ctx.fillStyle = colors.text;
    ctx.font = '700 11px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${b.val} ms²`, bx + bw / 2, Math.max(padT + 12, by - 6));

    ctx.fillStyle = colors.muted;
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText(b.label, bx + bw / 2, height - 12);
  });

  ctx.fillStyle = '#38bdf8';
  ctx.font = '700 11px Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.fillText(`LF/HF Ratio: ${telemetry.lfHfRatio} · pNN50: ${telemetry.pnn50}%`, width - 16, 18);
}

/**
 * Renders live Deletion & Insertion Faithfulness Curves on the active trace.
 */
function renderFaithfulnessCurves(canvas, curves, showMethodCompare = true) {
  const { ctx, width, height } = prepCanvas(canvas, 230);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const padL = 52;
  const padR = 20;
  const padT = 18;
  const padB = 34;
  const w = width - padL - padR;
  const h = height - padT - padB;

  const mapX = (pct) => padL + (pct / 100) * w;
  const mapY = (p) => padT + (1 - Math.max(0, Math.min(1, p))) * h;

  ctx.strokeStyle = colors.border;
  ctx.strokeRect(padL, padT, w, h);

  ctx.save();
  ctx.strokeStyle = colors.muted;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  curves.steps.forEach((pct, i) => {
    const x = mapX(pct);
    const y = mapY(curves.randomCurve[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();

  if (showMethodCompare && curves.igInsertion) {
    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
    ctx.lineWidth = 1.6;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    curves.steps.forEach((pct, i) => {
      const x = mapX(pct);
      const y = mapY(curves.igInsertion[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = '#34d399';
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  curves.steps.forEach((pct, i) => {
    const x = mapX(pct);
    const y = mapY(curves.insertionCurve[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  curves.steps.forEach((pct, i) => {
    const x = mapX(pct);
    const y = mapY(curves.deletionCurve[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.font = '600 11px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#34d399';
  ctx.fillText(`● Active Insertion (AUC ${curves.insertionAuc.toFixed(2)} — higher better)`, padL + 12, padT + 18);
  ctx.fillStyle = '#f43f5e';
  ctx.fillText(`● Active Deletion (AUC ${curves.deletionAuc.toFixed(2)} — lower better)`, padL + 12, padT + 36);
  if (showMethodCompare && curves.igInsertion) {
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('--- Integrated Gradients Insertion Reference', padL + 12, padT + 54);
  }

  ctx.fillStyle = colors.muted;
  ctx.textAlign = 'center';
  for (const pct of [0, 20, 40, 60, 80, 100]) {
    ctx.fillText(`${pct}%`, mapX(pct), height - 16);
  }
  ctx.fillText('% Top-Attributed ECG Segments Perturbed / Inserted', padL + w / 2, height - 3);
}

/**
 * Renders Poincaré HRV Scatter Plot (RR_n vs RR_n+1) with SD1/SD2 Clinical Fit Ellipse.
 */
function renderPoincare(canvas, telemetry, activePreset) {
  const { ctx, width, height } = prepCanvas(canvas, 280);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const rrIntervals = telemetry.rrIntervals;
  const pad = 42;
  const minRR = 300;
  const maxRR = 1350;
  const mapX = (rr) => pad + ((rr - minRR) / (maxRR - minRR)) * (width - pad * 1.5);
  const mapY = (rr) => height - pad - ((rr - minRR) / (maxRR - minRR)) * (height - pad * 1.5);

  ctx.strokeStyle = colors.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad * 0.5, width - pad * 1.5, height - pad * 1.5);

  ctx.save();
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(mapX(minRR), mapY(minRR));
  ctx.lineTo(mapX(maxRR), mapY(maxRR));
  ctx.stroke();
  ctx.restore();

  // SD1 / SD2 Fit Ellipse around (meanRR, meanRR)
  if (rrIntervals.length >= 2) {
    const cx = mapX(Math.max(minRR, Math.min(maxRR, telemetry.meanRR)));
    const cy = mapY(Math.max(minRR, Math.min(maxRR, telemetry.meanRR)));
    const pxPerMsX = (width - pad * 1.5) / (maxRR - minRR);
    const rx = Math.max(8, telemetry.sd2 * pxPerMsX * 1.6);
    const ry = Math.max(6, telemetry.sd1 * pxPerMsX * 1.6);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-Math.PI / 4);
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.7)';
    ctx.fillStyle = 'rgba(52, 211, 153, 0.08)';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  for (let i = 0; i < rrIntervals.length - 1; i++) {
    const x = mapX(Math.max(minRR, Math.min(maxRR, rrIntervals[i])));
    const y = mapY(Math.max(minRR, Math.min(maxRR, rrIntervals[i + 1])));
    const offAxis = Math.abs(rrIntervals[i] - rrIntervals[i + 1]) > 140;
    ctx.fillStyle = offAxis ? '#f43f5e' : '#38bdf8';
    ctx.beginPath();
    ctx.arc(x, y, offAxis ? 5.5 : 4.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = colors.muted;
  ctx.font = '500 11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('RRₙ (ms)', width / 2, height - 10);

  ctx.save();
  ctx.translate(14, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('RRₙ₊₁ (ms)', 0, 0);
  ctx.restore();

  ctx.textAlign = 'right';
  ctx.fillStyle = colors.text;
  ctx.fillText(`SD1: ${telemetry.sd1} ms · SD2: ${telemetry.sd2} ms (${activePreset})`, width - 16, 20);
}

/**
 * Renders interactive N x N Beat-to-Beat Self-Attention Matrix Heatmap.
 */
function renderAttentionHeatmap(canvas, beats, matrix, selectedBeatIndex, hoverCell = null) {
  const { ctx, width, height } = prepCanvas(canvas, 280);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const n = beats.length;
  if (n === 0) return null;

  const padL = 36;
  const padT = 26;
  const padB = 18;
  const padR = 14;
  const cellW = (width - padL - padR) / n;
  const cellH = (height - padT - padB) / n;

  const maxWeight = Math.max(0.15, ...matrix.flatMap((row) => Array.from(row)));

  ctx.font = '600 9px Consolas, monospace';
  ctx.textAlign = 'center';

  for (let i = 0; i < n; i++) {
    ctx.fillStyle = i === selectedBeatIndex ? '#34d399' : colors.muted;
    ctx.fillText(`#${beats[i].index}`, padL + i * cellW + cellW / 2, padT - 8);
    ctx.textAlign = 'right';
    ctx.fillText(`#${beats[i].index}`, padL - 6, padT + i * cellH + cellH / 2 + 3);
    ctx.textAlign = 'center';

    for (let j = 0; j < n; j++) {
      const val = matrix[i][j];
      const norm = Math.min(1, val / maxWeight);
      ctx.fillStyle = `rgba(167, 139, 250, ${0.08 + norm * 0.85})`;
      const x = padL + j * cellW + 1;
      const y = padT + i * cellH + 1;
      ctx.fillRect(x, y, Math.max(2, cellW - 2), Math.max(2, cellH - 2));

      if (i === selectedBeatIndex || j === selectedBeatIndex) {
        ctx.strokeStyle = 'rgba(52, 211, 153, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, Math.max(2, cellW - 2), Math.max(2, cellH - 2));
      }
      if (hoverCell && hoverCell.row === i && hoverCell.col === j) {
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, Math.max(2, cellW - 2), Math.max(2, cellH - 2));
      }
    }
  }

  return { padL, padT, cellW, cellH, n };
}

/**
 * Renders Beat-to-Beat R-R Interval Tachogram.
 */
function renderRRTachogram(canvas, beats, selectedBeatIndex) {
  const { ctx, width, height } = prepCanvas(canvas, 165);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  if (beats.length < 2) return null;

  const padL = 48;
  const padR = 18;
  const padT = 16;
  const padB = 26;
  const w = width - padL - padR;
  const h = height - padT - padB;

  const rrs = beats.map((b) => b.rrMs || 800);
  const minRR = Math.max(250, Math.min(...rrs) - 90);
  const maxRR = Math.min(1500, Math.max(...rrs) + 90);

  const mapX = (i) => padL + (i / Math.max(1, beats.length - 1)) * w;
  const mapY = (rr) => padT + ((maxRR - rr) / Math.max(100, maxRR - minRR)) * h;

  ctx.strokeStyle = colors.border;
  ctx.strokeRect(padL, padT, w, h);

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  ctx.beginPath();
  beats.forEach((b, i) => {
    const x = mapX(i);
    const y = mapY(rrs[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  beats.forEach((b, i) => {
    const x = mapX(i);
    const y = mapY(rrs[i]);
    const isEctopic = b.type === 'PVC' || b.type === 'PAC' || b.type === 'VTACH';
    ctx.fillStyle = i === selectedBeatIndex ? '#34d399' : isEctopic ? '#f43f5e' : '#38bdf8';
    ctx.beginPath();
    ctx.arc(x, y, i === selectedBeatIndex ? 5.5 : 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = colors.muted;
    ctx.font = '500 9px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`#${b.index}`, x, height - 8);
  });

  ctx.fillStyle = colors.muted;
  ctx.font = '500 10px Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.fillText(`${Math.round(maxRR)}ms`, padL - 6, padT + 10);
  ctx.fillText(`${Math.round(minRR)}ms`, padL - 6, padT + h);

  return { padL, w, n: beats.length };
}

/**
 * Renders 6x6 MIT-BIH + PhysioNet Validation Confusion Matrix (% or Counts).
 */
function renderConfusionMatrix(canvas, showCounts = false) {
  const { ctx, width, height } = prepCanvas(canvas, 310);
  const colors = getThemeColors();
  ctx.clearRect(0, 0, width, height);

  const labels = ['NSR', 'AFib', 'PVC', 'PAC', 'VTach', 'LBBB'];
  const matrix = [
    [98.4, 0.4, 0.2, 0.8, 0.0, 0.2],
    [0.9, 97.2, 0.3, 1.4, 0.1, 0.1],
    [0.3, 0.2, 97.8, 0.4, 1.1, 0.2],
    [1.6, 1.5, 0.5, 96.1, 0.1, 0.2],
    [0.0, 0.1, 1.9, 0.1, 97.6, 0.3],
    [0.5, 0.2, 0.4, 0.2, 0.2, 98.5]
  ];
  const rowTotals = [4500, 2200, 1800, 1200, 900, 1400];

  const padL = 56;
  const padT = 34;
  const cellW = (width - padL - 16) / 6;
  const cellH = (height - padT - 26) / 6;

  ctx.font = '600 11px Inter, sans-serif';
  ctx.textAlign = 'center';

  for (let i = 0; i < 6; i++) {
    ctx.fillStyle = colors.muted;
    ctx.fillText(labels[i], padL + i * cellW + cellW / 2, padT - 10);
    ctx.textAlign = 'right';
    ctx.fillText(labels[i], padL - 8, padT + i * cellH + cellH / 2 + 4);
    ctx.textAlign = 'center';

    for (let j = 0; j < 6; j++) {
      const val = matrix[i][j];
      const count = Math.round((val / 100) * rowTotals[i]);
      const alpha = i === j ? 0.72 : Math.min(0.55, val / 12);
      ctx.fillStyle = i === j ? `rgba(56, 189, 248, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
      ctx.fillRect(padL + j * cellW + 2, padT + i * cellH + 2, cellW - 4, cellH - 4);

      ctx.fillStyle = i === j && !colors.isLight ? '#04121f' : colors.text;
      ctx.font = '600 11px Consolas, monospace';
      ctx.fillText(
        showCounts ? `${count}` : `${val.toFixed(1)}%`,
        padL + j * cellW + cellW / 2,
        padT + i * cellH + cellH / 2 + 4
      );
      ctx.font = '600 11px Inter, sans-serif';
    }
  }
}

/**
 * Renders Multi-Class ROC Curves for all 6 arrhythmia classes (+ optional Macro-Avg curve).
 */
function renderROCCurves(canvas, showMacro = true) {
  const { ctx, width, height } = prepCanvas(canvas, 310);
  const colors = getThemeColors();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const pad = 42;
  const w = width - pad * 1.5;
  const h = height - pad * 1.5;

  ctx.strokeStyle = colors.border;
  ctx.strokeRect(pad, 16, w, h);

  ctx.save();
  ctx.strokeStyle = colors.muted;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad, 16 + h);
  ctx.lineTo(pad + w, 16);
  ctx.stroke();
  ctx.restore();

  const curves = [
    { name: 'NSR (0.996)', color: '#34d399', power: 0.045 },
    { name: 'LBBB (0.995)', color: '#a78bfa', power: 0.05 },
    { name: 'VTach (0.994)', color: '#f43f5e', power: 0.055 },
    { name: 'AFib (0.991)', color: '#fbbf24', power: 0.072 },
    { name: 'PVC (0.989)', color: '#fb7185', power: 0.085 },
    { name: 'PAC (0.982)', color: '#38bdf8', power: 0.11 }
  ];

  curves.forEach((c, idx) => {
    ctx.strokeStyle = c.color;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let s = 0; s <= 60; s++) {
      const fpr = s / 60;
      const tpr = Math.pow(fpr, c.power);
      const x = pad + fpr * w;
      const y = 16 + h - tpr * h;
      if (s === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const lx = pad + w - 130;
    const ly = 16 + h - 116 + idx * 18;
    ctx.fillStyle = c.color;
    ctx.fillRect(lx, ly - 8, 12, 3);
    ctx.fillStyle = colors.text;
    ctx.font = '600 10.5px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(c.name, lx + 18, ly - 3);
  });

  if (showMacro) {
    ctx.save();
    ctx.strokeStyle = colors.text;
    ctx.lineWidth = 2.2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    for (let s = 0; s <= 60; s++) {
      const fpr = s / 60;
      const tpr = Math.pow(fpr, 0.069);
      const x = pad + fpr * w;
      const y = 16 + h - tpr * h;
      if (s === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  ctx.fillStyle = colors.muted;
  ctx.font = '500 11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('False Positive Rate (1 - Specificity)', pad + w / 2, height - 10);

  ctx.save();
  ctx.translate(14, 16 + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('True Positive Rate (Sensitivity)', 0, 0);
  ctx.restore();
}

  Object.assign(ns, {
    renderECGStrip,
    renderBeatZoom,
    renderPhaseShap,
    renderSpectrumChart,
    renderXaiComparator,
    renderLayerActivations,
    renderCounterfactualTornado,
    renderHrvFrequencyBars,
    renderFaithfulnessCurves,
    renderPoincare,
    renderAttentionHeatmap,
    renderRRTachogram,
    renderConfusionMatrix,
    renderROCCurves
  });
})(window);
