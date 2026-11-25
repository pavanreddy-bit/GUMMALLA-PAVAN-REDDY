(function (global) {
  const {
    FS,
    WINDOW_SAMPLES,
    SENSOR_PROFILES,
    ACTIVITY_PROFILES,
    RHYTHM_PRESETS,
    SAMPLE_CASES,
    generateSyntheticECG,
    reprocessCustomRecord,
    analyzeSignalTelemetry,
    computePowerSpectrum,
    generateHolterSession,
    generateSampleCSVText,
    parseUploadedECG,
    runExplainableInference,
    HARDWARE_TARGETS,
    PIPELINE_STAGES,
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
  } = global.CardioXAI;

const state = {
  sensorProfile: 'WATCH_I',
  activityProfile: 'RESTING',
  preset: 'NSR',
  bpm: 72,
  seed: 101,
  motionNoise: 10,
  baselineWander: 15,
  emgNoise: 8,
  powerline60: false,
  denoise: true,
  showGhostRaw: true,
  counterfactuals: {
    restoreP: false,
    regularizeRR: false,
    narrowQrs: false
  },
  xaiMode: 'gradcam',
  saliencyThreshold: 0.14,
  gainScale: 1.0,
  sweepSpeed: 1.0,
  showTemplateOverlay: true,
  showFiducials: true,
  caliperMode: false,
  calipers: { startIdx: null, endIdx: null, dragging: false },
  audioEnabled: false,
  audioCtx: null,
  lastBeepBeatIdx: -1,
  isPlaying: true,
  playheadIdx: 0,
  selectedBeatIndex: 2,
  stripHoverIdx: null,
  zoomHoverOffset: null,
  spectrumHoverBin: null,
  spectrumBounds: null,
  zoomBounds: null,
  matrixBounds: null,
  matrixHoverCell: null,
  beatFilter: 'ALL',
  beatSort: 'TIME',
  holterEpochs: [],
  activeEpochIdx: 0,
  selectedStageId: 'stage-1',
  cmShowCounts: false,
  rocShowMacro: true,
  selectedHardware: 'CORTEX_M55_INT8',
  customBaseRecord: null,
  ecgRecord: null,
  spectrum: null,
  telemetry: null,
  inference: null,
  alertDismissed: false
};

const els = {
  sensorPills: document.querySelectorAll('.sensor-pill'),
  activityPills: document.querySelectorAll('.activity-pill'),
  sensorBadge: document.getElementById('sensor-badge'),
  sensorDesc: document.getElementById('sensor-desc'),
  sourceModePill: document.getElementById('source-mode-pill'),
  presetBtns: document.querySelectorAll('.preset-btn'),
  xaiPills: document.querySelectorAll('.xai-pill'),
  navTabs: document.querySelectorAll('.nav-tab'),
  tabPanels: document.querySelectorAll('.tab-panel'),
  sliderBpm: document.getElementById('slider-bpm'),
  valBpm: document.getElementById('val-bpm'),
  sliderMotion: document.getElementById('slider-motion'),
  valMotion: document.getElementById('val-motion'),
  sliderBaseline: document.getElementById('slider-baseline'),
  valBaseline: document.getElementById('val-baseline'),
  sliderEmg: document.getElementById('slider-emg'),
  valEmg: document.getElementById('val-emg'),
  chkPowerline: document.getElementById('chk-powerline'),
  chkDenoise: document.getElementById('chk-denoise'),
  chkGhostRaw: document.getElementById('chk-ghost-raw'),
  cfRestoreP: document.getElementById('cf-restore-p'),
  cfRegularizeRR: document.getElementById('cf-regularize-rr'),
  cfNarrowQrs: document.getElementById('cf-narrow-qrs'),
  btnResetCf: document.getElementById('btn-reset-cf'),
  btnApplySinusReset: document.getElementById('btn-apply-sinus-reset'),
  selectGain: document.getElementById('select-gain'),
  selectSpeed: document.getElementById('select-speed'),
  sliderSalThresh: document.getElementById('slider-sal-thresh'),
  valSalThresh: document.getElementById('val-sal-thresh'),
  btnCalipers: document.getElementById('btn-calipers'),
  btnClearCalipers: document.getElementById('btn-clear-calipers'),
  stripHelperText: document.getElementById('strip-helper-text'),
  chkTemplateOverlay: document.getElementById('chk-template-overlay'),
  chkFiducials: document.getElementById('chk-fiducials'),
  btnPlayPause: document.getElementById('btn-play-pause'),
  btnPrevBeat: document.getElementById('btn-prev-beat'),
  btnNextBeat: document.getElementById('btn-next-beat'),
  btnTheme: document.getElementById('btn-theme'),
  btnAudioToggle: document.getElementById('btn-audio-toggle'),
  selectSampleCase: document.getElementById('select-sample-case'),
  btnLoadSample: document.getElementById('btn-load-sample'),
  btnUploadTrigger: document.getElementById('btn-upload-trigger'),
  fileUpload: document.getElementById('file-upload'),
  btnDownloadSample: document.getElementById('btn-download-sample'),
  btnExportReport: document.getElementById('btn-export-report'),
  reportModal: document.getElementById('report-modal'),
  modalBackdrop: document.getElementById('modal-backdrop'),
  btnCloseModal: document.getElementById('btn-close-modal'),
  btnPrintReport: document.getElementById('btn-print-report'),
  btnExportJson: document.getElementById('btn-export-json'),
  reportBody: document.getElementById('report-body'),
  reportTimestamp: document.getElementById('report-timestamp'),
  alertBanner: document.getElementById('alert-banner'),
  alertTitle: document.getElementById('alert-title'),
  alertMessage: document.getElementById('alert-message'),
  btnDismissAlert: document.getElementById('btn-dismiss-alert'),
  snrBadge: document.getElementById('snr-badge'),
  kpiClass: document.getElementById('kpi-class'),
  kpiConf: document.getElementById('kpi-conf'),
  kpiHr: document.getElementById('kpi-hr'),
  kpiRr: document.getElementById('kpi-rr'),
  kpiHrv: document.getElementById('kpi-hrv'),
  kpiIrregularity: document.getElementById('kpi-irregularity'),
  kpiMorph: document.getElementById('kpi-morph'),
  kpiSqi: document.getElementById('kpi-sqi'),
  beatSelectorLabel: document.getElementById('beat-selector-label'),
  probBars: document.getElementById('prob-bars'),
  clinicalRationale: document.getElementById('clinical-rationale'),
  triageBadge: document.getElementById('triage-badge'),
  triageSteps: document.getElementById('triage-steps'),
  faithfulnessCurvePill: document.getElementById('faithfulness-curve-pill'),
  barEpistemic: document.getElementById('bar-epistemic'),
  valEpistemic: document.getElementById('val-epistemic'),
  barAleatoric: document.getElementById('bar-aleatoric'),
  valAleatoric: document.getElementById('val-aleatoric'),
  metricDeletion: document.getElementById('metric-deletion'),
  metricInsertion: document.getElementById('metric-insertion'),
  metricFaithfulness: document.getElementById('metric-faithfulness'),
  bandBasePill: document.getElementById('band-base-pill'),
  bandQrsPill: document.getElementById('band-qrs-pill'),
  bandEmgPill: document.getElementById('band-emg-pill'),
  spectrum60HzPill: document.getElementById('spectrum-60hz-pill'),
  consensusBadge: document.getElementById('consensus-badge'),
  holterTimeline: document.getElementById('holter-timeline'),
  holterActivePill: document.getElementById('holter-active-pill'),
  poincarePattern: document.getElementById('poincare-pattern'),
  matrixHoverInfo: document.getElementById('matrix-hover-info'),
  tachogramPill: document.getElementById('tachogram-pill'),
  hrvFreqPill: document.getElementById('hrv-freq-pill'),
  selectBeatFilter: document.getElementById('select-beat-filter'),
  selectBeatSort: document.getElementById('select-beat-sort'),
  beatCountPill: document.getElementById('beat-count-pill'),
  beatTableBody: document.getElementById('beat-table-body'),
  pipeNodes: document.querySelectorAll('.pipe-node'),
  stageInspector: document.getElementById('stage-inspector'),
  btnCmMode: document.getElementById('btn-cm-mode'),
  chkRocMacro: document.getElementById('chk-roc-macro'),
  selectHardware: document.getElementById('select-hardware'),
  hardwareProfileStrip: document.getElementById('hardware-profile-strip'),
  ecgStripCanvas: document.getElementById('ecg-strip-canvas'),
  beatZoomCanvas: document.getElementById('beat-zoom-canvas'),
  shapCanvas: document.getElementById('shap-canvas'),
  spectrumCanvas: document.getElementById('spectrum-canvas'),
  xaiComparatorCanvas: document.getElementById('xai-comparator-canvas'),
  layerActivationsCanvas: document.getElementById('layer-activations-canvas'),
  cfTornadoCanvas: document.getElementById('cf-tornado-canvas'),
  hrvFreqCanvas: document.getElementById('hrv-freq-canvas'),
  faithfulnessCanvas: document.getElementById('faithfulness-canvas'),
  poincareCanvas: document.getElementById('poincare-canvas'),
  attentionMatrixCanvas: document.getElementById('attention-matrix-canvas'),
  tachogramCanvas: document.getElementById('tachogram-canvas'),
  confusionCanvas: document.getElementById('confusion-canvas'),
  rocCanvas: document.getElementById('roc-canvas'),
  stripTooltip: document.getElementById('strip-tooltip'),
  zoomTooltip: document.getElementById('zoom-tooltip'),
  spectrumTooltip: document.getElementById('spectrum-tooltip')
};

function triggerQrsBeep(isEctopic) {
  if (!state.audioEnabled) return;
  try {
    if (!state.audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      state.audioCtx = new Ctx();
    }
    if (state.audioCtx.state === 'suspended') state.audioCtx.resume();
    const osc = state.audioCtx.createOscillator();
    const gain = state.audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.value = isEctopic ? 620 : 880;
    gain.gain.setValueAtTime(0.06, state.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, state.audioCtx.currentTime + 0.055);
    osc.connect(gain);
    gain.connect(state.audioCtx.destination);
    osc.start();
    osc.stop(state.audioCtx.currentTime + 0.06);
  } catch (e) {
    // Ignore audio policy errors cleanly
  }
}

function rebuildHolter() {
  state.holterEpochs = generateHolterSession(state.preset, state.bpm);
  state.activeEpochIdx = 0;
}

function recomputePipeline() {
  if (state.customBaseRecord) {
    state.ecgRecord = reprocessCustomRecord(state.customBaseRecord, {
      denoise: state.denoise,
      sensorProfile: state.sensorProfile
    });
  } else {
    state.ecgRecord = generateSyntheticECG({
      preset: state.preset,
      sensorProfile: state.sensorProfile,
      bpm: state.bpm,
      motionNoise: state.motionNoise,
      baselineWander: state.baselineWander,
      emgNoise: state.emgNoise,
      powerline60: state.powerline60,
      denoise: state.denoise,
      counterfactuals: state.counterfactuals,
      seed: state.seed
    });
  }

  state.spectrum = computePowerSpectrum(state.ecgRecord.signal, state.ecgRecord.raw, FS);
  state.telemetry = analyzeSignalTelemetry(state.ecgRecord.signal, state.ecgRecord.beats, FS);

  if (state.telemetry.beats.length > 0) {
    state.selectedBeatIndex = Math.max(0, Math.min(state.selectedBeatIndex, state.telemetry.beats.length - 1));
  } else {
    state.selectedBeatIndex = 0;
  }

  state.inference = runExplainableInference({
    ecgRecord: state.ecgRecord,
    telemetry: state.telemetry,
    preset: state.preset,
    counterfactuals: state.counterfactuals,
    denoise: state.denoise,
    xaiMode: state.xaiMode
  });

  if (state.holterEpochs.length === 0) {
    rebuildHolter();
  }

  updateDOM();
  renderAllCanvases();
}

function updateDOM() {
  const { ecgRecord, telemetry, inference, spectrum } = state;
  const top = inference.topClass;
  const profile = SENSOR_PROFILES[state.sensorProfile];

  els.sensorBadge.textContent = profile.badge;
  els.sensorDesc.textContent = profile.description;
  els.sourceModePill.textContent = ecgRecord.isCustomUpload ? (ecgRecord.sourceLabel || 'Imported CSV/JSON') : 'Synthetic Live';
  els.sourceModePill.className = `pill ${ecgRecord.isCustomUpload ? 'pill-purple' : ''}`;

  const snr = ecgRecord.snrDb;
  els.snrBadge.textContent = `SNR: ${snr.toFixed(1)} dB`;
  els.snrBadge.className = `pill ${snr >= 18 ? 'pill-good' : snr >= 10 ? 'pill-warn' : 'pill-danger'}`;

  if (spectrum.bandShares) {
    els.bandBasePill.textContent = `Drift: ${spectrum.bandShares.baselinePct}%`;
    els.bandQrsPill.textContent = `QRS Band: ${spectrum.bandShares.qrsPct}%`;
    els.bandEmgPill.textContent = `EMG: ${spectrum.bandShares.emgPct}%`;
  }

  els.spectrum60HzPill.textContent = state.powerline60
    ? state.denoise
      ? '60 Hz Hum: Attenuated'
      : '60 Hz Hum: Active Spike'
    : '60 Hz Notch: Standby';
  els.spectrum60HzPill.className = `pill ${
    state.powerline60 && !state.denoise ? 'pill-danger' : state.powerline60 ? 'pill-warn' : ''
  }`;

  els.kpiClass.textContent = top.label;
  els.kpiClass.style.color = top.color;
  els.kpiConf.textContent = `Confidence: ${(top.prob * 100).toFixed(1)}% · Entropy: ${inference.entropy.toFixed(2)}`;
  els.kpiHr.innerHTML = `${telemetry.heartRate} <small>BPM</small>`;
  els.kpiRr.textContent = `Mean R-R: ${telemetry.meanRR} ms`;
  els.kpiHrv.innerHTML = `${telemetry.rmssd} / ${telemetry.sdnn} <small>ms</small>`;
  els.kpiIrregularity.textContent =
    telemetry.irregularityRatio > 0.14
      ? 'Rhythm Regularity: Irregular'
      : telemetry.irregularityRatio > 0.07
      ? 'Rhythm Regularity: Ectopic Shifts'
      : 'Rhythm Regularity: Stable';
  els.kpiMorph.innerHTML = `${telemetry.meanQrs} / ${telemetry.meanPr ?? '—'} <small>ms</small>`;
  els.kpiSqi.textContent = `Signal Quality Index: ${telemetry.sqi}%`;

  if (!state.alertDismissed && (top.key === 'VTACH' || (top.key === 'AFIB' && top.prob > 0.65))) {
    els.alertBanner.classList.remove('hidden');
    els.alertTitle.textContent =
      top.key === 'VTACH' ? 'CRITICAL ALERT: Ventricular Tachycardia' : 'WARNING: Atrial Fibrillation Detected';
    els.alertMessage.textContent = `${(top.prob * 100).toFixed(1)}% posterior confidence · Heart Rate ${telemetry.heartRate} BPM · Review XAI attribution immediately.`;
  } else {
    els.alertBanner.classList.add('hidden');
  }

  const activeBeat = telemetry.beats[state.selectedBeatIndex];
  els.beatSelectorLabel.textContent = activeBeat
    ? `Beat #${activeBeat.index} (${activeBeat.type})`
    : 'No Beat Selected';

  els.probBars.innerHTML = inference.classProbs
    .slice(0, 5)
    .map(
      (c) => `
      <div class="prob-item">
        <span class="mono">${c.short}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width:${(c.prob * 100).toFixed(1)}%; background:${c.color};"></div>
        </div>
        <span class="mono">${(c.prob * 100).toFixed(1)}%</span>
      </div>`
    )
    .join('');

  const r = inference.rationale;
  els.clinicalRationale.innerHTML = `
    <div class="rationale-callout">
      <strong>${r.headline}</strong><br/>
      ${r.morphologyDesc}
    </div>
    <div>${r.cfSentence}</div>
    <div class="helper-text">${r.signalNote}</div>
  `;

  els.triageBadge.textContent = inference.triage.level;
  els.triageBadge.className = `pill ${inference.triage.badgeClass}`;
  els.triageSteps.innerHTML = inference.triage.steps.map((s) => `<li>${s}</li>`).join('');

  els.barEpistemic.style.width = `${Math.min(100, inference.epistemicUncertainty * 180).toFixed(0)}%`;
  els.valEpistemic.textContent = inference.epistemicUncertainty.toFixed(2);
  els.barAleatoric.style.width = `${Math.min(100, inference.aleatoricUncertainty * 180).toFixed(0)}%`;
  els.valAleatoric.textContent = inference.aleatoricUncertainty.toFixed(2);

  els.faithfulnessCurvePill.textContent = `Insertion AUC: ${inference.insertionAuc.toFixed(2)} · Deletion AUC: ${inference.deletionAuc.toFixed(2)}`;
  els.metricDeletion.textContent = inference.deletionAuc.toFixed(2);
  els.metricInsertion.textContent = inference.insertionAuc.toFixed(2);
  els.metricFaithfulness.textContent = inference.faithfulness.toFixed(2);

  els.consensusBadge.textContent = `Inter-Method Consensus: ${(inference.consensusScore * 100).toFixed(0)}%`;

  // Holter 60s Epoch Cards
  els.holterTimeline.innerHTML = state.holterEpochs
    .map((ep, idx) => {
      const sevClass =
        ep.severity === 'critical'
          ? 'pill-danger'
          : ep.severity === 'warning'
          ? 'pill-warn'
          : ep.severity === 'caution'
          ? 'pill-purple'
          : 'pill-good';
      return `
        <button class="holter-epoch ${idx === state.activeEpochIdx ? 'active' : ''}" data-epoch="${idx}" type="button">
          <div class="holter-epoch-top">
            <span class="mono">${ep.label}</span>
            <span class="pill ${sevClass}">${ep.shortName}</span>
          </div>
          <div class="holter-epoch-rhythm mono">${ep.bpm} BPM</div>
          <div class="holter-epoch-meta">SDNN ${ep.sdnn} ms · Ectopy: ${ep.ectopicCount}</div>
        </button>`;
    })
    .join('');

  document.querySelectorAll('.holter-epoch').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.epoch);
      const ep = state.holterEpochs[idx];
      if (!ep) return;
      state.activeEpochIdx = idx;
      state.customBaseRecord = null;
      state.alertDismissed = false;
      state.preset = ep.preset;
      state.bpm = ep.bpm;
      state.seed = ep.seed;
      els.sliderBpm.value = state.bpm;
      els.valBpm.textContent = `${state.bpm} BPM`;
      els.presetBtns.forEach((b) => b.classList.toggle('active', b.dataset.preset === ep.preset));
      els.holterActivePill.textContent = `Epoch ${idx + 1} (${ep.label}) · ${ep.shortName}`;
      recomputePipeline();
    });
  });

  els.tachogramPill.textContent = `Mean R-R: ${telemetry.meanRR} ms · SDNN: ${telemetry.sdnn} ms`;
  els.hrvFreqPill.textContent = `LF/HF: ${telemetry.lfHfRatio} · SD1/SD2: ${telemetry.sd1}/${telemetry.sd2} ms`;
  els.poincarePattern.textContent =
    top.key === 'AFIB'
      ? 'Dispersed Stochastic Cloud'
      : top.key === 'PVC' || top.key === 'PAC'
      ? 'Multi-Lobe Ectopic Outliers'
      : 'Tight Elliptical Cluster';

  // Filter & Sort Beat Table
  let rows = inference.beatSummaries.map((b, origIdx) => ({ ...b, origIdx }));
  if (state.beatFilter === 'NORMAL') {
    rows = rows.filter((b) => b.type === 'NSR' || b.type === 'DETECTED');
  } else if (state.beatFilter === 'ECTOPIC') {
    rows = rows.filter((b) => b.type !== 'NSR' && b.type !== 'DETECTED');
  }

  if (state.beatSort === 'SALIENCY') {
    rows.sort((a, b) => b.peakSaliency - a.peakSaliency);
  } else if (state.beatSort === 'QRS') {
    rows.sort((a, b) => b.qrsWidthMs - a.qrsWidthMs);
  } else if (state.beatSort === 'RR') {
    rows.sort((a, b) => (b.rrMs || 0) - (a.rrMs || 0));
  }

  els.beatCountPill.textContent = `${rows.length} / ${inference.beatSummaries.length} Beats Shown`;
  els.beatTableBody.innerHTML = rows
    .map(
      (b) => `
      <tr class="${b.origIdx === state.selectedBeatIndex ? 'row-highlight' : ''}">
        <td class="mono">#${b.index} (${b.type})</td>
        <td class="mono">${b.timeSec.toFixed(2)}s</td>
        <td class="mono">${b.rrMs ?? '—'}</td>
        <td class="mono">${b.qrsWidthMs}</td>
        <td class="mono">${b.peakSaliency.toFixed(2)}</td>
        <td>${b.dominantRegion}</td>
        <td><button class="btn btn-xs inspect-beat-btn" data-idx="${b.origIdx}">Inspect</button></td>
      </tr>`
    )
    .join('');

  document.querySelectorAll('.inspect-beat-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.selectedBeatIndex = Number(btn.dataset.idx);
      switchTab('workstation');
      updateDOM();
      renderAllCanvases();
    });
  });

  // Pipeline Stage Inspector
  const stage = PIPELINE_STAGES.find((s) => s.id === state.selectedStageId) || PIPELINE_STAGES[0];
  els.stageInspector.innerHTML = `
    <div>
      <h3>${stage.step}: ${stage.title}</h3>
      <p class="helper-text">${stage.details}</p>
    </div>
    <div class="stage-specs">
      <div class="stage-spec-item"><span class="mini-label">Output Tensor</span><strong class="mono">${stage.tensor}</strong></div>
      <div class="stage-spec-item"><span class="mini-label">Parameters</span><strong class="mono">${stage.params}</strong></div>
      <div class="stage-spec-item"><span class="mini-label">Receptive Field</span><strong>${stage.receptiveField}</strong></div>
      <div class="stage-spec-item"><span class="mini-label">XAI Hook</span><strong>${stage.xaiHook}</strong></div>
    </div>
  `;

  // Hardware Profiler Strip
  const hw = HARDWARE_TARGETS[state.selectedHardware] || HARDWARE_TARGETS.CORTEX_M55_INT8;
  els.hardwareProfileStrip.innerHTML = `
    <div class="hw-metric"><span>Inference Latency</span><strong class="mono">${hw.latencyMs} ms</strong></div>
    <div class="hw-metric"><span>Peak Arena RAM</span><strong class="mono">${hw.ramMb} MB</strong></div>
    <div class="hw-metric"><span>Energy / 10s Strip</span><strong class="mono">${hw.energyMj} mJ</strong></div>
    <div class="hw-metric"><span>Throughput</span><strong class="mono">${hw.throughputHz} strips/s</strong></div>
    <div class="hw-metric"><span>Watch Battery / 24h</span><strong class="mono">${hw.batteryPctDay}%</strong></div>
  `;
}

function renderAllCanvases() {
  const activeTab = document.querySelector('.tab-panel.active')?.id;

  if (activeTab === 'tab-workstation') {
    renderECGStrip(els.ecgStripCanvas, {
      ecgRecord: state.ecgRecord,
      inference: state.inference,
      telemetry: state.telemetry,
      xaiMode: state.xaiMode,
      playheadIdx: state.playheadIdx,
      selectedBeatIndex: state.selectedBeatIndex,
      hoverIdx: state.stripHoverIdx,
      gainScale: state.gainScale,
      saliencyThreshold: state.saliencyThreshold,
      showGhostRaw: state.showGhostRaw,
      calipers: state.calipers,
      preset: state.preset
    });

    state.zoomBounds = renderBeatZoom(els.beatZoomCanvas, {
      ecgRecord: state.ecgRecord,
      inference: state.inference,
      beat: state.telemetry.beats[state.selectedBeatIndex],
      xaiMode: state.xaiMode,
      hoverSampleOffset: state.zoomHoverOffset,
      showTemplateOverlay: state.showTemplateOverlay,
      showFiducials: state.showFiducials,
      counterfactuals: state.counterfactuals,
      sensorProfileKey: state.sensorProfile,
      gainScale: state.gainScale
    });

    renderPhaseShap(els.shapCanvas, state.inference.phaseShap);
    state.spectrumBounds = renderSpectrumChart(els.spectrumCanvas, state.spectrum, state.spectrumHoverBin);
  } else if (activeTab === 'tab-comparator') {
    renderXaiComparator(els.xaiComparatorCanvas, {
      ecgRecord: state.ecgRecord,
      inference: state.inference,
      consensusScore: state.inference.consensusScore
    });
    renderLayerActivations(els.layerActivationsCanvas, state.inference.layerActivations);
    renderCounterfactualTornado(els.cfTornadoCanvas, state.inference.cfSensitivity);
  } else if (activeTab === 'tab-batch') {
    renderPoincare(els.poincareCanvas, state.telemetry, state.inference.topClass.short);
    state.matrixBounds = renderAttentionHeatmap(
      els.attentionMatrixCanvas,
      state.telemetry.beats,
      state.inference.attentionMatrix,
      state.selectedBeatIndex,
      state.matrixHoverCell
    );
    renderRRTachogram(els.tachogramCanvas, state.telemetry.beats, state.selectedBeatIndex);
    renderHrvFrequencyBars(els.hrvFreqCanvas, state.telemetry);
  } else if (activeTab === 'tab-architecture') {
    renderConfusionMatrix(els.confusionCanvas, state.cmShowCounts);
    renderROCCurves(els.rocCanvas, state.rocShowMacro);
    renderFaithfulnessCurves(els.faithfulnessCanvas, state.inference.faithfulnessCurves, true);
  }
}

function switchTab(tabName) {
  els.navTabs.forEach((t) => t.classList.toggle('active', t.dataset.tab === tabName));
  els.tabPanels.forEach((p) => p.classList.toggle('active', p.id === `tab-${tabName}`));
  requestAnimationFrame(renderAllCanvases);
}

// Navigation
els.navTabs.forEach((tab) => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// Sensor & Activity Profiles
els.sensorPills.forEach((pill) => {
  pill.addEventListener('click', () => {
    state.sensorProfile = pill.dataset.sensor;
    els.sensorPills.forEach((p) => p.classList.toggle('active', p === pill));
    recomputePipeline();
  });
});

els.activityPills.forEach((pill) => {
  pill.addEventListener('click', () => {
    const key = pill.dataset.activity;
    const act = ACTIVITY_PROFILES[key];
    if (!act) return;
    state.activityProfile = key;
    state.motionNoise = act.motion;
    state.baselineWander = act.baseline;
    state.emgNoise = act.emg;
    state.powerline60 = act.powerline60;

    els.sliderMotion.value = act.motion;
    els.valMotion.textContent = `${act.motion}%`;
    els.sliderBaseline.value = act.baseline;
    els.valBaseline.textContent = `${act.baseline}%`;
    els.sliderEmg.value = act.emg;
    els.valEmg.textContent = `${act.emg}%`;
    els.chkPowerline.checked = act.powerline60;

    els.activityPills.forEach((p) => p.classList.toggle('active', p === pill));
    recomputePipeline();
  });
});

// Rhythm Presets
els.presetBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    state.customBaseRecord = null;
    state.alertDismissed = false;
    state.preset = btn.dataset.preset;
    state.seed = 101;
    state.bpm = RHYTHM_PRESETS[state.preset].defaultBpm;
    els.sliderBpm.value = state.bpm;
    els.valBpm.textContent = `${state.bpm} BPM`;

    els.presetBtns.forEach((b) => b.classList.toggle('active', b === btn));
    rebuildHolter();
    recomputePipeline();
  });
});

// XAI Mode & Strip Controls
els.xaiPills.forEach((pill) => {
  pill.addEventListener('click', () => {
    state.xaiMode = pill.dataset.xai;
    els.xaiPills.forEach((p) => p.classList.toggle('active', p === pill));
    recomputePipeline();
  });
});

els.selectGain.addEventListener('change', (e) => {
  state.gainScale = Number(e.target.value);
  renderAllCanvases();
});

els.selectSpeed.addEventListener('change', (e) => {
  state.sweepSpeed = Number(e.target.value);
});

els.sliderSalThresh.addEventListener('input', (e) => {
  state.saliencyThreshold = Number(e.target.value) / 100;
  els.valSalThresh.textContent = state.saliencyThreshold.toFixed(2);
  renderAllCanvases();
});

els.btnCalipers.addEventListener('click', () => {
  state.caliperMode = !state.caliperMode;
  els.btnCalipers.classList.toggle('active-toggle', state.caliperMode);
  els.ecgStripCanvas.style.cursor = state.caliperMode ? 'crosshair' : 'pointer';
  els.stripHelperText.textContent = state.caliperMode
    ? 'CALIPER MODE ACTIVE: Click-and-drag across any two points on the ECG strip to measure Δt (ms), Rate (BPM) & ΔV (mV)'
    : '25 mm/s clinical grid · Hover for saliency · Click beat markers to focus · Toggle Calipers to measure ms/mV';
});

els.btnClearCalipers.addEventListener('click', () => {
  state.calipers = { startIdx: null, endIdx: null, dragging: false };
  els.btnClearCalipers.classList.add('hidden');
  renderAllCanvases();
});

els.chkTemplateOverlay.addEventListener('change', (e) => {
  state.showTemplateOverlay = e.target.checked;
  renderAllCanvases();
});

els.chkFiducials.addEventListener('change', (e) => {
  state.showFiducials = e.target.checked;
  renderAllCanvases();
});

els.sliderBpm.addEventListener('input', (e) => {
  state.customBaseRecord = null;
  state.bpm = Number(e.target.value);
  els.valBpm.textContent = `${state.bpm} BPM`;
  recomputePipeline();
});

els.sliderMotion.addEventListener('input', (e) => {
  state.motionNoise = Number(e.target.value);
  els.valMotion.textContent = `${state.motionNoise}%`;
  recomputePipeline();
});

els.sliderBaseline.addEventListener('input', (e) => {
  state.baselineWander = Number(e.target.value);
  els.valBaseline.textContent = `${state.baselineWander}%`;
  recomputePipeline();
});

els.sliderEmg.addEventListener('input', (e) => {
  state.emgNoise = Number(e.target.value);
  els.valEmg.textContent = `${state.emgNoise}%`;
  recomputePipeline();
});

els.chkPowerline.addEventListener('change', (e) => {
  state.powerline60 = e.target.checked;
  recomputePipeline();
});

els.chkDenoise.addEventListener('change', (e) => {
  state.denoise = e.target.checked;
  recomputePipeline();
});

els.chkGhostRaw.addEventListener('change', (e) => {
  state.showGhostRaw = e.target.checked;
  renderAllCanvases();
});

const syncCounterfactual = () => {
  state.counterfactuals.restoreP = els.cfRestoreP.checked;
  state.counterfactuals.regularizeRR = els.cfRegularizeRR.checked;
  state.counterfactuals.narrowQrs = els.cfNarrowQrs.checked;
  recomputePipeline();
};

els.cfRestoreP.addEventListener('change', syncCounterfactual);
els.cfRegularizeRR.addEventListener('change', syncCounterfactual);
els.cfNarrowQrs.addEventListener('change', syncCounterfactual);

els.btnResetCf.addEventListener('click', () => {
  els.cfRestoreP.checked = false;
  els.cfRegularizeRR.checked = false;
  els.cfNarrowQrs.checked = false;
  syncCounterfactual();
});

els.btnApplySinusReset.addEventListener('click', () => {
  els.cfRestoreP.checked = true;
  els.cfRegularizeRR.checked = true;
  els.cfNarrowQrs.checked = true;
  syncCounterfactual();
});

els.btnPlayPause.addEventListener('click', () => {
  state.isPlaying = !state.isPlaying;
  els.btnPlayPause.textContent = state.isPlaying ? '⏸ Pause Strip' : '▶ Resume Strip';
});

els.btnAudioToggle.addEventListener('click', () => {
  state.audioEnabled = !state.audioEnabled;
  els.btnAudioToggle.textContent = state.audioEnabled ? '🔊 QRS Tone' : '🔇 QRS Tone';
  els.btnAudioToggle.classList.toggle('active-toggle', state.audioEnabled);
  if (state.audioEnabled) triggerQrsBeep(false);
});

els.btnPrevBeat.addEventListener('click', () => {
  if (state.telemetry.beats.length === 0) return;
  state.selectedBeatIndex =
    (state.selectedBeatIndex - 1 + state.telemetry.beats.length) % state.telemetry.beats.length;
  updateDOM();
  renderAllCanvases();
});

els.btnNextBeat.addEventListener('click', () => {
  if (state.telemetry.beats.length === 0) return;
  state.selectedBeatIndex = (state.selectedBeatIndex + 1) % state.telemetry.beats.length;
  updateDOM();
  renderAllCanvases();
});

els.btnDismissAlert.addEventListener('click', () => {
  state.alertDismissed = true;
  els.alertBanner.classList.add('hidden');
});

els.btnTheme.addEventListener('click', () => {
  const html = document.documentElement;
  const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  renderAllCanvases();
});

// Strip Mouse Interactions (Caliper drag OR Beat selection + Hover)
function stripClientXToIdx(clientX) {
  const rect = els.ecgStripCanvas.getBoundingClientRect();
  const relX = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  return Math.round(relX * (WINDOW_SAMPLES - 1));
}

els.ecgStripCanvas.addEventListener('mousedown', (e) => {
  if (!state.caliperMode) return;
  const idx = stripClientXToIdx(e.clientX);
  state.calipers = { startIdx: idx, endIdx: idx, dragging: true };
  els.btnClearCalipers.classList.remove('hidden');
  renderAllCanvases();
});

window.addEventListener('mouseup', () => {
  if (state.calipers.dragging) {
    state.calipers.dragging = false;
  }
});

els.ecgStripCanvas.addEventListener('click', (e) => {
  if (state.caliperMode) return;
  if (!state.telemetry.beats.length) return;
  const clickIdx = stripClientXToIdx(e.clientX);
  let nearest = 0;
  let bestDist = Infinity;
  state.telemetry.beats.forEach((b, idx) => {
    const d = Math.abs(b.sampleIdx - clickIdx);
    if (d < bestDist) {
      bestDist = d;
      nearest = idx;
    }
  });
  state.selectedBeatIndex = nearest;
  updateDOM();
  renderAllCanvases();
});

els.ecgStripCanvas.addEventListener('mousemove', (e) => {
  const rect = els.ecgStripCanvas.getBoundingClientRect();
  const idx = stripClientXToIdx(e.clientX);
  state.stripHoverIdx = idx;

  if (state.caliperMode && state.calipers.dragging) {
    state.calipers.endIdx = idx;
  }

  const t = (idx / FS).toFixed(3);
  const mv = state.ecgRecord.signal[idx].toFixed(3);
  const cam = state.inference.gradCam[idx].toFixed(2);
  const ig = state.inference.integratedGradients[idx].toFixed(2);
  const sg = state.inference.smoothGrad[idx].toFixed(2);

  els.stripTooltip.classList.remove('hidden');
  els.stripTooltip.innerHTML = `<strong>t = ${t} s</strong> · ${mv} mV<br/>CAM: ${cam} · IG: ${ig} · SG: ${sg}`;
  els.stripTooltip.style.left = `${Math.min(rect.width - 190, Math.max(8, e.clientX - rect.left + 12))}px`;
  els.stripTooltip.style.top = `${Math.max(8, e.clientY - rect.top - 42)}px`;
  renderECGStrip(els.ecgStripCanvas, {
    ecgRecord: state.ecgRecord,
    inference: state.inference,
    telemetry: state.telemetry,
    xaiMode: state.xaiMode,
    playheadIdx: state.playheadIdx,
    selectedBeatIndex: state.selectedBeatIndex,
    hoverIdx: state.stripHoverIdx,
    gainScale: state.gainScale,
    saliencyThreshold: state.saliencyThreshold,
    showGhostRaw: state.showGhostRaw,
    calipers: state.calipers,
    preset: state.preset
  });
});

els.ecgStripCanvas.addEventListener('mouseleave', () => {
  state.stripHoverIdx = null;
  els.stripTooltip.classList.add('hidden');
});

els.beatZoomCanvas.addEventListener('mousemove', (e) => {
  if (!state.zoomBounds) return;
  const rect = els.beatZoomCanvas.getBoundingClientRect();
  const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const k = Math.round(relX * (state.zoomBounds.totalSamples - 1));
  state.zoomHoverOffset = k;
  const sIdx = Math.max(0, Math.min(WINDOW_SAMPLES - 1, state.zoomBounds.startIdx + k));
  const beat = state.telemetry.beats[state.selectedBeatIndex];
  const offsetMs = beat ? Math.round(((sIdx - beat.sampleIdx) / FS) * 1000) : 0;

  els.zoomTooltip.classList.remove('hidden');
  els.zoomTooltip.innerHTML = `<strong>Δt = ${offsetMs >= 0 ? '+' : ''}${offsetMs} ms</strong> · ${state.ecgRecord.signal[sIdx].toFixed(3)} mV<br/>Saliency: ${state.inference.gradCam[sIdx].toFixed(2)}`;
  els.zoomTooltip.style.left = `${Math.min(rect.width - 150, Math.max(8, e.clientX - rect.left + 10))}px`;
  els.zoomTooltip.style.top = `${Math.max(8, e.clientY - rect.top - 36)}px`;

  renderBeatZoom(els.beatZoomCanvas, {
    ecgRecord: state.ecgRecord,
    inference: state.inference,
    beat,
    xaiMode: state.xaiMode,
    hoverSampleOffset: k,
    showTemplateOverlay: state.showTemplateOverlay,
    showFiducials: state.showFiducials,
    counterfactuals: state.counterfactuals,
    sensorProfileKey: state.sensorProfile,
    gainScale: state.gainScale
  });
});

els.beatZoomCanvas.addEventListener('mouseleave', () => {
  state.zoomHoverOffset = null;
  els.zoomTooltip.classList.add('hidden');
});

// Spectrum Hover Readout
els.spectrumCanvas.addEventListener('mousemove', (e) => {
  if (!state.spectrumBounds || !state.spectrum) return;
  const rect = els.spectrumCanvas.getBoundingClientRect();
  const relX = (e.clientX - rect.left - state.spectrumBounds.padL) / state.spectrumBounds.w;
  const freq = Math.max(0, Math.min(75, relX * 75));
  let bestBin = 0;
  let bestDiff = Infinity;
  state.spectrum.freqs.forEach((f, idx) => {
    const d = Math.abs(f - freq);
    if (d < bestDiff) {
      bestDiff = d;
      bestBin = idx;
    }
  });
  state.spectrumHoverBin = bestBin;
  els.spectrumTooltip.classList.remove('hidden');
  els.spectrumTooltip.innerHTML = `<strong>${state.spectrum.freqs[bestBin].toFixed(1)} Hz</strong><br/>Clean: ${state.spectrum.cleanDb[bestBin].toFixed(1)} dB · Raw: ${state.spectrum.rawDb[bestBin].toFixed(1)} dB`;
  els.spectrumTooltip.style.left = `${Math.min(rect.width - 180, Math.max(8, e.clientX - rect.left + 12))}px`;
  els.spectrumTooltip.style.top = `${Math.max(8, e.clientY - rect.top - 42)}px`;
  renderSpectrumChart(els.spectrumCanvas, state.spectrum, state.spectrumHoverBin);
});

els.spectrumCanvas.addEventListener('mouseleave', () => {
  state.spectrumHoverBin = null;
  els.spectrumTooltip.classList.add('hidden');
  renderSpectrumChart(els.spectrumCanvas, state.spectrum, null);
});

// Attention Matrix Hover + Click
els.attentionMatrixCanvas.addEventListener('mousemove', (e) => {
  if (!state.matrixBounds) return;
  const { padL, padT, cellW, cellH, n } = state.matrixBounds;
  const rect = els.attentionMatrixCanvas.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left - padL) / cellW);
  const row = Math.floor((e.clientY - rect.top - padT) / cellH);
  if (row >= 0 && row < n && col >= 0 && col < n) {
    state.matrixHoverCell = { row, col };
    const w = state.inference.attentionMatrix[row][col];
    els.matrixHoverInfo.textContent = `Attn(#${row + 1} → #${col + 1}): ${(w * 100).toFixed(1)}%`;
    renderAttentionHeatmap(
      els.attentionMatrixCanvas,
      state.telemetry.beats,
      state.inference.attentionMatrix,
      state.selectedBeatIndex,
      state.matrixHoverCell
    );
  }
});

els.attentionMatrixCanvas.addEventListener('mouseleave', () => {
  state.matrixHoverCell = null;
  renderAttentionHeatmap(
    els.attentionMatrixCanvas,
    state.telemetry.beats,
    state.inference.attentionMatrix,
    state.selectedBeatIndex,
    null
  );
});

els.attentionMatrixCanvas.addEventListener('click', (e) => {
  if (!state.matrixBounds) return;
  const { padL, padT, cellW, cellH, n } = state.matrixBounds;
  const rect = els.attentionMatrixCanvas.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left - padL) / cellW);
  const row = Math.floor((e.clientY - rect.top - padT) / cellH);
  if (row >= 0 && row < n && col >= 0 && col < n) {
    state.selectedBeatIndex = col;
    const w = state.inference.attentionMatrix[row][col];
    els.matrixHoverInfo.textContent = `Selected #${col + 1} · Attn(#${row + 1}→#${col + 1}): ${(w * 100).toFixed(1)}%`;
    updateDOM();
    renderAllCanvases();
  }
});

// Beat Filter & Sort
els.selectBeatFilter.addEventListener('change', (e) => {
  state.beatFilter = e.target.value;
  updateDOM();
});

els.selectBeatSort.addEventListener('change', (e) => {
  state.beatSort = e.target.value;
  updateDOM();
});

// Architecture Interactive Pipeline Stages, Confusion Matrix Mode, ROC Macro, Hardware Profiler
els.pipeNodes.forEach((node) => {
  node.addEventListener('click', () => {
    state.selectedStageId = node.dataset.stage;
    els.pipeNodes.forEach((n) => n.classList.toggle('active', n === node));
    updateDOM();
  });
});

els.btnCmMode.addEventListener('click', () => {
  state.cmShowCounts = !state.cmShowCounts;
  els.btnCmMode.textContent = state.cmShowCounts ? 'Show %' : 'Show Counts';
  renderConfusionMatrix(els.confusionCanvas, state.cmShowCounts);
});

els.chkRocMacro.addEventListener('change', (e) => {
  state.rocShowMacro = e.target.checked;
  renderROCCurves(els.rocCanvas, state.rocShowMacro);
});

els.selectHardware.addEventListener('change', (e) => {
  state.selectedHardware = e.target.value;
  updateDOM();
});

// Sample Case Loader & CSV/JSON File Import
els.btnLoadSample.addEventListener('click', () => {
  const caseKey = els.selectSampleCase.value;
  const sc = SAMPLE_CASES[caseKey] || SAMPLE_CASES.CASE_PVC_BIGEMINY;
  const csvText = generateSampleCSVText(sc.preset, {
    bpm: sc.bpm,
    motion: sc.motion,
    baseline: sc.baseline,
    emg: sc.emg,
    seed: sc.seed
  });
  state.customBaseRecord = parseUploadedECG(csvText, {
    denoise: state.denoise,
    sensorProfile: state.sensorProfile,
    label: sc.label
  });
  state.alertDismissed = false;
  els.presetBtns.forEach((b) => b.classList.remove('active'));
  recomputePipeline();
});

els.btnUploadTrigger.addEventListener('click', () => els.fileUpload.click());

els.fileUpload.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    state.customBaseRecord = parseUploadedECG(text, {
      denoise: state.denoise,
      sensorProfile: state.sensorProfile,
      label: file.name
    });
    state.alertDismissed = false;
    els.presetBtns.forEach((b) => b.classList.remove('active'));
    recomputePipeline();
  } catch (err) {
    alert(`Could not import ECG file: ${err.message}`);
  } finally {
    e.target.value = '';
  }
});

els.btnDownloadSample.addEventListener('click', () => {
  const rows = ['time_sec,amplitude_mv,gradcam_saliency,smoothgrad_saliency'];
  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    rows.push(
      `${(i / FS).toFixed(4)},${state.ecgRecord.signal[i].toFixed(5)},${state.inference.gradCam[i].toFixed(4)},${state.inference.smoothGrad[i].toFixed(4)}`
    );
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `cardioxai_${state.preset.toLowerCase()}_250hz.csv`;
  a.click();
  URL.revokeObjectURL(url);
});

// Clinical Report Modal
els.btnExportReport.addEventListener('click', () => {
  const { inference, telemetry, ecgRecord } = state;
  const profileLabel = ecgRecord.sensorProfile?.label || 'Wearable Lead I';
  els.reportTimestamp.textContent = `Generated ${new Date().toLocaleString()} · Sampling Rate: 250 Hz · ${profileLabel}`;
  els.reportBody.innerHTML = `
    <div class="report-grid">
      <div class="report-box">
        <h4>Primary AI Classification</h4>
        <strong>${inference.topClass.label}</strong><br/>
        Posterior Confidence: ${(inference.topClass.prob * 100).toFixed(1)}%<br/>
        Predictive Entropy: ${inference.entropy.toFixed(2)} (Epistemic ${inference.epistemicUncertainty.toFixed(2)} / Aleatoric ${inference.aleatoricUncertainty.toFixed(2)})<br/>
        Triage Level: <strong>${inference.triage.level}</strong>
      </div>
      <div class="report-box">
        <h4>Wearable Telemetry &amp; HRV Metrics</h4>
        Heart Rate: <strong>${telemetry.heartRate} BPM</strong> (Mean R-R: ${telemetry.meanRR} ms)<br/>
        RMSSD / SDNN / pNN50: ${telemetry.rmssd} ms / ${telemetry.sdnn} ms / ${telemetry.pnn50}%<br/>
        Poincaré SD1 / SD2: ${telemetry.sd1} / ${telemetry.sd2} ms · LF/HF: ${telemetry.lfHfRatio}<br/>
        Mean QRS / PR: ${telemetry.meanQrs} / ${telemetry.meanPr ?? 'Absent'} ms · SNR ${ecgRecord.snrDb.toFixed(1)} dB
      </div>
    </div>
    <div class="report-box" style="margin-bottom:12px;">
      <h4>Explainable AI Clinical Rationale</h4>
      <p style="margin:4px 0;">${inference.rationale.morphologyDesc}</p>
      <p style="margin:4px 0;">${inference.rationale.cfSentence}</p>
      <p style="margin:4px 0; color:var(--text-muted); font-size:0.82rem;">
        Faithfulness: ${inference.faithfulness} · Insertion AUC: ${inference.insertionAuc} · Deletion AUC: ${inference.deletionAuc} · XAI Consensus: ${(inference.consensusScore * 100).toFixed(0)}%
      </p>
    </div>
    <div class="report-box" style="margin-bottom:12px;">
      <h4>Phase SHAP Attribution Breakdown</h4>
      <p style="margin:4px 0;">
        ${inference.phaseShap.map((p) => `<strong>${p.phase}:</strong> ${p.value >= 0 ? '+' : ''}${p.value.toFixed(2)}`).join(' &nbsp;|&nbsp; ')}
      </p>
    </div>
    <div class="report-box">
      <h4>Recommended Clinical Triage Steps</h4>
      <ul style="margin:4px 0; padding-left:18px;">
        ${inference.triage.steps.map((s) => `<li>${s}</li>`).join('')}
      </ul>
    </div>
  `;
  els.reportModal.classList.remove('hidden');
});

const closeModal = () => els.reportModal.classList.add('hidden');
els.btnCloseModal.addEventListener('click', closeModal);
els.modalBackdrop.addEventListener('click', closeModal);
els.btnPrintReport.addEventListener('click', () => window.print());

els.btnExportJson.addEventListener('click', () => {
  const payload = {
    timestamp: new Date().toISOString(),
    sensorProfile: state.sensorProfile,
    activityProfile: state.activityProfile,
    preset: state.ecgRecord.isCustomUpload ? 'CUSTOM_UPLOAD' : state.preset,
    prediction: state.inference.topClass,
    triage: state.inference.triage,
    probabilities: state.inference.classProbs,
    uncertainty: {
      epistemic: state.inference.epistemicUncertainty,
      aleatoric: state.inference.aleatoricUncertainty
    },
    telemetry: {
      heartRateBpm: state.telemetry.heartRate,
      meanRrMs: state.telemetry.meanRR,
      rmssdMs: state.telemetry.rmssd,
      sdnnMs: state.telemetry.sdnn,
      pnn50Pct: state.telemetry.pnn50,
      sd1Ms: state.telemetry.sd1,
      sd2Ms: state.telemetry.sd2,
      lfHfRatio: state.telemetry.lfHfRatio,
      meanQrsMs: state.telemetry.meanQrs,
      sqiPercent: state.telemetry.sqi,
      snrDb: Number(state.ecgRecord.snrDb.toFixed(2))
    },
    xai: {
      faithfulness: state.inference.faithfulness,
      consensusScore: state.inference.consensusScore,
      insertionAuc: state.inference.insertionAuc,
      deletionAuc: state.inference.deletionAuc,
      phaseShap: state.inference.phaseShap,
      cfSensitivity: state.inference.cfSensitivity,
      rationale: state.inference.rationale
    }
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `cardioxai_report_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

window.addEventListener('resize', () => renderAllCanvases());

let lastTick = performance.now();
function tick(now) {
  const dt = now - lastTick;
  if (state.isPlaying && dt >= 33) {
    const prevIdx = state.playheadIdx;
    state.playheadIdx =
      (state.playheadIdx + Math.max(1, Math.round((dt / 1000) * FS * state.sweepSpeed))) %
      WINDOW_SAMPLES;
    lastTick = now;

    if (state.audioEnabled && state.telemetry?.beats) {
      state.telemetry.beats.forEach((b, idx) => {
        const crossed =
          state.playheadIdx >= prevIdx
            ? b.sampleIdx >= prevIdx && b.sampleIdx < state.playheadIdx
            : b.sampleIdx >= prevIdx || b.sampleIdx < state.playheadIdx;
        if (crossed && state.lastBeepBeatIdx !== idx) {
          state.lastBeepBeatIdx = idx;
          triggerQrsBeep(b.type === 'PVC' || b.type === 'PAC' || b.type === 'VTACH');
        }
      });
    }

    if (document.getElementById('tab-workstation').classList.contains('active')) {
      renderECGStrip(els.ecgStripCanvas, {
        ecgRecord: state.ecgRecord,
        inference: state.inference,
        telemetry: state.telemetry,
        xaiMode: state.xaiMode,
        playheadIdx: state.playheadIdx,
        selectedBeatIndex: state.selectedBeatIndex,
        hoverIdx: state.stripHoverIdx,
        gainScale: state.gainScale,
        saliencyThreshold: state.saliencyThreshold,
        showGhostRaw: state.showGhostRaw,
        calipers: state.calipers,
        preset: state.preset
      });
    }
  }
  requestAnimationFrame(tick);
}

rebuildHolter();
recomputePipeline();
requestAnimationFrame(tick);
})(window);
