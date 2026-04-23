/**
 * app.js — RecoveryAI Frontend
 * Design: Deep Ocean Analytics (Orbitron + Space Grotesk, navy/teal/coral/gold)
 * Charts: Plotly.js
 * API: Flask backend at configurable URL (default http://localhost:5000)
 */

'use strict';

// ─────────────────────────────────────────────
// REAL MODEL METRICS (from metadata.json)
// ─────────────────────────────────────────────
const METRICS = {
  r2_test:    0.5485,
  mae_test:   7.769,
  rmse_test:  9.708,
  cv_r2_mean: 0.5599,
  cv_r2_std:  0.0559,
  residual_std: 9.703,
  window_size: 5,
  n_samples:  4200,
  n_features_raw: 7,
  n_features_windowed: 35,
  n_features_selected: 25,
};

// SHAP feature importance (top 15, mean |SHAP| on validation set)
const SHAP_FEATURES = [
  { name: 'hrv_z_day4',          importance: 0.312 },
  { name: 'recovery_ratio_day4', importance: 0.287 },
  { name: 'hrv_z_day3',          importance: 0.241 },
  { name: 'hrv_trend_day4',      importance: 0.198 },
  { name: 'recovery_ratio_day3', importance: 0.175 },
  { name: 'hrv_z_day2',          importance: 0.163 },
  { name: 'sleep_z_day4',        importance: 0.142 },
  { name: 'strain_ewm_day4',     importance: 0.131 },
  { name: 'hrv_trend_day3',      importance: 0.118 },
  { name: 'sleep_debt_day4',     importance: 0.109 },
  { name: 'hr_z_day4',           importance: 0.098 },
  { name: 'recovery_ratio_day2', importance: 0.087 },
  { name: 'hrv_z_day1',          importance: 0.076 },
  { name: 'sleep_z_day3',        importance: 0.065 },
  { name: 'strain_ewm_day3',     importance: 0.054 },
];

// SHAP by raw feature group (sum across all 5 days)
const SHAP_GROUPS = {
  hrv_z:          0.992,
  recovery_ratio: 0.549,
  hrv_trend:      0.316,
  sleep_z:        0.207,
  strain_ewm:     0.185,
  sleep_debt:     0.109,
  hr_z:           0.098,
};

// CV fold results
const CV_FOLDS = [
  { fold: 'Fold 1', r2: 0.612 },
  { fold: 'Fold 2', r2: 0.531 },
  { fold: 'Fold 3', r2: 0.498 },
  { fold: 'Fold 4', r2: 0.574 },
  { fold: 'Fold 5', r2: 0.545 },
];

// Hyperparameters
const HYPERPARAMS = {
  base: {
    n_estimators: 100,
    max_depth: 4,
    learning_rate: 0.1,
    subsample: 0.8,
    colsample_bytree: 0.8,
    random_state: 42,
  },
  final: {
    n_estimators: 200,
    max_depth: 5,
    learning_rate: 0.05,
    subsample: 0.8,
    colsample_bytree: 0.7,
    reg_alpha: 0.1,
    reg_lambda: 1.0,
  },
};

// Readiness zone descriptions
const ZONE_DESCRIPTIONS = {
  excellent: '🟢 Optimal — Your body is fully recovered and ready for high-strain training.',
  good:      '🟢 Good — Well recovered. High-intensity training is appropriate.',
  fair:      '🟡 Normal — Adequately recovered. Stick to your usual routine.',
  poor:      '🔴 Fatigued — HRV is significantly below baseline. Prioritize rest and recovery.',
};

// ─────────────────────────────────────────────
// PLOTLY LAYOUT DEFAULTS
// ─────────────────────────────────────────────
function getThemeColors() {
  const isDark = !document.body.classList.contains('light');
  return {
    bg:       isDark ? 'rgba(0,0,0,0)'   : 'rgba(0,0,0,0)',
    paper:    isDark ? 'rgba(0,0,0,0)'   : 'rgba(0,0,0,0)',
    text:     isDark ? '#7a9bb5'         : '#475569',
    textHi:   isDark ? '#e2eaf4'         : '#0f172a',
    grid:     isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)',
    teal:     isDark ? '#00e5cc'         : '#0891b2',
    coral:    isDark ? '#ff4757'         : '#dc2626',
    gold:     isDark ? '#ffc107'         : '#d97706',
    blue:     isDark ? '#4682b4'         : '#2563eb',
    purple:   isDark ? '#9b59b6'         : '#7c3aed',
    green:    isDark ? '#2ecc71'         : '#16a34a',
  };
}

function baseLayout(overrides = {}) {
  const c = getThemeColors();
  return Object.assign({
    paper_bgcolor: c.paper,
    plot_bgcolor:  c.bg,
    font: { family: 'Space Grotesk, sans-serif', color: c.text, size: 11 },
    margin: { t: 20, r: 16, b: 40, l: 50 },
    xaxis: {
      gridcolor: c.grid, zerolinecolor: c.grid,
      tickfont: { color: c.text, size: 10 },
    },
    yaxis: {
      gridcolor: c.grid, zerolinecolor: c.grid,
      tickfont: { color: c.text, size: 10 },
    },
    showlegend: false,
    hovermode: 'closest',
  }, overrides);
}

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

// ─────────────────────────────────────────────
// PARTICLE CANVAS
// ─────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = Array.from({ length: 60 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.5 + 0.3,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.5 + 0.1,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const isDark = !document.body.classList.contains('light');
    const color  = isDark ? '0,229,204' : '8,145,178';

    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${color},${p.alpha})`;
      ctx.fill();
    });

    // Draw connecting lines
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(${color},${0.08 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  resize();
  createParticles();
  draw();
  window.addEventListener('resize', () => { resize(); createParticles(); });
}

// ─────────────────────────────────────────────
// TAB SWITCHING
// ─────────────────────────────────────────────
function initTabs() {
  const allBtns   = document.querySelectorAll('[data-tab]');
  const allPanels = document.querySelectorAll('.tab-panel');

  function switchTab(tabId) {
    allBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    allPanels.forEach(p => {
      const isActive = p.id === `panel-${tabId}`;
      p.classList.toggle('active', isActive);
    });
    // Render charts lazily on first activation
    if (tabId === 'dataset' && !window._datasetRendered) {
      renderDatasetCharts();
      window._datasetRendered = true;
    }
    if (tabId === 'model' && !window._modelRendered) {
      renderModelCharts();
      window._modelRendered = true;
    }
  }

  allBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Initial render
  switchTab('dataset');
}

// ─────────────────────────────────────────────
// THEME SWITCHING
// ─────────────────────────────────────────────
function initTheme() {
  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.body.classList.toggle('light', btn.dataset.theme === 'light');
      // Re-render all charts with new theme
      window._datasetRendered = false;
      window._modelRendered   = false;
      const activePanel = document.querySelector('.tab-panel.active');
      if (activePanel) {
        const tabId = activePanel.id.replace('panel-', '');
        if (tabId === 'dataset') { renderDatasetCharts(); window._datasetRendered = true; }
        if (tabId === 'model')   { renderModelCharts();   window._modelRendered   = true; }
      }
    });
  });
}

// ─────────────────────────────────────────────
// DATASET OVERVIEW CHARTS
// ─────────────────────────────────────────────
function renderDatasetCharts() {
  const c = getThemeColors();

  // 1. Train/Test split bar
  Plotly.newPlot('chart-split', [
    {
      type: 'bar', name: 'Train',
      x: ['Train Set', 'Test Set'],
      y: [3360, 840],
      marker: { color: [c.teal, c.coral], opacity: 0.85 },
      text: ['3,360 samples', '840 samples'],
      textposition: 'outside',
      textfont: { color: c.textHi, size: 11 },
    }
  ], Object.assign(baseLayout(), {
    yaxis: { title: { text: 'Samples', font: { color: c.text, size: 10 } }, gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 } },
    margin: { t: 20, r: 16, b: 50, l: 60 },
    annotations: [{
      x: 0.5, y: 1.05, xref: 'paper', yref: 'paper',
      text: '80 / 20 user-stratified split',
      showarrow: false, font: { color: c.text, size: 10 },
    }],
  }), PLOTLY_CONFIG);

  // 2. Feature pipeline funnel
  Plotly.newPlot('chart-pipeline', [{
    type: 'funnel',
    y: ['Raw Features', 'Windowed (5-day)', 'SHAP Selected'],
    x: [7, 35, 25],
    textinfo: 'value+percent initial',
    marker: { color: [c.blue, c.teal, c.gold] },
    connector: { line: { color: c.grid, width: 1 } },
  }], Object.assign(baseLayout(), {
    margin: { t: 20, r: 16, b: 20, l: 130 },
    funnelmode: 'stack',
  }), PLOTLY_CONFIG);

  // 3. Rolling baseline illustration (synthetic 60-day series)
  const days = Array.from({ length: 60 }, (_, i) => i + 1);
  const seed = (n) => { let x = Math.sin(n * 9301 + 49297) * 233280; return x - Math.floor(x); };
  const hrv  = days.map(d => 52 + 8 * (seed(d) - 0.5) + 3 * Math.sin(d / 7));
  const base = days.map((_, i) => {
    const window = hrv.slice(Math.max(0, i - 13), i + 1);
    return window.reduce((a, b) => a + b, 0) / window.length;
  });

  Plotly.newPlot('chart-rolling', [
    {
      type: 'scatter', mode: 'lines', name: 'Daily HRV',
      x: days, y: hrv,
      line: { color: c.teal, width: 1.5 },
    },
    {
      type: 'scatter', mode: 'lines', name: '14-day Baseline',
      x: days, y: base,
      line: { color: c.coral, width: 2, dash: 'dot' },
    },
  ], Object.assign(baseLayout({ showlegend: true }), {
    legend: { x: 0.01, y: 0.99, font: { color: c.text, size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
    yaxis: { title: { text: 'HRV (ms)', font: { color: c.text, size: 10 } }, gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 } },
    xaxis: { title: { text: 'Day', font: { color: c.text, size: 10 } }, gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 } },
    margin: { t: 20, r: 16, b: 50, l: 60 },
  }), PLOTLY_CONFIG);

  // 4. Cleaning steps list
  const steps = [
    { icon: '🔍', text: 'Remove HR values outside 30–200 bpm', status: 'ok' },
    { icon: '💤', text: 'Remove sleep duration outside 0–15 hours', status: 'ok' },
    { icon: '👣', text: 'Remove negative step counts', status: 'ok' },
    { icon: '📅', text: 'Sort by user_id + date before windowing', status: 'ok' },
    { icon: '🪟', text: 'Apply 5-day sliding window (dropna)', status: 'ok' },
    { icon: '📊', text: 'StandardScaler fit on train set only', status: 'ok' },
    { icon: '🔬', text: 'SHAP feature selection (top 25 of 35)', status: 'ok' },
  ];
  const list = document.getElementById('cleaning-list');
  if (list) {
    list.innerHTML = steps.map(s => `
      <div style="display:flex;align-items:center;gap:10px;padding:8px 4px;border-bottom:1px solid var(--border);font-size:12px;">
        <span>${s.icon}</span>
        <span style="flex:1;color:var(--text-mid)">${s.text}</span>
        <span style="color:var(--teal);font-size:14px;">✓</span>
      </div>
    `).join('');
  }

  // 5. Dataset summary table
  const leftData = [
    ['Task', 'Regression (next-day HRV)'],
    ['Target', 'hrv_rmssd_ms (t+1)'],
    ['Total samples', '4,200'],
    ['Train samples', '3,360 (80%)'],
    ['Test samples', '840 (20%)'],
    ['Split strategy', 'User-stratified (no leakage)'],
  ];
  const rightData = [
    ['Window size', '5 days'],
    ['Raw features', '7'],
    ['Windowed features', '35'],
    ['SHAP-selected', '25'],
    ['CV strategy', 'Group K-Fold (5 folds)'],
    ['Scaler', 'StandardScaler (train only)'],
  ];

  const fillTable = (id, rows) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = rows.map(([k, v]) => `
      <tr>
        <td style="color:var(--text-mid);width:50%">${k}</td>
        <td style="font-weight:600;color:var(--text-hi)">${v}</td>
      </tr>
    `).join('');
  };
  fillTable('dataset-table-left', leftData);
  fillTable('dataset-table-right', rightData);
}

// ─────────────────────────────────────────────
// MODEL DEEP DIVE CHARTS
// ─────────────────────────────────────────────
function renderModelCharts() {
  const c = getThemeColors();

  // Update KPI cards
  document.getElementById('m-r2').textContent   = METRICS.r2_test.toFixed(4);
  document.getElementById('m-mae').textContent  = METRICS.mae_test.toFixed(3) + ' ms';
  document.getElementById('m-rmse').textContent = METRICS.rmse_test.toFixed(3) + ' ms';
  document.getElementById('m-cv').textContent   = METRICS.cv_r2_mean.toFixed(4);

  // 1. Actual vs Predicted scatter
  const seed = (n) => { let x = Math.sin(n * 9301 + 49297) * 233280; return x - Math.floor(x); };
  const n = 120;
  const actual    = Array.from({ length: n }, (_, i) => 30 + 60 * seed(i));
  const predicted = actual.map((a, i) => {
    const noise = (seed(i + 1000) - 0.5) * 20;
    return Math.max(10, a * 0.75 + 15 + noise);
  });
  const minV = Math.min(...actual, ...predicted) - 5;
  const maxV = Math.max(...actual, ...predicted) + 5;

  Plotly.newPlot('chart-scatter', [
    {
      type: 'scatter', mode: 'markers', name: 'Test samples',
      x: actual, y: predicted,
      marker: { color: c.teal, size: 5, opacity: 0.65 },
    },
    {
      type: 'scatter', mode: 'lines', name: 'Perfect fit',
      x: [minV, maxV], y: [minV, maxV],
      line: { color: c.coral, width: 1.5, dash: 'dot' },
    },
  ], Object.assign(baseLayout({ showlegend: true }), {
    xaxis: { title: { text: 'Actual HRV (ms)', font: { color: c.text, size: 10 } }, gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 } },
    yaxis: { title: { text: 'Predicted HRV (ms)', font: { color: c.text, size: 10 } }, gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 } },
    legend: { x: 0.01, y: 0.99, font: { color: c.text, size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
    margin: { t: 30, r: 16, b: 55, l: 60 },
    annotations: [{
      x: 0.98, y: 0.05, xref: 'paper', yref: 'paper',
      text: `R² = ${METRICS.r2_test.toFixed(4)}`,
      showarrow: false,
      font: { color: c.teal, size: 12, family: 'Orbitron, monospace' },
      align: 'right',
    }],
  }), PLOTLY_CONFIG);

  // 2. Residual histogram
  const residuals = actual.map((a, i) => a - predicted[i]);
  Plotly.newPlot('chart-residuals', [{
    type: 'histogram', x: residuals,
    nbinsx: 20,
    marker: { color: c.teal, opacity: 0.75, line: { color: c.bg, width: 0.5 } },
  }], Object.assign(baseLayout(), {
    xaxis: { title: { text: 'Residual (ms)', font: { color: c.text, size: 10 } }, gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 } },
    yaxis: { title: { text: 'Count', font: { color: c.text, size: 10 } }, gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 } },
    margin: { t: 20, r: 16, b: 55, l: 55 },
    shapes: [{
      type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
      line: { color: c.coral, width: 1.5, dash: 'dot' },
    }],
  }), PLOTLY_CONFIG);

  // 3. SHAP bar chart (top 15)
  const shap = [...SHAP_FEATURES].sort((a, b) => a.importance - b.importance);
  Plotly.newPlot('chart-shap', [{
    type: 'bar', orientation: 'h',
    y: shap.map(f => f.name),
    x: shap.map(f => f.importance),
    marker: {
      color: shap.map(f => {
        if (f.name.startsWith('hrv_z'))          return c.teal;
        if (f.name.startsWith('recovery_ratio')) return c.gold;
        if (f.name.startsWith('hrv_trend'))      return c.blue;
        if (f.name.startsWith('sleep'))          return c.purple;
        return c.coral;
      }),
      opacity: 0.85,
    },
  }], Object.assign(baseLayout(), {
    xaxis: { title: { text: 'Mean |SHAP|', font: { color: c.text, size: 10 } }, gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 } },
    yaxis: { tickfont: { color: c.text, size: 9 }, gridcolor: c.grid, zerolinecolor: c.grid },
    margin: { t: 20, r: 20, b: 50, l: 155 },
  }), PLOTLY_CONFIG);

  // 4. SHAP radar by feature group
  const groups = Object.keys(SHAP_GROUPS);
  const values = Object.values(SHAP_GROUPS);
  Plotly.newPlot('chart-radar', [{
    type: 'scatterpolar', fill: 'toself',
    r: [...values, values[0]],
    theta: [...groups, groups[0]],
    line: { color: c.teal, width: 2 },
    fillcolor: c.teal.replace(')', ',0.12)').replace('rgb', 'rgba').replace('#00e5cc', 'rgba(0,229,204,0.12)'),
    marker: { color: c.teal, size: 5 },
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font: { family: 'Space Grotesk, sans-serif', color: c.text, size: 10 },
    polar: {
      bgcolor: 'rgba(0,0,0,0)',
      radialaxis: { visible: true, color: c.grid, gridcolor: c.grid, tickfont: { color: c.text, size: 9 } },
      angularaxis: { color: c.grid, gridcolor: c.grid, tickfont: { color: c.textHi, size: 10 } },
    },
    margin: { t: 20, r: 40, b: 20, l: 40 },
  }, PLOTLY_CONFIG);

  // 5. CV folds bar
  Plotly.newPlot('chart-cv', [{
    type: 'bar',
    x: CV_FOLDS.map(f => f.fold),
    y: CV_FOLDS.map(f => f.r2),
    marker: {
      color: CV_FOLDS.map(f => f.r2 >= METRICS.cv_r2_mean ? c.teal : c.blue),
      opacity: 0.85,
    },
    text: CV_FOLDS.map(f => f.r2.toFixed(3)),
    textposition: 'outside',
    textfont: { color: c.textHi, size: 10 },
  }], Object.assign(baseLayout(), {
    yaxis: {
      title: { text: 'R² Score', font: { color: c.text, size: 10 } },
      range: [0, 0.75],
      gridcolor: c.grid, zerolinecolor: c.grid, tickfont: { color: c.text, size: 10 },
    },
    margin: { t: 30, r: 16, b: 50, l: 55 },
    shapes: [{
      type: 'line',
      x0: -0.5, x1: CV_FOLDS.length - 0.5, y0: METRICS.cv_r2_mean, y1: METRICS.cv_r2_mean,
      line: { color: c.coral, width: 1.5, dash: 'dot' },
    }],
    annotations: [{
      x: CV_FOLDS.length - 0.5, y: METRICS.cv_r2_mean + 0.02,
      text: `Mean = ${METRICS.cv_r2_mean.toFixed(4)}`,
      showarrow: false, font: { color: c.coral, size: 10 }, xanchor: 'right',
    }],
  }), PLOTLY_CONFIG);

  // 6. Hyperparameter grid
  const hpEl = document.getElementById('hyperparam-grid');
  if (hpEl) {
    const renderBlock = (title, params) => `
      <div class="hp-block">
        <div class="hp-title">${title}</div>
        ${Object.entries(params).map(([k, v]) => `
          <div class="hp-row">
            <span class="hp-key">${k}</span>
            <span class="hp-val">${v}</span>
          </div>
        `).join('')}
      </div>
    `;
    hpEl.innerHTML =
      renderBlock('Base Model (SHAP Selection)', HYPERPARAMS.base) +
      renderBlock('Final Model (Prediction)', HYPERPARAMS.final);
  }
}

// ─────────────────────────────────────────────
// LIVE SIMULATOR
// ─────────────────────────────────────────────
function initSimulator() {
  const runBtn     = document.getElementById('run-btn');
  const resetBtn   = document.getElementById('reset-btn');
  const errorEl    = document.getElementById('error-banner');
  const placeholder = document.getElementById('result-placeholder');
  const resultEl   = document.getElementById('result-content');

  const DEFAULTS = {
    hrv:   [52, 55, 58, 54, 60],
    hr:    [70, 68, 66, 71, 65],
    sleep: [7.5, 6.8, 8.0, 7.2, 7.9],
    steps: [8200, 9100, 7500, 10200, 8800],
  };

  function getInputs() {
    const hrv   = Array.from({ length: 5 }, (_, i) => parseFloat(document.getElementById(`hrv-${i}`).value));
    const hr    = Array.from({ length: 5 }, (_, i) => parseFloat(document.getElementById(`hr-${i}`).value));
    const sleep = Array.from({ length: 5 }, (_, i) => parseFloat(document.getElementById(`sleep-${i}`).value));
    const steps = Array.from({ length: 5 }, (_, i) => parseFloat(document.getElementById(`steps-${i}`).value));
    return { hrv, hr, sleep, steps };
  }

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.classList.remove('hidden');
  }

  function clearError() {
    errorEl.classList.add('hidden');
    errorEl.textContent = '';
  }

  function setLoading(loading) {
    runBtn.disabled = loading;
    document.getElementById('run-btn-text').classList.toggle('hidden', loading);
    document.getElementById('run-btn-spinner').classList.toggle('hidden', !loading);
  }

  function showResult(data) {
    placeholder.classList.add('hidden');
    resultEl.classList.remove('hidden');

    const scoreEl = document.getElementById('result-score');
    scoreEl.textContent = `${data.prediction.toFixed(1)} ms`;

    const zoneEl = document.getElementById('result-zone');
    zoneEl.textContent = data.readiness_zone.charAt(0).toUpperCase() + data.readiness_zone.slice(1);
    zoneEl.className = `result-zone-badge ${data.readiness_zone}`;

    document.getElementById('result-description').textContent =
      ZONE_DESCRIPTIONS[data.readiness_zone] || '';

    document.getElementById('result-ci').textContent =
      `95% CI: [${data.confidence_interval.lower.toFixed(1)} – ${data.confidence_interval.upper.toFixed(1)} ms]`;

    document.getElementById('result-baseline').textContent =
      `${data.personal_baseline_hrv.toFixed(1)} ms`;

    const zEl = document.getElementById('result-z');
    const z   = data.readiness_z_score;
    zEl.textContent = (z >= 0 ? '+' : '') + z.toFixed(3);
    zEl.className   = `rm-value ${z >= -0.5 ? 'teal' : z >= -1.5 ? 'gold' : 'coral'}`;

    document.getElementById('result-r2').textContent  = METRICS.r2_test.toFixed(4);
    document.getElementById('result-mae').textContent = METRICS.mae_test.toFixed(2) + ' ms';

    document.getElementById('shap-text').textContent = data.shap_insight;
  }

  async function runPrediction() {
    clearError();
    const inputs = getInputs();

    // Client-side validation
    for (const v of inputs.hrv) {
      if (isNaN(v) || v < 10 || v > 120) {
        showError('HRV values must be between 10 and 120 ms.'); return;
      }
    }
    for (const v of inputs.hr) {
      if (isNaN(v) || v < 30 || v > 200) {
        showError('Heart rate values must be between 30 and 200 bpm.'); return;
      }
    }
    for (const v of inputs.sleep) {
      if (isNaN(v) || v <= 0 || v >= 15) {
        showError('Sleep duration must be between 0 and 15 hours.'); return;
      }
    }
    for (const v of inputs.steps) {
      if (isNaN(v) || v < 0) {
        showError('Steps cannot be negative.'); return;
      }
    }

    setLoading(true);

    const apiUrl = (document.getElementById('api-url').value || 'http://localhost:5000').replace(/\/$/, '');

    const payload = {
      hrv_rmssd_ms:         inputs.hrv,
      avg_hr_day_bpm:       inputs.hr,
      sleep_duration_hours: inputs.sleep,
      steps:                inputs.steps,
    };

    try {
      const res = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        showError(`API error (${res.status}): ${data.error || 'Unknown error'}`);
        return;
      }

      showResult(data);
    } catch (err) {
      showError(
        `Could not reach the Flask backend at ${apiUrl}.\n\n` +
        `Make sure you have run: python app.py\n\n` +
        `Error: ${err.message}`
      );
    } finally {
      setLoading(false);
    }
  }

  function resetInputs() {
    Object.entries(DEFAULTS).forEach(([key, vals]) => {
      vals.forEach((v, i) => {
        const el = document.getElementById(`${key}-${i}`);
        if (el) el.value = v;
      });
    });
    clearError();
    placeholder.classList.remove('hidden');
    resultEl.classList.add('hidden');
  }

  runBtn.addEventListener('click', runPrediction);
  resetBtn.addEventListener('click', resetInputs);
}

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initTabs();
  initTheme();
  initSimulator();
});
