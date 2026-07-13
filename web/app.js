'use strict';

const API_BASE = (() => {
  const src = document.currentScript?.src;
  return src ? new URL(src).pathname.replace(/\/[^/]+$/, '') : '';
})();

async function apiFetch(path, opts = {}) {
  const res = await fetch(API_BASE + path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ─── Formatting ───────────────────────────────────────────────────────────────

function fmtBytes(b) {
  if (!b) return '—';
  if (b >= 1e12) return (b / 1e12).toFixed(1) + ' TB';
  if (b >= 1e9)  return (b / 1e9).toFixed(1) + ' GB';
  if (b >= 1e6)  return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3)  return (b / 1e3).toFixed(0) + ' KB';
  return b + ' B';
}

function fmtRelTime(ts) {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s/60)}m ago`;
  return `${Math.floor(s/3600)}h ago`;
}

// ─── State ────────────────────────────────────────────────────────────────────

let state = {
  markets: [],
  datasets: [],
  availDates: {},      // { date: {raw, bronze, silver, gold} }
  availStats: null,
  selectedMarket: null,
  selectedDataset: null,
  datasetCompareMode: false,
  datasetPinned: [],
  datasetPartitionsByDataset: {},
  datasetAvailability: null,
  datasetPartitions: {},
  datasetCalendarViewYear: null,
  datasetCalendarViewMonth: null,
  datasetCalendarInfos: {},
  rangeStart: null,
  rangeEnd: null,
  selecting: false,
  viewYear: null,
  viewMonth: null,     // 0-11
  jobs: [],
};

// ─── Boot ─────────────────────────────────────────────────────────────────────

function getHashMarket() {
  const m = location.hash.match(/^#market\/(.+)$/)
  return m ? decodeURIComponent(m[1]) : null
}

async function boot() {
  loadGlobalStats();
  await loadDatasets();
  loadBacktestEnvs();
  await loadMarkets();
  if (state.markets.length > 0) {
    const hashMarket = getHashMarket()
    const initial = state.markets.find(m => m.market === hashMarket) || state.markets[0]
    document.getElementById('market-select').value = initial.market
    await selectMarket(initial)
  }
  loadJobs();
  setInterval(loadJobs, 8000);

  window.addEventListener('hashchange', async () => {
    const marketKey = getHashMarket()
    if (!marketKey) return
    const mkt = state.markets.find(m => m.market === marketKey)
    if (mkt && mkt.market !== state.selectedMarket?.market) {
      document.getElementById('market-select').value = mkt.market
      await selectMarket(mkt, false)
    }
  })
}

// ─── Global stats ─────────────────────────────────────────────────────────────

async function loadGlobalStats() {
  try {
    const stats = await apiFetch('/api/stats');
    if (stats.loading) {
      // Cache still warming — retry in 5 seconds
      setTimeout(loadGlobalStats, 5000);
      return;
    }
    document.getElementById('stat-raw').textContent = fmtBytes(stats.raw);
    document.getElementById('stat-raw-count').textContent = `${stats.raw_file_count} files`;
    document.getElementById('stat-bronze').textContent = fmtBytes(stats.bronze);
    document.getElementById('stat-silver').textContent = fmtBytes(stats.silver);
    document.getElementById('stat-gold').textContent = fmtBytes(stats.gold);
  } catch (e) {
    console.error('stats error', e);
  }
}

// ─── Markets ──────────────────────────────────────────────────────────────────

async function loadMarkets() {
  const markets = await apiFetch('/api/markets');
  state.markets = markets;
  const sel = document.getElementById('market-select');
  sel.innerHTML = markets.map(m =>
    `<option value="${m.market}" data-exchange="${m.exchange}" data-symbol="${m.symbol}">${m.market}</option>`
  ).join('');
  sel.addEventListener('change', () => {
    const opt = sel.options[sel.selectedIndex];
    selectMarket({
      market: opt.value,
      exchange: opt.dataset.exchange,
      symbol: opt.dataset.symbol,
    });
  });
}

// ─── Datasets ────────────────────────────────────────────────────────────────

async function loadDatasets() {
  const datasets = await apiFetch('/api/datasets');
  if (datasets && datasets.loading) {
    setTimeout(loadDatasets, 2000);
    return;
  }
  state.datasets = Array.isArray(datasets) ? datasets : [];
  if (!state.datasetPinned.length && state.datasets.length) {
    state.datasetPinned = [state.datasets[0].dataset];
  } else {
    state.datasetPinned = state.datasetPinned.filter(d => state.datasets.some(x => x.dataset === d));
    if (!state.datasetPinned.length && state.datasets.length) {
      state.datasetPinned = [state.datasets[0].dataset];
    }
  }
  renderDatasetCatalog();

  const sel = document.getElementById('dataset-select');
  sel.innerHTML = state.datasets.map(d =>
    `<option value="${escAttr(d.dataset)}">${escHtml(d.dataset)}</option>`
  ).join('');
  sel.onchange = () => selectDataset(sel.value);

  if (state.datasets.length > 0) {
    const initial = state.datasets.find(d => d.dataset === state.selectedDataset) || state.datasets[0];
    if (!state.selectedDataset || !state.datasets.some(d => d.dataset === state.selectedDataset)) {
      await selectDataset(initial.dataset);
    } else {
      sel.value = state.selectedDataset;
    }
  }
  renderDatasetCalendarControls();
  await refreshDatasetCalendar();
}

function renderDatasetCatalog() {
  const el = document.getElementById('dataset-table');
  if (!state.datasets.length) {
    el.innerHTML = '<div class="jobs-empty">No datasets found</div>';
    return;
  }

  const rows = state.datasets.map(d => `
    <tr class="dataset-row ${state.selectedDataset === d.dataset ? 'is-active' : ''}" data-dataset="${escAttr(d.dataset)}">
      <td>
        <input
          type="checkbox"
          class="dataset-pin"
          data-pin-dataset="${escAttr(d.dataset)}"
          ${state.datasetPinned.includes(d.dataset) ? 'checked' : ''}
        />
      </td>
      <td>${escHtml(d.dataset)}</td>
      <td>${escHtml(d.layer)}</td>
      <td>${escHtml((d.keys || []).join(' / '))}</td>
      <td>${d.exists ? escHtml(d.latest_date || '—') : 'missing'}</td>
      <td>${d.exists ? `${d.date_count || 0}d` : '—'}</td>
      <td>${d.exists ? `${d.file_count || 0}` : '—'}</td>
      <td>${d.exists ? fmtBytes(d.total_bytes || 0) : '—'}</td>
    </tr>
  `).join('');

  el.innerHTML = `
    <table class="dataset-table">
      <thead>
        <tr>
          <th>Pin</th>
          <th>Dataset</th>
          <th>Layer</th>
          <th>Keys</th>
          <th>Latest</th>
          <th>Dates</th>
          <th>Files</th>
          <th>Size</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  el.querySelectorAll('.dataset-row').forEach(row => {
    row.addEventListener('click', () => selectDataset(row.dataset.dataset));
  });
  el.querySelectorAll('.dataset-pin').forEach(pin => {
    pin.addEventListener('click', e => e.stopPropagation());
    pin.addEventListener('change', () => {
      const dataset = pin.dataset.pinDataset;
      if (pin.checked) {
        if (!state.datasetPinned.includes(dataset)) state.datasetPinned.push(dataset);
      } else {
        state.datasetPinned = state.datasetPinned.filter(d => d !== dataset);
      }
      if (!state.datasetPinned.length && state.selectedDataset) {
        state.datasetPinned = [state.selectedDataset];
      }
      renderDatasetCatalog();
      renderDatasetCalendarControls();
      refreshDatasetCalendar();
    });
  });
}

async function selectDataset(dataset) {
  if (!dataset) return;
  state.selectedDataset = dataset;
  state.datasetPartitions = state.datasetPartitionsByDataset[dataset] || {};
  state.datasetPartitionsByDataset[dataset] = { ...state.datasetPartitions };
  state.datasetCalendarViewYear = null;
  state.datasetCalendarViewMonth = null;
  document.getElementById('dataset-select').value = dataset;
  renderDatasetCatalog();
  await refreshDatasetAvailability();
  initDatasetCalendarView();
  renderDatasetCalendarControls();
  await refreshDatasetCalendar();
}

async function refreshDatasetAvailability() {
  const dataset = state.selectedDataset;
  if (!dataset) return;

  const params = new URLSearchParams({ dataset });
  for (const [key, value] of Object.entries(state.datasetPartitions)) {
    if (value) params.set(key, value);
  }
  if (state.rangeStart) params.set('start', state.rangeStart);
  if (state.rangeEnd) params.set('end', state.rangeEnd);

  const info = await apiFetch(`/api/dataset-availability?${params.toString()}`);
  state.datasetAvailability = info;
  renderDatasetDetail(info);
}

function renderDatasetDetail(info) {
  const summaryEl = document.getElementById('dataset-summary');
  const partitionsEl = document.getElementById('dataset-partitions');
  const pathEl = document.getElementById('dataset-path');
  const controlsEl = document.getElementById('dataset-controls');
  const valuesEl = document.getElementById('dataset-values');
  const rangesEl = document.getElementById('dataset-ranges');
  const statusEl = document.getElementById('dataset-status');

  if (!info || info.error) {
    summaryEl.textContent = info?.error || 'Select a dataset.';
    partitionsEl.textContent = '—';
    pathEl.textContent = '—';
    controlsEl.innerHTML = '';
    valuesEl.innerHTML = '';
    rangesEl.innerHTML = '';
    statusEl.textContent = 'No dataset selected';
    return;
  }

  const keys = info.keys || [];
  const resolved = info.resolved_partitions || {};
  const stateLabel = {
    missing_dataset: 'Dataset missing on disk',
    needs_partition: `Select ${info.next_key}`,
    missing_partition: `Missing ${info.missing_key}=${info.missing_value}`,
    summary: 'Partition ready',
  }[info.state] || info.state || '';

  statusEl.textContent = `${info.dataset} · ${stateLabel}`;
  summaryEl.textContent = `${info.layer} layer · ${keys.length} key${keys.length === 1 ? '' : 's'}`;
  partitionsEl.textContent = keys.map(k => `${k}=${resolved[k] || '…'}`).join(' / ') || '—';
  pathEl.textContent = info.path || info.base_path || '—';

  controlsEl.innerHTML = renderPartitionControls(info);
  valuesEl.innerHTML = renderAvailableValues(info);
  rangesEl.innerHTML = renderMissingRanges(info);

  controlsEl.querySelectorAll('select[data-partition-key]').forEach(sel => {
    sel.addEventListener('change', async () => {
      const key = sel.dataset.partitionKey;
      state.datasetPartitions[key] = sel.value || null;
      state.datasetPartitionsByDataset[state.selectedDataset] = { ...state.datasetPartitions };
      const keyIndex = keys.indexOf(key);
      keys.slice(keyIndex + 1).forEach(k => { delete state.datasetPartitions[k]; });
      state.datasetPartitionsByDataset[state.selectedDataset] = { ...state.datasetPartitions };
      await refreshDatasetAvailability();
      await refreshDatasetCalendar();
    });
  });
}

function renderPartitionControls(info) {
  const keys = info.keys || [];
  const resolved = info.resolved_partitions || {};
  const key = info.next_key || info.missing_key || null;
  const values = info.available_values || [];

  const resolvedChips = keys
    .filter(k => resolved[k])
    .map(k => `<span class="pill">${escHtml(k)}=${escHtml(resolved[k])}</span>`)
    .join('');

  const selector = key ? `
    <label class="control-select-wrap">
      <span class="control-label">${escHtml(key)}</span>
      <select data-partition-key="${escAttr(key)}">
        <option value="">Select…</option>
        ${values.map(v => `<option value="${escAttr(v)}" ${state.datasetPartitions[key] === v ? 'selected' : ''}>${escHtml(v)}</option>`).join('')}
      </select>
    </label>
  ` : '';

  return `
    <div class="partition-row">
      ${resolvedChips || '<span class="muted">No partitions selected yet.</span>'}
    </div>
    ${selector}
  `;
}

function renderAvailableValues(info) {
  if (info.state !== 'needs_partition' && info.state !== 'missing_partition') return '';
  const values = info.available_values || [];
  if (!values.length) return '<div class="muted">No values available for the next partition key.</div>';
  return `
    <div class="detail-label">Available values for ${escHtml(info.next_key || info.missing_key)}</div>
    <div class="pill-list">
      ${values.map(v => `<span class="pill">${escHtml(v)}</span>`).join('')}
    </div>
  `;
}

function renderMissingRanges(info) {
  if (info.state !== 'summary') {
    return info.exists
      ? '<div class="muted">Pick a partition value to inspect date coverage.</div>'
      : '<div class="muted">Dataset not found on disk.</div>';
  }

  const ranges = info.missing_ranges || [];
  const rangeHtml = ranges.length
    ? `<div class="pill-list">${ranges.map(r => `<span class="pill pill-warn">${escHtml(r)}</span>`).join('')}</div>`
    : '<div class="muted">No missing dates in the selected window.</div>';

  return `
    <div class="detail-label">Date coverage</div>
    <div class="coverage-grid">
      <div class="coverage-item"><span>Min</span><strong>${escHtml(info.min_date || '—')}</strong></div>
      <div class="coverage-item"><span>Max</span><strong>${escHtml(info.max_date || '—')}</strong></div>
      <div class="coverage-item"><span>Days</span><strong>${escHtml(String(info.total_days || 0))}</strong></div>
      <div class="coverage-item"><span>Missing</span><strong>${escHtml(String(info.missing_days || 0))}</strong></div>
    </div>
    <div class="detail-label" style="margin-top:10px">Missing ranges</div>
    ${rangeHtml}
  `;
}

// ─── Dataset calendar ────────────────────────────────────────────────────────

const DATASET_CAL_DOW = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'];
const DATASET_CAL_MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
const DATASET_CAL_COLORS = ['#00dc96', '#3ab4f5', '#f0c430', '#f04060'];

function getDatasetCalendarDatasets() {
  const pinned = state.datasetCompareMode ? state.datasetPinned : [state.selectedDataset];
  return pinned.filter(Boolean).filter((d, i, arr) => arr.indexOf(d) === i);
}

function getDatasetPartitions(dataset) {
  return state.datasetPartitionsByDataset[dataset] || (dataset === state.selectedDataset ? state.datasetPartitions : {});
}

function initDatasetCalendarView() {
  if (state.datasetCalendarViewYear != null && state.datasetCalendarViewMonth != null) return;
  const info = state.datasetAvailability;
  const seed = info && info.max_date ? info.max_date : new Date().toISOString().slice(0, 10);
  const [y, m] = seed.split('-');
  state.datasetCalendarViewYear = parseInt(y, 10);
  state.datasetCalendarViewMonth = parseInt(m, 10) - 1;
}

function datasetCalendarRange() {
  initDatasetCalendarView();
  let y = state.datasetCalendarViewYear;
  let m = state.datasetCalendarViewMonth;
  const first = `${y}-${String(m + 1).padStart(2, '0')}-01`;
  m += 1;
  if (m > 11) {
    m = 0;
    y += 1;
  }
  const last = new Date(y, m + 1, 0);
  return {
    start: first,
    end: last.toISOString().slice(0, 10),
  };
}

function renderDatasetCalendarControls() {
  const singleBtn = document.getElementById('dataset-mode-single');
  const compareBtn = document.getElementById('dataset-mode-compare');
  singleBtn?.classList.toggle('is-active', !state.datasetCompareMode);
  compareBtn?.classList.toggle('is-active', !!state.datasetCompareMode);

  const legend = document.getElementById('dataset-calendar-legend');
  const datasets = getDatasetCalendarDatasets();
  legend.innerHTML = datasets.length ? datasets.map((dataset, idx) => `
    <span class="dataset-legend-chip">
      <span class="dataset-legend-swatch" style="background:${DATASET_CAL_COLORS[idx % DATASET_CAL_COLORS.length]}"></span>
      ${escHtml(dataset)}
    </span>
  `).join('') : '<span class="muted">No dataset selected.</span>';
}

async function refreshDatasetCalendar() {
  const grid = document.getElementById('dataset-calendar-grid');
  const spinner = document.getElementById('dataset-calendar-spinner');
  const status = document.getElementById('dataset-calendar-status');
  const label = document.getElementById('dataset-cal-month-label');
  const datasets = getDatasetCalendarDatasets();
  if (!datasets.length) {
    grid.innerHTML = '';
    label.textContent = '';
    status.textContent = 'Select a dataset.';
    return;
  }

  const { start, end } = datasetCalendarRange();
  spinner.style.display = 'flex';
  grid.style.display = 'none';
  status.textContent = `Loading ${datasets.length} dataset${datasets.length === 1 ? '' : 's'}…`;

  try {
    const infos = await Promise.all(datasets.map(async (dataset) => {
      const params = new URLSearchParams({ dataset, start, end });
      const partitions = getDatasetPartitions(dataset);
      for (const [key, value] of Object.entries(partitions)) {
        if (value) params.set(key, value);
      }
      const info = await apiFetch(`/api/dataset-availability?${params.toString()}`);
      return info && info.dataset ? info : { dataset, error: 'load failed' };
    }));
    state.datasetCalendarInfos = Object.fromEntries(infos.map(info => [info.dataset, info]));
    renderDatasetCalendar();
    const rendered = infos.find(info => info.state === 'summary') || infos[0];
    status.textContent = rendered?.state === 'summary'
      ? `${rendered.dataset} · ${rendered.total_days} days · ${rendered.missing_days} missing`
      : `${rendered?.dataset || 'Dataset'} · ${rendered?.state || 'unavailable'}`;
  } catch (e) {
    status.textContent = `Dataset calendar error: ${e.message}`;
    grid.innerHTML = '';
  } finally {
    spinner.style.display = 'none';
    grid.style.display = 'grid';
  }
}

function renderDatasetCalendar() {
  const label = document.getElementById('dataset-cal-month-label');
  const grid  = document.getElementById('dataset-calendar-grid');
  const datasets = getDatasetCalendarDatasets();
  const infos = state.datasetCalendarInfos || {};

  if (!datasets.length) {
    label.textContent = '';
    grid.innerHTML = '';
    return;
  }

  const months = [];
  let y = state.datasetCalendarViewYear;
  let m = state.datasetCalendarViewMonth;
  for (let i = 0; i < 2; i++) {
    months.push({ year: y, month: m });
    m++;
    if (m > 11) { m = 0; y++; }
  }

  const first = months[0], last = months[months.length - 1];
  label.textContent = `${DATASET_CAL_MONTHS[first.month]} ${first.year}  ·  ${DATASET_CAL_MONTHS[last.month]} ${last.year}`;

  grid.innerHTML = months.map(({ year, month }) => renderDatasetCalendarMonth(year, month, datasets, infos)).join('');
}

function renderDatasetCalendarMonth(year, month, datasets, infos) {
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const firstDow = (new Date(year, month, 1).getDay() + 6) % 7;
  const headers = DATASET_CAL_DOW.map(d => `<div class="dow-header">${d}</div>`).join('');
  const blanks = Array(firstDow).fill('<div class="day-cell empty"></div>').join('');
  const days = [];
  for (let d = 1; d <= daysInMonth; d++) {
    const iso = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
    days.push(renderDatasetCalendarDay(iso, d, datasets, infos));
  }
  return `
    <div class="month-block">
      <div class="month-name">${DATASET_CAL_MONTHS[month]} ${year}</div>
      <div class="month-grid">
        ${headers}
        ${blanks}
        ${days.join('')}
      </div>
    </div>
  `;
}

function renderDatasetCalendarDay(iso, dayNum, datasets, infos) {
  const strips = datasets.map((dataset, idx) => {
    const info = infos[dataset];
    const present = !!(info && info.state === 'summary' && info.date_map && info.date_map[iso] && info.date_map[iso].present);
    const size = info && info.state === 'summary' && info.date_map && info.date_map[iso] ? info.date_map[iso].size : 0;
    return { dataset, present, size, color: DATASET_CAL_COLORS[idx % DATASET_CAL_COLORS.length] };
  });

  const hasAny = strips.some(s => s.present);
  const cls = [
    'day-cell',
    'dataset-day-cell',
    hasAny ? 'has-data' : 'off',
  ].join(' ');

  const tooltipRows = strips.map(s => `
    <div class="tt-row">
      <span class="tt-label"><span class="dataset-tt-swatch" style="background:${s.color}"></span>${escHtml(s.dataset)}</span>
      <span class="tt-val">${s.present ? fmtBytes(s.size) : 'missing'}</span>
    </div>
  `).join('');

  const bars = strips.map(s => `
    <span class="dataset-bar ${s.present ? 'on' : 'off'}" style="--dataset-bar-color:${s.color}"></span>
  `).join('');

  return `
    <div class="${cls}" data-date="${iso}">
      <span class="day-num">${dayNum}</span>
      <div class="dataset-bars">${bars}</div>
      <div class="day-tooltip dataset-tooltip">
        <div class="tt-date">${iso}</div>
        ${tooltipRows}
      </div>
    </div>
  `;
}

// ─── Availability ─────────────────────────────────────────────────────────────

async function selectMarket(mkt, pushHash = true) {
  state.selectedMarket = mkt;
  document.getElementById('market-select').value = mkt.market;
  if (pushHash) location.hash = `#market/${encodeURIComponent(mkt.market)}`;

  document.getElementById('calendar-spinner').style.display = 'flex';
  document.getElementById('calendar-grid').style.display = 'none';

  try {
    const data = await apiFetch(`/api/availability?exchange=${mkt.exchange}&symbol=${mkt.symbol}`);
    if (data.loading) {
      // Cache warming — retry in 3 seconds
      setTimeout(() => selectMarket(mkt, false), 3000);
      return;
    }
    state.availDates = data.dates;
    state.availStats = data.stats;
    updateLegendCounts(data.stats);
    initCalendarView();
    renderCalendar();
    document.getElementById('calendar-spinner').style.display = 'none';
    document.getElementById('calendar-grid').style.display = 'grid';
    checkBronzeAvailable();
  } catch (e) {
    document.getElementById('calendar-spinner').style.display = 'none';
    document.getElementById('calendar-grid').style.display = 'grid';
    console.error('availability error', e);
  }
}

function updateLegendCounts(stats) {
  document.getElementById('leg-raw').textContent = stats.raw_dates + 'd';
  document.getElementById('leg-bronze').textContent = stats.bronze_dates + 'd';
  document.getElementById('leg-silver').textContent = stats.silver_dates + 'd';
  document.getElementById('leg-gold').textContent = stats.gold_dates + 'd';
}

// ─── Calendar ─────────────────────────────────────────────────────────────────

const DOW = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'];
const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];

let _dayHandlersAttached = false;

function initCalendarView() {
  const dates = Object.keys(state.availDates).sort();
  if (dates.length === 0) return;
  const last = dates[dates.length - 1].split('-');
  state.viewYear = parseInt(last[0]);
  state.viewMonth = parseInt(last[1]) - 1;
  // Back up one so the last month is on the right
  let m = state.viewMonth - 1;
  let y = state.viewYear;
  if (m < 0) { m += 12; y--; }
  state.viewYear = y;
  state.viewMonth = m;
  if (!_dayHandlersAttached) {
    attachDayHandlers();
    _dayHandlersAttached = true;
  }
}

function renderCalendar() {
  const label = document.getElementById('cal-month-label');
  const grid  = document.getElementById('calendar-grid');

  // Show 2 months starting from viewYear/viewMonth
  const months = [];
  let y = state.viewYear, m = state.viewMonth;
  for (let i = 0; i < 2; i++) {
    months.push({ year: y, month: m });
    m++;
    if (m > 11) { m = 0; y++; }
  }

  const first = months[0], last = months[months.length - 1];
  label.textContent = `${MONTHS[first.month]} ${first.year}  ·  ${MONTHS[last.month]} ${last.year}`;

  grid.innerHTML = months.map(({ year, month }) => renderMonth(year, month)).join('');
}

// Update only range-related classes in-place — no HTML rebuild needed
function updateRangeHighlight() {
  const grid = document.getElementById('calendar-grid');
  grid.querySelectorAll('.day-cell:not(.empty)').forEach(cell => {
    const d = cell.dataset.date;
    const inRange = state.rangeStart && state.rangeEnd && d >= state.rangeStart && d <= state.rangeEnd;
    cell.classList.toggle('in-range',    !!inRange);
    cell.classList.toggle('range-start', d === state.rangeStart);
    cell.classList.toggle('range-end',   d === state.rangeEnd);
  });
}

function renderMonth(year, month) {
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const firstDow = (new Date(year, month, 1).getDay() + 6) % 7; // Mon=0
  const name = `${MONTHS[month]} ${year}`;

  const headers = DOW.map(d => `<div class="dow-header">${d}</div>`).join('');
  const blanks  = Array(firstDow).fill('<div class="day-cell empty"></div>').join('');

  const days = [];
  for (let d = 1; d <= daysInMonth; d++) {
    const iso = `${year}-${String(month+1).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
    days.push(renderDay(iso, d));
  }

  return `
    <div class="month-block">
      <div class="month-name">${name}</div>
      <div class="month-grid">
        ${headers}
        ${blanks}
        ${days.join('')}
      </div>
    </div>
  `;
}

function renderDay(iso, dayNum) {
  const info = state.availDates[iso];
  const hasAny = info && (info.raw || info.bronze || info.silver || info.gold);

  const inRange  = state.rangeStart && state.rangeEnd && iso >= state.rangeStart && iso <= state.rangeEnd;
  const isStart  = iso === state.rangeStart;
  const isEnd    = iso === state.rangeEnd;

  const cls = [
    'day-cell',
    hasAny   ? 'has-data' : '',
    inRange  ? 'in-range' : '',
    isStart  ? 'range-start' : '',
    isEnd    ? 'range-end' : '',
  ].filter(Boolean).join(' ');

  const indicators = (info && hasAny) ? `
    <div class="day-indicators">
      <div class="day-ind ${info.raw    ? 'raw'    : 'off'}"></div>
      <div class="day-ind ${info.bronze ? 'bronze' : 'off'}"></div>
      <div class="day-ind ${info.silver ? 'silver' : 'off'}"></div>
      <div class="day-ind ${info.gold   ? 'gold'   : 'off'}"></div>
    </div>
  ` : '';

  const tooltip = info && hasAny ? buildTooltip(iso, info) : '';

  return `
    <div class="${cls}" data-date="${iso}">
      <span class="day-num">${dayNum}</span>
      ${indicators}
      ${tooltip}
    </div>
  `;
}

function buildTooltip(iso, info) {
  const rows = [
    info.raw    && `<div class="tt-row"><span class="tt-label">Raw</span><span class="tt-val">${fmtBytes(info.raw)}</span></div>`,
    info.bronze && `<div class="tt-row"><span class="tt-label">Bronze</span><span class="tt-val">${fmtBytes(info.bronze)}</span></div>`,
    info.silver && `<div class="tt-row"><span class="tt-label">Silver</span><span class="tt-val">${fmtBytes(info.silver)}</span></div>`,
    info.gold   && `<div class="tt-row"><span class="tt-label">Gold</span><span class="tt-val">${fmtBytes(info.gold)}</span></div>`,
  ].filter(Boolean).join('');
  return `<div class="day-tooltip"><div class="tt-date">${iso}</div>${rows}</div>`;
}

// ─── Day selection (click / drag) ─────────────────────────────────────────────

function attachDayHandlers() {
  const grid = document.getElementById('calendar-grid');

  grid.addEventListener('click', e => {
    const cell = e.target.closest('.day-cell');
    if (!cell || cell.classList.contains('empty')) return;
    const date = cell.dataset.date;

    if (!state.rangeStart) {
      // First click — set from date
      state.rangeStart = date;
      state.rangeEnd = date;
    } else if (date < state.rangeStart) {
      // Click before range — extend from date
      state.rangeStart = date;
    } else if (date > state.rangeEnd) {
      // Click after range — extend to date
      state.rangeEnd = date;
    } else {
      // Click within range — reset to this single day
      state.rangeStart = date;
      state.rangeEnd = date;
    }

    updateRangeUI();
    updateRangeHighlight();
    checkBronzeAvailable();
  });
}

function updateRangeUI() {
  document.getElementById('range-start').value = state.rangeStart || '';
  document.getElementById('range-end').value   = state.rangeEnd   || '';
  if (state.rangeStart && state.rangeEnd) {
    const days = Math.round(
      (new Date(state.rangeEnd) - new Date(state.rangeStart)) / 86400000
    ) + 1;
    document.getElementById('range-days').textContent = `${days} day${days === 1 ? '' : 's'} selected`;
  } else {
    document.getElementById('range-days').textContent = '';
  }
}

// ─── Navigation ───────────────────────────────────────────────────────────────

document.getElementById('cal-prev').addEventListener('click', () => {
  let m = state.viewMonth - 2;
  let y = state.viewYear;
  while (m < 0) { m += 12; y--; }
  state.viewMonth = m;
  state.viewYear = y;
  renderCalendar();
});

document.getElementById('cal-next').addEventListener('click', () => {
  let m = state.viewMonth + 2;
  let y = state.viewYear;
  while (m > 11) { m -= 12; y++; }
  state.viewMonth = m;
  state.viewYear = y;
  renderCalendar();
});

// Range inputs also update state
document.getElementById('range-start').addEventListener('change', e => {
  state.rangeStart = e.target.value;
  updateRangeUI();
  renderCalendar();
});
document.getElementById('range-end').addEventListener('change', e => {
  state.rangeEnd = e.target.value;
  updateRangeUI();
  renderCalendar();
  checkBronzeAvailable();
});

document.getElementById('dataset-mode-single').addEventListener('click', async () => {
  state.datasetCompareMode = false;
  renderDatasetCalendarControls();
  await refreshDatasetCalendar();
});

document.getElementById('dataset-mode-compare').addEventListener('click', async () => {
  state.datasetCompareMode = true;
  renderDatasetCalendarControls();
  await refreshDatasetCalendar();
});

document.getElementById('dataset-cal-prev').addEventListener('click', async () => {
  let m = state.datasetCalendarViewMonth - 2;
  let y = state.datasetCalendarViewYear;
  while (m < 0) { m += 12; y--; }
  state.datasetCalendarViewMonth = m;
  state.datasetCalendarViewYear = y;
  await refreshDatasetCalendar();
});

document.getElementById('dataset-cal-next').addEventListener('click', async () => {
  let m = state.datasetCalendarViewMonth + 2;
  let y = state.datasetCalendarViewYear;
  while (m > 11) { m -= 12; y++; }
  state.datasetCalendarViewMonth = m;
  state.datasetCalendarViewYear = y;
  await refreshDatasetCalendar();
});

// ─── Backtest envs ────────────────────────────────────────────────────────────

async function loadBacktestEnvs() {
  try {
    const envs = await apiFetch('/api/backtest/envs');
    const sel = document.getElementById('backtest-env');
    sel.innerHTML = '<option value="">Select env…</option>' +
      envs.map(e => `<option value="${e}">${e}</option>`).join('');
  } catch (e) {
    console.error('backtest envs error', e);
  }
}

function checkBronzeAvailable() {
  const btn = document.getElementById('btn-backtest');
  const status = document.getElementById('backtest-status');
  if (!state.rangeStart || !state.rangeEnd || !state.availDates) {
    btn.disabled = true;
    status.textContent = '';
    return;
  }
  // Collect all dates in range
  const dates = [];
  let d = new Date(state.rangeStart);
  const end = new Date(state.rangeEnd);
  while (d <= end) {
    dates.push(d.toISOString().slice(0, 10));
    d.setDate(d.getDate() + 1);
  }
  const missing = dates.filter(dt => !state.availDates[dt] || !state.availDates[dt].bronze);
  if (missing.length === 0) {
    btn.disabled = false;
    status.textContent = `Bronze OK (${dates.length} days)`;
    status.style.color = 'var(--accent)';
  } else {
    btn.disabled = true;
    status.textContent = `Missing bronze: ${missing.length} day(s)`;
    status.style.color = 'var(--red)';
  }
}

// ─── Operations ───────────────────────────────────────────────────────────────

function disableFor(btn, ms = 3000) {
  btn.disabled = true;
  setTimeout(() => { btn.disabled = false; checkBronzeAvailable(); }, ms);
}

document.getElementById('btn-download').addEventListener('click', async (e) => {
  if (!state.rangeStart || !state.rangeEnd) { alert('Select a date range first'); return; }
  const btn = e.currentTarget;
  disableFor(btn);
  const mkt = state.selectedMarket;
  const market = mkt ? `${mkt.exchange}:${mkt.symbol}` : '';
  try {
    const res = await apiFetch('/api/ops/download', {
      method: 'POST',
      body: JSON.stringify({ market, start: state.rangeStart, end: state.rangeEnd }),
    });
    await loadJobs();
    alert(`Download job started: ${res.job_id}`);
  } catch (e) {
    alert('Error: ' + e.message);
  }
});

document.getElementById('btn-populate').addEventListener('click', async (e) => {
  if (!state.rangeStart || !state.rangeEnd) { alert('Select a date range first'); return; }
  const mkt = state.selectedMarket;
  if (!mkt) { alert('Select a market first'); return; }
  const btn = e.currentTarget;
  disableFor(btn);
  const dataset = document.getElementById('populate-dataset').value;
  try {
    const res = await apiFetch('/api/ops/populate', {
      method: 'POST',
      body: JSON.stringify({
        dataset,
        exchange: mkt.exchange,
        symbol: mkt.symbol,
        start: state.rangeStart,
        end: state.rangeEnd,
      }),
    });
    await loadJobs();
    alert(`Populate job started: ${res.job_id}`);
  } catch (e) {
    alert('Error: ' + e.message);
  }
});

document.getElementById('btn-backtest').addEventListener('click', async (e) => {
  const env = document.getElementById('backtest-env').value;
  if (!env) { alert('Select a backtest env first'); return; }
  if (!state.rangeStart || !state.rangeEnd) { alert('Select a date range first'); return; }
  const btn = e.currentTarget;
  disableFor(btn);
  try {
    const res = await apiFetch('/api/ops/backtest', {
      method: 'POST',
      body: JSON.stringify({ env, start: state.rangeStart, end: state.rangeEnd, optimize: true }),
    });
    await loadJobs();
    document.getElementById('backtest-status').textContent = `Job started: ${res.job_id}`;
  } catch (e) {
    alert('Error: ' + e.message);
  }
});

// ─── Jobs ─────────────────────────────────────────────────────────────────────

async function loadJobs() {
  try {
    const jobs = await apiFetch('/api/jobs');
    state.jobs = jobs;
    renderJobs();
  } catch (e) {
    console.error('jobs error', e);
  }
}

function renderJobs() {
  const el = document.getElementById('jobs-list');
  if (state.jobs.length === 0) {
    el.innerHTML = '<div class="jobs-empty">No jobs yet</div>';
    return;
  }
  el.innerHTML = state.jobs.map(job => renderJobRow(job)).join('');
  el.querySelectorAll('.job-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const log = btn.closest('.job-wrap').querySelector('.job-log');
      log.classList.toggle('visible');
      btn.textContent = log.classList.contains('visible') ? 'hide' : 'log';
    });
  });
}

function renderJobRow(job) {
  const statusCls = `status-${job.status}`;
  const logLines = (job.log || []).join('\n');
  const elapsed = job.finished_at
    ? `${Math.round(job.finished_at - job.started_at)}s`
    : job.started_at ? fmtRelTime(job.started_at) : '';

  return `
    <div class="job-wrap">
      <div class="job-row ${statusCls}">
        <span class="job-status">${job.status}</span>
        <span class="job-label">${escHtml(job.label || job.cmd)}</span>
        <span class="job-id">#${job.id}</span>
        <span class="job-id">${elapsed}</span>
        ${logLines ? `<span class="job-toggle link-btn">log</span>` : ''}
      </div>
      ${logLines ? `<pre class="job-log">${escHtml(logLines)}</pre>` : ''}
    </div>
  `;
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escAttr(s) {
  return escHtml(String(s)).replace(/"/g, '&quot;');
}

document.getElementById('jobs-refresh').addEventListener('click', loadJobs);
document.getElementById('dataset-refresh').addEventListener('click', async () => {
  await loadDatasets();
  if (state.selectedDataset) {
    await refreshDatasetAvailability();
  }
});

// Viewport width indicator
const _vpEl = document.getElementById('viewport-indicator');
function updateVp() { _vpEl.textContent = window.innerWidth + 'px'; }
updateVp();
window.addEventListener('resize', updateVp);

// ─── Init ─────────────────────────────────────────────────────────────────────
boot();
