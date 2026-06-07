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
  availDates: {},      // { date: {raw, bronze, silver, gold} }
  availStats: null,
  selectedMarket: null,
  rangeStart: null,
  rangeEnd: null,
  selecting: false,
  viewYear: null,
  viewMonth: null,     // 0-11
  jobs: [],
};

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  loadGlobalStats();
  await loadMarkets();
  if (state.markets.length > 0) {
    const sel = document.getElementById('market-select');
    sel.value = state.markets[0].market;
    await selectMarket(state.markets[0]);
  }
  loadJobs();
  setInterval(loadJobs, 8000);
}

// ─── Global stats ─────────────────────────────────────────────────────────────

async function loadGlobalStats() {
  try {
    const stats = await apiFetch('/api/stats');
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

// ─── Availability ─────────────────────────────────────────────────────────────

async function selectMarket(mkt) {
  state.selectedMarket = mkt;
  document.getElementById('market-select').value = mkt.market;
  try {
    const data = await apiFetch(`/api/availability?exchange=${mkt.exchange}&symbol=${mkt.symbol}`);
    state.availDates = data.dates;
    state.availStats = data.stats;
    updateLegendCounts(data.stats);
    initCalendarView();
    renderCalendar();
  } catch (e) {
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

function initCalendarView() {
  const dates = Object.keys(state.availDates).sort();
  if (dates.length === 0) return;
  const last = dates[dates.length - 1].split('-');
  state.viewYear = parseInt(last[0]);
  state.viewMonth = parseInt(last[1]) - 1;
  // Back up one so we show the last 3 months centered
  let m = state.viewMonth - 2;
  let y = state.viewYear;
  if (m < 0) { m += 12; y--; }
  state.viewYear = y;
  state.viewMonth = m;
}

function renderCalendar() {
  const label = document.getElementById('cal-month-label');
  const grid  = document.getElementById('calendar-grid');

  // Show 3 months starting from viewYear/viewMonth
  const months = [];
  let y = state.viewYear, m = state.viewMonth;
  for (let i = 0; i < 3; i++) {
    months.push({ year: y, month: m });
    m++;
    if (m > 11) { m = 0; y++; }
  }

  const first = months[0], last = months[months.length - 1];
  label.textContent = `${MONTHS[first.month]} ${first.year}  ·  ${MONTHS[last.month]} ${last.year}`;

  grid.innerHTML = months.map(({ year, month }) => renderMonth(year, month)).join('');
  attachDayHandlers();
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

  grid.addEventListener('mousedown', e => {
    const cell = e.target.closest('.day-cell');
    if (!cell || cell.classList.contains('empty')) return;
    const date = cell.dataset.date;
    state.selecting = true;
    state.rangeStart = date;
    state.rangeEnd = date;
    updateRangeUI();
    renderCalendar();
  });

  grid.addEventListener('mouseover', e => {
    if (!state.selecting) return;
    const cell = e.target.closest('.day-cell');
    if (!cell || cell.classList.contains('empty')) return;
    const date = cell.dataset.date;
    if (date < state.rangeStart) {
      state.rangeEnd = state.rangeStart;
      state.rangeStart = date;
    } else {
      state.rangeEnd = date;
    }
    updateRangeUI();
    renderCalendar();
  });

  document.addEventListener('mouseup', () => { state.selecting = false; });
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
  let m = state.viewMonth - 3;
  let y = state.viewYear;
  while (m < 0) { m += 12; y--; }
  state.viewMonth = m;
  state.viewYear = y;
  renderCalendar();
});

document.getElementById('cal-next').addEventListener('click', () => {
  let m = state.viewMonth + 3;
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
});

// ─── Operations ───────────────────────────────────────────────────────────────

document.getElementById('btn-download').addEventListener('click', async () => {
  if (!state.rangeStart || !state.rangeEnd) { alert('Select a date range first'); return; }
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

document.getElementById('btn-populate').addEventListener('click', async () => {
  if (!state.rangeStart || !state.rangeEnd) { alert('Select a date range first'); return; }
  const mkt = state.selectedMarket;
  if (!mkt) { alert('Select a market first'); return; }
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

document.getElementById('jobs-refresh').addEventListener('click', loadJobs);

// ─── Init ─────────────────────────────────────────────────────────────────────
boot();
