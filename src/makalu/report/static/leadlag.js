// Lead-lag report page. Reads report.json, daily.csv and hourly.csv next to the page and draws everything from them.
(() => {
  const ACCENT = 'var(--accent)', MUTED = 'var(--series-muted)';
  const num = (v) => (v === '' || v == null ? NaN : Number(v));
  const median = (xs) => d3.median(xs.filter((x) => !isNaN(x)));
  const parseDay = (s) => new Date(s + 'T00:00:00Z');
  const fmtChg = (v) => (isNaN(v) ? '—' : Math.abs(v) < 0.5 ? '±0%' : d3.format('+.0f')(v) + '%');
  const compact = (v) => Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(v);

  let meta, daily, hourly, weekHourRows, cmp, basis, dowRange = 'all';
  // What sections 1–4 are measured on: whole JST days, or JST hours (each hour measured on its own).
  const BASES = {
    d: { label: 'Per day', unit: 'day', units: 'days', adj: 'daily' },
    h: { label: 'Per hour', unit: 'hour', units: 'hours', adj: 'hourly' },
  };
  const B = () => ({ ...BASES[basis], rows: basis === 'h' ? hourly : daily });
  // An hour's peak lag sits on the 100ms grid, so a median of hours only flips between two steps; average it instead.
  const agg = (m) => (basis === 'h' && m === 'peak_lag_ms' ? (xs) => d3.mean(xs) : median);
  const exch = (m) => m.split(':')[0];

  // Metrics that are lead-lag strength: a fall there is bad for the strategy. The rest are context.
  const STRENGTH = new Set(['rho1_1s', 'beta_1s_cum', 'rho0_1s', 'peak_rho']);
  const FMT = {
    peak_lag_ms: d3.format(',.0f'), rho0_1s: d3.format('.3f'), rho1_1s: d3.format('.3f'), beta_1s_cum: d3.format('.2f'),
    peak_rho: d3.format('.3f'), vol: d3.format(',.0f'), vol_lead_bps_d: d3.format(',.0f'), lead_tick_bps: d3.format('.2~r'),
    lag_spread_bps: d3.format('.2f'), lead_updates: compact, days: d3.format(','), dropped_rows: d3.format(','),
  };
  const fmt = (m, v) => (v == null || isNaN(v) ? '—' : (FMT[m] || d3.format('.3g'))(v));

  function inPeriod(day, side) { const [, from, to] = cmp[side]; return day >= from && day <= to; }

  // ── derived tables ──────────────────────────────────────────────────────────
  function compare() {
    const rows = B().rows;
    const a = rows.filter((r) => inPeriod(r.date, 'a')), b = rows.filter((r) => inPeriod(r.date, 'b'));
    const out = {};
    for (const m of Object.keys(FMT)) {
      if (m === 'days') continue;
      const ma = agg(m)(a.map((r) => r[m])), mb = agg(m)(b.map((r) => r[m]));
      out[m] = { a: ma, b: mb, chg: (mb / ma - 1) * 100 };
    }
    out.n = { a: a.length, b: b.length };
    return out;
  }
  function weekly() {
    const weekOf = (d) => d3.utcFormat('%Y-%m-%d')(d3.utcMonday.floor(parseDay(d)));
    return Array.from(d3.group(B().rows, (r) => weekOf(r.date)), ([week, rows]) => ({
      week, d: parseDay(week), n: rows.length,
      rho1_1s: median(rows.map((r) => r.rho1_1s)), peak_lag_ms: agg('peak_lag_ms')(rows.map((r) => r.peak_lag_ms)),
      vol: median(rows.map((r) => r.vol)), lead_tick_bps: median(rows.map((r) => r.lead_tick_bps)),
    })).sort((x, y) => d3.ascending(x.week, y.week));
  }
  function curve() {
    const out = [];
    for (const side of ['a', 'b']) {
      const rows = B().rows.filter((r) => inPeriod(r.date, side));
      for (const lag of meta.curve_lags_ms) {
        const col = lag < 0 ? `rho_lag_m${-lag}` : `rho_lag_${lag}`;
        out.push({ side, lag_ms: lag, rho: d3.mean(rows, (r) => r[col]) });
      }
    }
    return out;
  }
  const MONTH_COLS = ['days', 'peak_lag_ms', 'rho0_1s', 'rho1_1s', 'beta_1s_cum', 'peak_rho', 'vol_lead_bps_d', 'lead_tick_bps', 'lead_updates', 'lag_spread_bps', 'dropped_rows'];
  function monthly() {
    return Array.from(d3.group(daily, (r) => r.date.slice(0, 7)), ([month, rows]) => {
      const o = { month, days: rows.length };
      for (const c of MONTH_COLS.slice(1, -1)) o[c] = median(rows.map((r) => r[c]));
      o.dropped_rows = d3.sum(rows, (r) => (r.lead_dropped_rows || 0) + (r.lag_dropped_rows || 0));
      return o;
    }).sort((x, y) => d3.ascending(x.month, y.month));
  }

  const HOUR_TICKS = [0, 3, 6, 9, 12, 15, 18, 21];
  function hourProfile() {
    const out = [];
    for (const side of ['a', 'b']) {
      const byHour = d3.group(hourly.filter((r) => inPeriod(r.date, side)), (r) => r.hour);
      for (let hour = 0; hour < 24; hour++) {
        const rows = byHour.get(hour) || [];
        out.push({ side, hour, hours: rows.length, rho1_1s: median(rows.map((r) => r.rho1_1s)),
          // an hour's peak lag sits on the 100ms grid, so a median only flips between two steps; the mean shows the drift
          peak_lag_ms: d3.mean(rows, (r) => r.peak_lag_ms), vol_lead_bps_h: median(rows.map((r) => r.vol_lead_bps_h)) });
      }
    }
    return out;
  }
  function volBins() {
    // Deciles of hourly lead volatility over every hour in the report, so both periods share the same bins.
    const vols = hourly.map((r) => r.vol_lead_bps_h).filter((v) => !isNaN(v)).sort(d3.ascending);
    const edges = d3.range(1, 10).map((q) => d3.quantileSorted(vols, q / 10));
    const bin = (v) => d3.bisectLeft(edges, v);
    const out = [];
    for (const side of ['a', 'b']) {
      const groups = d3.group(hourly.filter((r) => inPeriod(r.date, side) && !isNaN(r.vol_lead_bps_h)), (r) => bin(r.vol_lead_bps_h));
      for (const [b, rows] of groups) {
        if (rows.length < 5) continue;
        out.push({ side, bin: b + 1, hours: rows.length, vol: median(rows.map((r) => r.vol_lead_bps_h)), rho1_1s: median(rows.map((r) => r.rho1_1s)) });
      }
    }
    return out.sort((x, y) => d3.ascending(x.side, y.side) || d3.ascending(x.bin, y.bin));
  }
  function weekHour() {
    const weekOf = (d) => d3.utcFormat('%Y-%m-%d')(d3.utcMonday.floor(parseDay(d)));
    return Array.from(d3.rollup(hourly, (rows) => ({ hours: rows.length, rho1_1s: median(rows.map((r) => r.rho1_1s)) }),
      (r) => weekOf(r.date), (r) => r.hour), ([week, byHour]) => Array.from(byHour, ([hour, v]) => ({ week, d: parseDay(week), hour, ...v }))).flat()
      .filter((r) => !isNaN(r.rho1_1s));
  }

  // ── chart helpers ───────────────────────────────────────────────────────────
  const clean = (g) => g.attr('font-size', null).attr('font-family', null)
    .call((s) => s.selectAll('text').style('fill', 'var(--muted)').style('font-size', 'var(--fs-caption)'))
    .call((s) => s.select('.domain').style('stroke', 'var(--axis)'))
    .call((s) => s.selectAll('.tick line').style('stroke', 'var(--grid)'));
  const measure = (svg, texts) => { const t = svg.append('text').style('font-size', 'var(--fs-caption)'); let m = 0; for (const s of texts) { t.text(s); m = Math.max(m, t.node().getComputedTextLength()); } t.remove(); return m; };
  const tipOf = (el) => { let t = el.querySelector('.tip'); if (!t) { t = document.createElement('div'); t.className = 'tip'; t.hidden = true; el.appendChild(t); } return t; };
  const placeTip = (el, tip, x, y) => { tip.hidden = false; tip.style.left = Math.max(0, Math.min(el.clientWidth - tip.offsetWidth, x + 12)) + 'px'; tip.style.top = Math.max(0, y - tip.offsetHeight - 8) + 'px'; };
  const tipLines = (tip, head, lines) => { tip.replaceChildren(); const d = document.createElement('div'); d.className = 'd'; d.textContent = head; tip.appendChild(d); for (const [color, text] of lines) { const l = document.createElement('div'); if (color) { const sw = document.createElement('i'); sw.className = 'sw'; sw.style.background = color; sw.style.marginRight = '6px'; l.appendChild(sw); } l.appendChild(document.createTextNode(text)); tip.appendChild(l); } };
  function legend(id, items) { const el = document.getElementById(id); el.replaceChildren(); for (const [c, n] of items) { const s = document.createElement('span'); const sw = document.createElement('i'); sw.className = 'sw'; sw.style.background = c; s.append(sw, document.createTextNode(n)); el.appendChild(s); } }
  const svgIn = (el, W, H, label) => d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img').attr('aria-label', label);
  const hoverRect = (svg, x0, y0, w, h) => svg.append('rect').attr('x', x0).attr('y', y0).attr('width', w).attr('height', h).style('fill', 'transparent');

  // ── sections ────────────────────────────────────────────────────────────────
  function header(c) {
    const lead = exch(meta.lead), lag = exch(meta.lag);
    const take = document.getElementById('takeaway'); take.replaceChildren();
    const b = (t) => { const e = document.createElement('b'); e.textContent = t; return e; };
    take.append(`${lag} follows ${lead} about `, b(fmt('peak_lag_ms', c.peak_lag_ms.b) + ' ms'), ` later in ${cmp.b[0]} (`,
      fmt('peak_lag_ms', c.peak_lag_ms.a) + ` ms in ${cmp.a[0]}). How strongly it follows one second later moved `, b(fmtChg(c.rho1_1s.chg)),
      `, pass-through within 3s moved `, b(fmtChg(c.beta_1s_cum.chg)), `, and ${lead} volatility moved `, b(fmtChg(c.vol.chg)), '.');
    const { units, adj } = B();
    document.getElementById('period-label').textContent =
      `${meta.lead} leads, ${meta.lag} lags. ${cmp.a[0]} (${d3.format(',')(c.n.a)} ${units}) against ${cmp.b[0]} (${d3.format(',')(c.n.b)} ${units}); sections 1–4 are ${adj} values from silver MarketPrice, execution size ${meta.execution_size}.`;
  }

  function tiles(c) {
    const lead = exch(meta.lead);
    const defs = [
      ['peak_lag_ms', basis === 'h' ? 'Peak lag, ms (100ms grid, mean)' : 'Peak lag, ms (100ms grid)'], ['rho1_1s', 'Delayed response, ρ next second'],
      ['beta_1s_cum', 'Pass-through within 3s (β)'], ['vol', `${lead} volatility, bps/${B().unit}`], ['lead_tick_bps', `${lead} tick size, bps`],
    ];
    const el = document.getElementById('tiles'); el.replaceChildren();
    defs.forEach(([m, label], i) => {
      const t = document.createElement('div'); t.className = 'tile'; t.id = `s1-${i + 1}`;
      const l = document.createElement('span'); l.className = 'tile-label';
      const n = document.createElement('span'); n.className = 'sec'; n.textContent = `1.${i + 1}`; l.append(n, label);
      const row = document.createElement('div'); row.className = 'tile-row';
      const big = document.createElement('span'); big.className = 'big'; big.textContent = fmt(m, c[m].b);
      const from = document.createElement('span'); from.className = 'from'; from.textContent = `from ${fmt(m, c[m].a)}`;
      row.append(big, from);
      const cr = document.createElement('span'); cr.className = 'chg-row';
      const chg = document.createElement('span'); chg.className = 'chg ' + (STRENGTH.has(m) ? (c[m].chg <= -5 ? 'bad' : c[m].chg >= 5 ? 'ok' : 'flat') : 'flat');
      chg.textContent = fmtChg(c[m].chg);
      cr.append(chg, ` vs ${cmp.a[0]}`);
      t.append(l, row, cr); el.appendChild(t);
    });
    document.getElementById('s1-note').textContent = basis === 'h'
      ? `Median of hourly values in ${cmp.b[0]} (peak lag: mean), with ${cmp.a[0]} after "from".`
      : `Median of daily values in ${cmp.b[0]}, with ${cmp.a[0]} after "from".`;
  }

  function multiples() {
    const el = document.getElementById('multi'); el.replaceChildren();
    const rows = weekly(), lead = exch(meta.lead);
    legend('band-legend', [[MUTED, cmp.a[0]], [ACCENT, cmp.b[0]]]);
    document.getElementById('s2-basis').textContent = basis === 'h' ? 'Weekly medians of hourly values (peak lag: mean).' : 'Weekly medians of daily values.';
    const panels = [
      { f: 'rho1_1s', name: 'Delayed response, ρ next second', color: ACCENT, fmt: d3.format('.2f') },
      { f: 'peak_lag_ms', name: basis === 'h' ? 'Peak lag, ms (mean)' : 'Peak lag, ms', color: 'var(--series-2)', fmt: d3.format(',.0f') },
      { f: 'vol', name: `${lead} volatility, bps/${B().unit}`, color: 'var(--series-3)', fmt: d3.format(',.0f') },
      { f: 'lead_tick_bps', name: `${lead} tick size, bps`, color: 'var(--series-4)', fmt: d3.format('.2f') },
    ];
    const W = el.clientWidth, ph = 118, gap = 24, top = 18, bottom = 26;
    const H = top + panels.length * ph + (panels.length - 1) * gap + bottom;
    const svg = svgIn(el, W, H, 'Weekly lead-lag strength, peak lag, volatility and tick size');
    const ys = panels.map((p) => d3.scaleLinear().domain(d3.extent(rows, (r) => r[p.f])).nice(4));
    const left = 8 + d3.max(panels.map((p, i) => measure(svg, ys[i].ticks(4).map(p.fmt)))), right = 12;
    const x = d3.scaleUtc(d3.extent(rows, (r) => r.d), [left, W - right]);
    panels.forEach((p, i) => {
      const y0 = top + i * (ph + gap); const y = ys[i].range([y0 + ph, y0]); p.y = y;
      for (const [side, color, op] of [['a', MUTED, 0.22], ['b', ACCENT, 0.12]]) {
        const lo = Math.max(+parseDay(cmp[side][1]), +x.domain()[0]), hi = Math.min(+parseDay(cmp[side][2]), +x.domain()[1]);
        if (hi > lo) svg.append('rect').attr('x', x(lo)).attr('width', x(hi) - x(lo)).attr('y', y0).attr('height', ph).style('fill', color).style('fill-opacity', op);
      }
      svg.append('g').attr('transform', `translate(${left},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(p.fmt).tickSize(-(W - left - right))).call(clean).call((s) => s.select('.domain').remove());
      svg.append('text').attr('x', left).attr('y', y0 - 6).text(`2.${i + 1}  ${p.name}`).style('fill', 'var(--fg)').style('font-size', 'var(--fs-caption)').style('font-weight', 600);
      svg.append('path').datum(rows).style('fill', 'none').style('stroke', p.color).attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round')
        .attr('d', d3.line().defined((r) => !isNaN(r[p.f])).x((r) => x(r.d)).y((r) => y(r[p.f])));
    });
    svg.append('g').attr('transform', `translate(0,${H - bottom})`).call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(W / 110))).tickFormat(d3.utcFormat('%b %Y')).tickSizeOuter(0)).call(clean);
    const rule = svg.append('line').attr('y1', top).attr('y2', H - bottom).style('stroke', 'var(--ref)').attr('visibility', 'hidden');
    const dots = panels.map((p) => svg.append('circle').attr('r', 3.5).style('fill', p.color).attr('visibility', 'hidden'));
    const tip = tipOf(el), bis = d3.bisector((r) => r.d).center;
    hoverRect(svg, left, top, W - left - right, H - top - bottom)
      .on('pointermove', (ev) => {
        const [mx, my] = d3.pointer(ev); const r = rows[bis(rows, x.invert(mx))]; if (!r) return;
        rule.attr('x1', x(r.d)).attr('x2', x(r.d)).attr('visibility', 'visible');
        panels.forEach((p, i) => dots[i].attr('cx', x(r.d)).attr('cy', p.y(r[p.f])).attr('visibility', isNaN(r[p.f]) ? 'hidden' : 'visible'));
        tipLines(tip, `Week of ${d3.utcFormat('%b %-d, %Y')(r.d)} · ${d3.format(',')(r.n)} ${B().units}`, panels.map((p) => [p.color, `${p.name}: ${p.fmt(r[p.f])}`]));
        placeTip(el, tip, x(r.d), my);
      })
      .on('pointerleave', () => { tip.hidden = true; rule.attr('visibility', 'hidden'); dots.forEach((d) => d.attr('visibility', 'hidden')); });
  }

  function lagCurve() {
    const el = document.getElementById('curve'); el.replaceChildren();
    const rows = curve(), series = [['a', MUTED, cmp.a[0]], ['b', ACCENT, cmp.b[0]]];
    document.getElementById('curve-note').textContent =
      `Average ${B().adj} correlation of ${exch(meta.lag)}'s 100ms return with ${exch(meta.lead)}'s return k ms earlier. Where the peak sits is the lag; how high it sits is the strength.`;
    legend('curve-legend', series.map(([, c, n]) => [c, n]));
    const W = el.clientWidth, H = 240, top = 12, bottom = 34;
    const svg = svgIn(el, W, H, 'Cross-correlation by lag for both periods');
    const y = d3.scaleLinear([Math.min(0, d3.min(rows, (r) => r.rho)), d3.max(rows, (r) => r.rho)], [H - bottom, top]).nice(5);
    const left = 8 + measure(svg, y.ticks(5).map(d3.format('.2f')));
    const right = 4 + measure(svg, [d3.format(',')(d3.max(meta.curve_lags_ms))]) / 2;
    const x = d3.scaleLinear(d3.extent(meta.curve_lags_ms), [left, W - right]);
    svg.append('g').attr('transform', `translate(${left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')).tickSize(-(W - left - right))).call(clean).call((s) => s.select('.domain').remove());
    svg.append('g').attr('transform', `translate(0,${H - bottom})`).call(d3.axisBottom(x).ticks(Math.max(4, Math.floor(W / 80))).tickFormat(d3.format(',')).tickSizeOuter(0)).call(clean);
    svg.append('text').attr('x', W - right).attr('y', H - 4).attr('text-anchor', 'end').text(`lag, ms (${exch(meta.lead)} first)`).style('fill', 'var(--muted)').style('font-size', 'var(--fs-caption)');
    svg.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', top).attr('y2', H - bottom).style('stroke', 'var(--ref)').attr('stroke-dasharray', '3 3');
    for (const [side, color] of series) {
      const s = rows.filter((r) => r.side === side && !isNaN(r.rho));
      svg.append('path').datum(s).style('fill', 'none').style('stroke', color).attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('d', d3.line().x((r) => x(r.lag_ms)).y((r) => y(r.rho)));
      const pk = d3.greatest(s, (r) => r.rho);
      if (pk) svg.append('circle').attr('cx', x(pk.lag_ms)).attr('cy', y(pk.rho)).attr('r', 3.5).style('fill', color);
    }
    const tip = tipOf(el);
    hoverRect(svg, left, top, W - left - right, H - top - bottom)
      .on('pointermove', (ev) => {
        const [mx, my] = d3.pointer(ev); const lag = Math.round(x.invert(mx) / 100) * 100;
        tipLines(tip, `Lag ${d3.format(',')(lag)} ms`, series.map(([side, color, name]) => { const r = rows.find((q) => q.side === side && q.lag_ms === lag); return [color, `${name}: ${r ? d3.format('.3f')(r.rho) : '—'}`]; }));
        placeTip(el, tip, mx, my);
      })
      .on('pointerleave', () => { tip.hidden = true; });
  }

  function scatter() {
    const el = document.getElementById('scatter'); el.replaceChildren();
    const side = (d) => (inPeriod(d, 'a') ? 'a' : inPeriod(d, 'b') ? 'b' : null);
    const { unit, units } = B(), perHour = basis === 'h';
    let rows = B().rows.map((r) => ({ date: r.date, hour: r.hour, v: r.vol, y: r.rho1_1s, s: side(r.date) })).filter((r) => r.v > 0 && !isNaN(r.y));
    // A handful of hours (a crash, a stalled feed) sit far outside the rest and would squash the cloud into a corner.
    let outside = 0;
    if (perHour) {
      const vs = rows.map((r) => r.v).sort(d3.ascending), ys = rows.map((r) => r.y).sort(d3.ascending);
      const [v0, v1, y0, y1] = [d3.quantileSorted(vs, 0.002), d3.quantileSorted(vs, 0.998), d3.quantileSorted(ys, 0.002), d3.quantileSorted(ys, 0.998)];
      const kept = rows.filter((r) => r.v >= v0 && r.v <= v1 && r.y >= y0 && r.y <= y1);
      outside = rows.length - kept.length; rows = kept;
    }
    legend('scatter-legend', [[MUTED, cmp.a[0]], [ACCENT, cmp.b[0]], ['var(--grid)', `Other ${units}`]]);
    document.getElementById('s4-unit').textContent = unit;
    document.getElementById('s4-basis').textContent = `Each dot is one ${unit}${outside ? ` (${d3.format(',')(outside)} extreme hours outside the axes are not drawn)` : ''}.`;
    const W = el.clientWidth, H = 240, top = 12, bottom = 34;
    const svg = svgIn(el, W, H, `Delayed-response correlation against lead volatility, per ${unit}`);
    const y = d3.scaleLinear(d3.extent(rows, (r) => r.y), [H - bottom, top]).nice(5);
    const left = 8 + measure(svg, y.ticks(5).map(d3.format('.2f')));
    const xd = d3.scaleLog().domain(d3.extent(rows, (r) => r.v)).nice().domain();
    const right = 4 + measure(svg, [d3.format(',.0f')(xd[1])]) / 2;
    const x = d3.scaleLog(xd, [left, W - right]);
    svg.append('g').attr('transform', `translate(${left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')).tickSize(-(W - left - right))).call(clean).call((s) => s.select('.domain').remove());
    svg.append('g').attr('transform', `translate(0,${H - bottom})`).call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(W / 90)), ',.0f').tickSizeOuter(0)).call(clean);
    svg.append('text').attr('x', W - right).attr('y', H - 4).attr('text-anchor', 'end').text(`${exch(meta.lead)} volatility, bps/${unit} (log)`).style('fill', 'var(--muted)').style('font-size', 'var(--fs-caption)');
    const order = (s) => (s === 'b' ? 2 : s === 'a' ? 1 : 0);
    const color = (s) => (s === 'a' ? MUTED : s === 'b' ? ACCENT : 'var(--axis)');
    const sorted = rows.slice().sort((m, n) => order(m.s) - order(n.s));
    svg.append('g').selectAll('circle').data(sorted).join('circle')
      .attr('cx', (r) => x(r.v)).attr('cy', (r) => y(r.y)).attr('r', (r) => (perHour ? (r.s ? 1.8 : 1.4) : r.s ? 3.2 : 2.4))
      .style('fill', (r) => color(r.s)).style('fill-opacity', (r) => (perHour ? (r.s ? 0.45 : 0.25) : r.s ? 0.85 : 0.45));
    const pts = sorted.map((r) => [x(r.v), y(r.y)]), del = d3.Delaunay.from(pts), tip = tipOf(el);
    const hi = svg.append('circle').attr('r', 5).style('fill', 'none').style('stroke', 'var(--fg)').attr('visibility', 'hidden');
    hoverRect(svg, left, top, W - left - right, H - top - bottom)
      .on('pointermove', (ev) => {
        const [mx, my] = d3.pointer(ev); const i = del.find(mx, my); const r = sorted[i]; if (!r) return;
        hi.attr('cx', pts[i][0]).attr('cy', pts[i][1]).attr('visibility', 'visible');
        tipLines(tip, d3.utcFormat('%a %b %-d, %Y')(parseDay(r.date)) + (perHour ? ` · ${r.hour}:00 JST` : ''), [[null, `ρ next second: ${d3.format('.3f')(r.y)}`], [null, `${exch(meta.lead)} vol: ${d3.format(',.0f')(r.v)} bps/${unit}`]]);
        placeTip(el, tip, pts[i][0], pts[i][1]);
      })
      .on('pointerleave', () => { tip.hidden = true; hi.attr('visibility', 'hidden'); });
  }

  function table() {
    const lead = exch(meta.lead), lag = exch(meta.lag);
    const heads = ['Month', 'Days', 'Peak lag ms', 'ρ same sec', 'ρ next sec', 'β 0–3s', 'Peak ρ 100ms', `${lead} vol bps`, 'Tick bps', `${lead} updates`, `${lag} spread bps`, 'Snapshots dropped'];
    const hr = document.querySelector('#monthly thead tr'); hr.replaceChildren(...heads.map((h) => { const th = document.createElement('th'); th.textContent = h; return th; }));
    const tb = document.querySelector('#monthly tbody'); tb.replaceChildren();
    for (const r of monthly()) {
      const tr = document.createElement('tr');
      const first = r.month + '-01', last = r.month + '-31';
      if (first >= cmp.a[1].slice(0, 8) + '01' && last <= cmp.a[2].slice(0, 8) + '31') tr.className = 'in-a';
      if (first >= cmp.b[1].slice(0, 8) + '01' && last <= cmp.b[2].slice(0, 8) + '31') tr.className = 'in-b';
      const td0 = document.createElement('td'); td0.textContent = r.month; tr.appendChild(td0);
      for (const c of MONTH_COLS) { const td = document.createElement('td'); td.textContent = fmt(c, r[c]); tr.appendChild(td); }
      tb.appendChild(tr);
    }
  }

  function hours() {
    const rows = hourProfile(), lead = exch(meta.lead);
    const series = [['a', MUTED, cmp.a[0]], ['b', ACCENT, cmp.b[0]]];
    legend('hour-legend', series.map(([, c, n]) => [c, n]));
    const panels = [
      { f: 'rho1_1s', name: 'Delayed response, ρ next second', fmt: d3.format('.2f') },
      { f: 'peak_lag_ms', name: 'Peak lag, ms (mean)', fmt: d3.format(',.0f') },
      { f: 'vol_lead_bps_h', name: `${lead} volatility, bps/hour`, fmt: d3.format(',.0f') },
    ];
    const box = document.getElementById('hour-profile');
    // all panels in the grid before any is measured, or the first one is sized as if it were alone
    const els = panels.map((p, i) => { const el = document.createElement('div'); el.id = `s6-${i + 1}`; return el; });
    box.replaceChildren(...els);
    panels.forEach((p, i) => {
      const el = els[i];
      const pr = rows.filter((r) => !isNaN(r[p.f]));
      const W = el.clientWidth, H = 180, top = 22, bottom = 26;
      const svg = svgIn(el, W, H, `${p.name} by JST hour, both periods`);
      const y = d3.scaleLinear(d3.extent(pr, (r) => r[p.f]), [H - bottom, top]).nice(4);
      const left = 8 + measure(svg, y.ticks(4).map(p.fmt)), right = 4 + measure(svg, ['21:00']) / 2;
      const x = d3.scaleLinear([0, 23], [left, W - right]);
      svg.append('text').attr('x', left).attr('y', 12).text(`6.${i + 1}  ${p.name}`).style('fill', 'var(--fg)').style('font-size', 'var(--fs-caption)').style('font-weight', 600);
      svg.append('g').attr('transform', `translate(${left},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(p.fmt).tickSize(-(W - left - right))).call(clean).call((s) => s.select('.domain').remove());
      svg.append('g').attr('transform', `translate(0,${H - bottom})`).call(d3.axisBottom(x).tickValues(HOUR_TICKS).tickFormat((h) => `${h}:00`).tickSizeOuter(0)).call(clean);
      for (const [side, color] of series) {
        svg.append('path').datum(pr.filter((r) => r.side === side)).style('fill', 'none').style('stroke', color).attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round')
          .attr('d', d3.line().x((r) => x(r.hour)).y((r) => y(r[p.f])));
      }
      const rule = svg.append('line').attr('y1', top).attr('y2', H - bottom).style('stroke', 'var(--ref)').attr('visibility', 'hidden');
      const tip = tipOf(el);
      hoverRect(svg, left, top, W - left - right, H - top - bottom)
        .on('pointermove', (ev) => {
          const [mx, my] = d3.pointer(ev); const hour = Math.max(0, Math.min(23, Math.round(x.invert(mx))));
          rule.attr('x1', x(hour)).attr('x2', x(hour)).attr('visibility', 'visible');
          tipLines(tip, `${hour}:00–${hour + 1}:00 JST`, series.map(([side, color, name]) => { const r = pr.find((q) => q.side === side && q.hour === hour); return [color, `${name}: ${r ? `${p.fmt(r[p.f])} · ${r.hours} h` : '—'}`]; }));
          placeTip(el, tip, x(hour), my);
        })
        .on('pointerleave', () => { tip.hidden = true; rule.attr('visibility', 'hidden'); });
    });
  }

  function volBinChart() {
    const el = document.getElementById('volbin'); el.replaceChildren();
    const rows = volBins(), lead = exch(meta.lead), series = [['a', MUTED, cmp.a[0]], ['b', ACCENT, cmp.b[0]]];
    document.getElementById('volbin-note').textContent =
      `Hours split into ten bins by ${lead} volatility in that hour, with the same bins for both periods; median ρ next second in each. If the later line sits below at the same volatility, a quieter market does not explain the change.`;
    legend('volbin-legend', series.map(([, c, n]) => [c, n]));
    const W = el.clientWidth, H = 240, top = 12, bottom = 34;
    const svg = svgIn(el, W, H, 'Median delayed-response correlation by hourly volatility bin, both periods');
    const y = d3.scaleLinear(d3.extent(rows, (r) => r.rho1_1s), [H - bottom, top]).nice(5);
    const left = 8 + measure(svg, y.ticks(5).map(d3.format('.2f')));
    const [v0, v1] = d3.extent(rows, (r) => r.vol);
    const right = 4 + measure(svg, [d3.format(',.0f')(v1)]) / 2;
    const x = d3.scaleLog([v0 / 1.08, v1 * 1.08], [left, W - right]);
    svg.append('g').attr('transform', `translate(${left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')).tickSize(-(W - left - right))).call(clean).call((s) => s.select('.domain').remove());
    svg.append('g').attr('transform', `translate(0,${H - bottom})`).call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(W / 90)), ',.0f').tickSizeOuter(0)).call(clean);
    svg.append('text').attr('x', W - right).attr('y', H - 4).attr('text-anchor', 'end').text(`${lead} volatility, bps/hour (log)`).style('fill', 'var(--muted)').style('font-size', 'var(--fs-caption)');
    const pts = [];
    for (const [side, color, name] of series) {
      const s = rows.filter((r) => r.side === side);
      svg.append('path').datum(s).style('fill', 'none').style('stroke', color).attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('d', d3.line().x((r) => x(r.vol)).y((r) => y(r.rho1_1s)));
      svg.append('g').selectAll('circle').data(s).join('circle').attr('cx', (r) => x(r.vol)).attr('cy', (r) => y(r.rho1_1s)).attr('r', 3.5).style('fill', color);
      for (const r of s) pts.push({ r, name, px: x(r.vol), py: y(r.rho1_1s) });
    }
    const del = d3.Delaunay.from(pts, (q) => q.px, (q) => q.py), tip = tipOf(el);
    const hi = svg.append('circle').attr('r', 6).style('fill', 'none').style('stroke', 'var(--fg)').attr('visibility', 'hidden');
    hoverRect(svg, left, top, W - left - right, H - top - bottom)
      .on('pointermove', (ev) => {
        const [mx, my] = d3.pointer(ev); const q = pts[del.find(mx, my)]; if (!q) return;
        hi.attr('cx', q.px).attr('cy', q.py).attr('visibility', 'visible');
        tipLines(tip, `${q.name} · volatility bin ${q.r.bin} of 10 · ${d3.format(',')(q.r.hours)} hours`, [[null, `ρ next second: ${d3.format('.3f')(q.r.rho1_1s)}`], [null, `${lead} vol: ${d3.format(',.0f')(q.r.vol)} bps/hour`]]);
        placeTip(el, tip, q.px, q.py);
      })
      .on('pointerleave', () => { tip.hidden = true; hi.attr('visibility', 'hidden'); });
  }

  function heat() {
    const el = document.getElementById('heat'); el.replaceChildren();
    const rows = weekHourRows;
    const vals = rows.map((r) => r.rho1_1s).sort(d3.ascending);
    const lo = d3.quantileSorted(vals, 0.02), hiV = d3.quantileSorted(vals, 0.98);
    const op = d3.scaleLinear([lo, hiV], [0.06, 1]).clamp(true);
    const lg = document.getElementById('heat-legend'); lg.replaceChildren();
    const ramp = document.createElement('span'); ramp.className = 'ramp';
    ramp.append(d3.format('.2f')(lo), document.createElement('i'), d3.format('.2f')(hiV));
    lg.appendChild(ramp);
    const W = el.clientWidth, top = 6, cellH = 11, bottom = 40, H = top + 24 * cellH + bottom;
    const svg = svgIn(el, W, H, 'Delayed-response correlation by week and JST hour');
    const left = 8 + measure(svg, ['21:00']), right = 8;
    const weeks = Array.from(new Set(rows.map((r) => r.week))).sort();
    const x = d3.scaleBand(weeks, [left, W - right]), yb = d3.scaleBand(d3.range(24), [top, top + 24 * cellH]);
    svg.append('g').selectAll('rect').data(rows).join('rect')
      .attr('x', (r) => x(r.week)).attr('y', (r) => yb(r.hour)).attr('width', Math.max(1, x.bandwidth() - 0.5)).attr('height', yb.bandwidth() - 0.5)
      .style('fill', ACCENT).style('fill-opacity', (r) => op(r.rho1_1s));
    svg.append('g').attr('transform', `translate(${left},0)`).call(d3.axisLeft(yb).tickValues(HOUR_TICKS).tickFormat((h) => `${h}:00`).tickSize(0)).call(clean).call((s) => s.select('.domain').remove());
    const xt = d3.scaleUtc([parseDay(weeks[0]), parseDay(weeks[weeks.length - 1])], [left + x.bandwidth() / 2, W - right - x.bandwidth() / 2]);
    const months = d3.utcMonth.count(...xt.domain()), fit = Math.max(2, Math.floor((W - left - right) / 80));
    const every = [1, 2, 3, 6, 12].find((m) => months / m <= fit) || 12;
    svg.append('g').attr('transform', `translate(0,${top + 24 * cellH + 2})`).call(d3.axisBottom(xt).ticks(d3.utcMonth.every(every)).tickFormat(d3.utcFormat('%b %Y')).tickSizeOuter(0)).call(clean).call((s) => s.select('.domain').remove());
    for (const [side, color] of [['a', MUTED], ['b', ACCENT]]) {
      const a = Math.max(+parseDay(cmp[side][1]), +xt.domain()[0]), b = Math.min(+parseDay(cmp[side][2]), +xt.domain()[1]);
      if (b > a) svg.append('rect').attr('x', xt(a)).attr('width', xt(b) - xt(a)).attr('y', H - 10).attr('height', 4).attr('rx', 2).style('fill', color);
    }
    const byKey = new Map(rows.map((r) => [`${r.week}|${r.hour}`, r])), tip = tipOf(el);
    const mark = svg.append('rect').style('fill', 'none').style('stroke', 'var(--fg)').attr('visibility', 'hidden');
    hoverRect(svg, left, top, W - left - right, 24 * cellH)
      .on('pointermove', (ev) => {
        const [mx, my] = d3.pointer(ev);
        const week = weeks[Math.max(0, Math.min(weeks.length - 1, Math.floor((mx - left) / x.step())))];
        const hour = Math.max(0, Math.min(23, Math.floor((my - top) / cellH)));
        const r = byKey.get(`${week}|${hour}`);
        if (!r) { tip.hidden = true; mark.attr('visibility', 'hidden'); return; }
        mark.attr('x', x(week)).attr('y', yb(hour)).attr('width', x.bandwidth()).attr('height', yb.bandwidth()).attr('visibility', 'visible');
        tipLines(tip, `Week of ${d3.utcFormat('%b %-d, %Y')(r.d)} · ${hour}:00 JST · ${r.hours} h`, [[null, `ρ next second: ${d3.format('.3f')(r.rho1_1s)}`]]);
        placeTip(el, tip, mx, my);
      })
      .on('pointerleave', () => { tip.hidden = true; mark.attr('visibility', 'hidden'); });
  }

  const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  // Sections 9–11: one weekday × hour grid per metric, all following one weeks switch.
  const DOW_CHARTS = [
    { id: 'dow', f: 'rho1_1s', color: ACCENT, label: () => 'ρ next second', fmt: d3.format('.3f'), cell: (v) => d3.format('.2f')(v).replace(/^0/, '') },
    { id: 'dowvol', f: 'vol', color: 'var(--series-3)', label: () => `${exch(meta.lead)} vol, bps/hour`, fmt: d3.format(',.1f'), cell: d3.format('.0f') },
    { id: 'dowspr', f: 'lag_spread_bps', color: 'var(--series-2)', label: () => `${exch(meta.lag)} spread, bps`, fmt: d3.format('.2f'), cell: d3.format('.2f') },
  ];
  const dowCache = new Map();
  function dowCells(range) {
    const key = `${range}|${cmp.id}`;
    if (!dowCache.has(key)) {
      const rows = range === 'all' ? hourly : hourly.filter((r) => inPeriod(r.date, range));
      const weekday = (d) => (parseDay(d).getUTCDay() + 6) % 7;
      dowCache.set(key, Array.from(d3.rollup(rows, (rs) => Object.fromEntries([['hours', rs.length], ...DOW_CHARTS.map((c) => [c.f, median(rs.map((r) => r[c.f]))])]),
        (r) => weekday(r.date), (r) => r.hour), ([day, byHour]) => Array.from(byHour, ([hour, v]) => ({ day, hour, ...v }))).flat());
    }
    return dowCache.get(key);
  }

  function dowToggles() {
    for (const el of document.querySelectorAll('.dow-range')) {
      el.replaceChildren();
      for (const [id, label] of [['all', 'All weeks'], ['a', cmp.a[0]], ['b', cmp.b[0]]]) {
        const btn = document.createElement('button'); btn.type = 'button';
        btn.textContent = label; btn.dataset.range = id; btn.setAttribute('aria-pressed', String(id === dowRange));
        btn.addEventListener('click', () => { dowRange = id; dowAll(); });
        el.appendChild(btn);
      }
    }
  }
  function dowAll() { dowToggles(); for (const c of DOW_CHARTS) dowChart(c); }

  function dowChart(c) {
    const el = document.getElementById(c.id); el.replaceChildren();
    const rows = dowCells(dowRange).filter((r) => !isNaN(r[c.f]));
    if (!rows.length) return;
    // one scale for all three views, so a shade means the same value when switching periods
    const [lo, hiV] = d3.extent(['all', 'a', 'b'].flatMap((k) => dowCells(k)), (r) => r[c.f]);
    const op = d3.scaleLinear([lo, hiV], [0.06, 1]);
    const lg = document.getElementById(`${c.id}-legend`); lg.replaceChildren();
    const ramp = document.createElement('span'); ramp.className = 'ramp'; ramp.style.setProperty('--ramp', c.color);
    ramp.append(c.fmt(lo), document.createElement('i'), c.fmt(hiV));
    lg.appendChild(ramp);
    const W = el.clientWidth, top = 4, bottom = 24;
    const svg = svgIn(el, W, 100, `${c.label()} by JST weekday and hour`);
    const left = 8 + measure(svg, WEEKDAYS), right = 4;
    const cellW = (W - left - right) / 24, cellH = Math.max(18, Math.min(30, cellW * 0.8)), H = top + 7 * cellH + bottom;
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const x = d3.scaleBand(d3.range(24), [left, W - right]), y = d3.scaleBand(d3.range(7), [top, top + 7 * cellH]);
    const g = svg.append('g').selectAll('g').data(rows).join('g');
    g.append('rect').attr('x', (r) => x(r.hour)).attr('y', (r) => y(r.day)).attr('width', x.bandwidth() - 1).attr('height', y.bandwidth() - 1)
      .style('fill', c.color).style('fill-opacity', (r) => op(r[c.f]));
    if (cellW >= 34) {
      // values in the cells when they fit; dark cells get light text
      g.append('text').attr('x', (r) => x(r.hour) + x.bandwidth() / 2).attr('y', (r) => y(r.day) + y.bandwidth() / 2).attr('dy', '0.35em').attr('text-anchor', 'middle')
        .text((r) => c.cell(r[c.f])).style('font-size', '10px').style('font-variant-numeric', 'tabular-nums')
        .style('fill', (r) => (op(r[c.f]) > 0.55 ? 'var(--panel)' : 'var(--fg)')).style('pointer-events', 'none');
    }
    svg.append('g').attr('transform', `translate(${left},0)`).call(d3.axisLeft(y).tickFormat((d) => WEEKDAYS[d]).tickSize(0)).call(clean).call((s) => s.select('.domain').remove());
    svg.append('g').attr('transform', `translate(0,${top + 7 * cellH})`).call(d3.axisBottom(x).tickValues(cellW >= 34 ? d3.range(24) : HOUR_TICKS).tickFormat((h) => (cellW >= 34 ? h : `${h}:00`)).tickSize(0)).call(clean).call((s) => s.select('.domain').remove());
    const byKey = new Map(rows.map((r) => [`${r.day}|${r.hour}`, r])), tip = tipOf(el);
    const mark = svg.append('rect').style('fill', 'none').style('stroke', 'var(--fg)').attr('visibility', 'hidden');
    hoverRect(svg, left, top, W - left - right, 7 * cellH)
      .on('pointermove', (ev) => {
        const [mx, my] = d3.pointer(ev);
        const hour = Math.max(0, Math.min(23, Math.floor((mx - left) / x.step()))), day = Math.max(0, Math.min(6, Math.floor((my - top) / cellH)));
        const r = byKey.get(`${day}|${hour}`);
        if (!r) { tip.hidden = true; mark.attr('visibility', 'hidden'); return; }
        mark.attr('x', x(hour)).attr('y', y(day)).attr('width', x.bandwidth() - 1).attr('height', y.bandwidth() - 1).attr('visibility', 'visible');
        tipLines(tip, `${WEEKDAYS[day]} ${hour}:00–${hour + 1}:00 JST · ${d3.format(',')(r.hours)} hours`,
          DOW_CHARTS.map((o) => [o.id === c.id ? c.color : null, `${o.label()}: ${o.fmt(r[o.f])}`]));
        placeTip(el, tip, mx, my);
      })
      .on('pointerleave', () => { tip.hidden = true; mark.attr('visibility', 'hidden'); });
  }

  function method() {
    const lead = exch(meta.lead), lag = exch(meta.lag);
    const el = document.getElementById('method-text'); el.replaceChildren();
    const paras = [
      ['Data. ', `Silver MarketPrice mid at execution size ${meta.execution_size}, one file per JST day, for ${meta.lead} and ${meta.lag}, ${meta.first_date} to ${meta.last_date} (${meta.days} days). Days missing on either side are left out, including the 2026-04-02 to 04-09 collector outage.`],
      ['100ms grid. ', `Last quote per 100ms bucket, carried forward; log returns, de-meaned per day. ρ(k) is the correlation of ${lag}'s return at t with ${lead}'s return at t−k, for k from −1s to +3s. Peak lag is the k with the highest ρ.`],
      ['1s grid. ', `The same on 1-second buckets, which is much less affected by ${lead}'s tick size. "ρ next second" is the correlation at a one-bucket lag: the part of ${lag}'s move that arrives after ${lead}'s second is over. β 0–3s is how much of a ${lead} move ${lag} has reflected after three seconds (1.0 = all of it).`],
      ['Source. ', `MarketPrice is derived from bronze by Hase. It leaves out snapshots the book cannot fill at the execution size, and snapshots whose walked spread is over 100 bps — collector artefacts, a quote far from the market. "Snapshots dropped" in section 5 counts both per month, both markets together.`],
      ['Context. ', `Volatility is the daily sum of 100ms squared returns, square-rooted, in bps. Tick size is ${lead}'s smallest bid step over the day's median price. Updates count mid-price changes per day.`],
      ['Hourly. ', `Sections 6–11, and sections 1–4 when "Per hour" is selected, run the same 100ms and 1s calculations inside each JST hour, with returns de-meaned per hour, so an hour's ρ compares with the daily one but is noisier. Hourly volatility is that hour's own sum of 100ms squared returns, square-rooted. An hour is left out when either market has quotes in fewer than 54 of its 60 minutes, or when ${lag}'s price changes in fewer than 55 of them (its weekly maintenance, Saturday about 9:00–11:00 JST, when quotes keep arriving but are frozen)${hourly ? ` (${d3.format(',')(hourly.length)} hours kept)` : ''}.`],
    ];
    for (const [h, t] of paras) { const p = document.createElement('p'); const b = document.createElement('b'); b.textContent = h; p.append(b, t); el.appendChild(p); }
    document.getElementById('foot').textContent = `Generated ${meta.generated_at.replace('T', ' ').slice(0, 16)} UTC by python src/cli.py report leadlag. Data: daily.csv and hourly.csv next to this page.`;
  }

  function draw() {
    if (!daily) return;
    const c = compare();
    header(c); tiles(c); multiples(); lagCurve(); scatter(); table();
    if (hourly) { hours(); volBinChart(); heat(); dowAll(); }
  }

  function basisToggle() {
    const el = document.getElementById('basis'); el.replaceChildren();
    for (const [id, b] of Object.entries(BASES)) {
      const btn = document.createElement('button'); btn.type = 'button'; btn.id = `basis-${id}`;
      btn.textContent = b.label; btn.setAttribute('aria-pressed', String(id === basis));
      btn.addEventListener('click', () => { basis = id; el.querySelectorAll('button').forEach((o) => o.setAttribute('aria-pressed', String(o === btn))); draw(); });
      el.appendChild(btn);
    }
  }

  function toggle() {
    const el = document.getElementById('toggle'); el.replaceChildren();
    for (const comparison of meta.comparisons) {
      const b = document.createElement('button'); b.type = 'button'; b.id = `cmp-${comparison.id}`;
      b.textContent = `${comparison.a[0]} vs ${comparison.b[0]}`; b.setAttribute('aria-pressed', String(comparison === cmp));
      b.addEventListener('click', () => { cmp = comparison; el.querySelectorAll('button').forEach((o) => o.setAttribute('aria-pressed', String(o === b))); draw(); });
      el.appendChild(b);
    }
  }

  function fail(msg) {
    document.getElementById('takeaway').textContent = msg;
    for (const id of ['multi', 'curve', 'scatter']) { const e = document.getElementById(id); e.replaceChildren(); const d = document.createElement('div'); d.className = 'fail'; d.textContent = 'Failed to load'; e.appendChild(d); }
  }

  const parseRows = (rows) => rows.map((r) => { const o = { date: r.date }; for (const [k, v] of Object.entries(r)) if (k !== 'date') o[k] = num(v); return o; });
  // A report built before hourly.csv existed still draws its daily sections.
  Promise.all([d3.json('report.json'), d3.csv('daily.csv'), d3.csv('hourly.csv').catch(() => null)]).then(([m, rows, hrows]) => {
    meta = m;
    daily = parseRows(rows);
    hourly = hrows && hrows.length ? parseRows(hrows) : null;
    for (const r of daily) r.vol = r.vol_lead_bps_d;
    if (hourly) for (const r of hourly) r.vol = r.vol_lead_bps_h;
    basis = hourly ? 'h' : 'd';
    if (hourly) weekHourRows = weekHour();
    else document.querySelectorAll('.hourly').forEach((e) => { e.hidden = true; });
    cmp = meta.comparisons[0];
    document.title = meta.title; document.getElementById('title').textContent = meta.title;
    document.querySelectorAll('.lead-name').forEach((e) => { e.textContent = exch(meta.lead); });
    document.querySelectorAll('.lag-name').forEach((e) => { e.textContent = exch(meta.lag); });
    toggle(); if (hourly) basisToggle(); method(); draw();
    let w = window.innerWidth;
    window.addEventListener('resize', () => { if (window.innerWidth !== w) { w = window.innerWidth; draw(); } });
  }).catch((e) => fail(`Could not load this report's data (${e.message}). Rebuild it with python src/cli.py report leadlag.`));
})();
