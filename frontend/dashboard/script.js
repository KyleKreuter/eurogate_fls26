let predictionChartInstance = null;
let historyChartInstance = null;
let globalData = [];

document.addEventListener('DOMContentLoaded', () => {
    // 1. Setup Tab interactions — lazy-load Historical tab on first click
    const tabBtns = document.querySelectorAll('.tab-nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    let historyLoaded = false;

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');

            // Lazy-load historical tab on first open
            if (targetId === 'tab-history' && !historyLoaded) {
                historyLoaded = true;
                fetchOverviewAnalytics();
                loadContainerPage();
            }
        });
    });

    // 2. Fetch Dashboard Payload and map charts
    Papa.parse('dashboard_data.csv', {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: function(results) {
            globalData = results.data.filter(row => row.timestamp_utc != null);
            
            // Set bindings for timeline toggles
            document.getElementById('btn-24h').addEventListener('click', () => {
                document.getElementById('btn-24h').classList.add('active');
                document.getElementById('btn-14d').classList.remove('active');
                renderDashboard(24);
            });

            document.getElementById('btn-14d').addEventListener('click', () => {
                document.getElementById('btn-14d').classList.add('active');
                document.getElementById('btn-24h').classList.remove('active');
                renderDashboard(336);
            });

            renderDashboard(336); // Trigger immediate build
            // Historical tab loads lazily on first click — not here
        }
    });
});

function renderDashboard(hours) {
    const targetData = globalData.slice(0, hours);
    
    const labels = targetData.map(row => {
        const date = new Date(row.timestamp_utc);
        return date.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit' });
    });
    
    const powerKw = targetData.map(row => row.pred_power_kw);
    const p90Kw = targetData.map(row => row.pred_p90_kw);
    const historyLastYear = targetData.map(row => row.history_lastyear_kw);

    // ==========================================
    // CHART 1: Prediction Graph Allocation
    // ==========================================
    const ctxPred = document.getElementById('predictionChart').getContext('2d');
    if (predictionChartInstance) predictionChartInstance.destroy();
    
    Chart.defaults.font.family = "'Outfit', -apple-system, sans-serif";
    Chart.defaults.color = '#7F8C8D'; 

    const baseOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { position: 'top', labels: { useBorderRadius: true, borderRadius: 0, boxWidth: 20, boxHeight: 2, padding: 25, font: { size: 13, weight: 500 }, color: '#2C3E50' } },
            tooltip: { backgroundColor: '#FFFFFF', titleColor: '#2C3E50', bodyColor: '#7F8C8D', borderColor: '#E2E8F0', borderWidth: 1, padding: 12, cornerRadius: 4, titleFont: { size: 13, weight: 600, family: 'Outfit' }, bodyFont: { size: 12, family: 'Outfit' }, displayColors: true, boxPadding: 4, intersect: false }
        },
        scales: {
            x: { grid: { color: '#E2E8F0', drawBorder: false }, ticks: { color: '#7F8C8D', font: { size: 11 }, maxRotation: 45, minRotation: 45, maxTicksLimit: hours === 24 ? 24 : 14 } },
            y: { grid: { color: '#E2E8F0', drawBorder: false }, ticks: { color: '#7F8C8D', font: { size: 11 }, padding: 8 }, title: { display: true, text: 'Consumption (kW)', font: { size: 12, weight: 600 }, color: '#2C3E50', padding: { bottom: 10 } } }
        }
    };

    predictionChartInstance = new Chart(ctxPred, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Predicted Power (kW)', data: powerKw, borderColor: '#2980B9', borderWidth: 2, tension: 0.1, pointRadius: 0, pointHoverRadius: 4, fill: false },
                { label: 'Risk Boundary P90 (kW)', data: p90Kw, borderColor: '#C0392B', borderWidth: 1.5, tension: 0.1, pointRadius: 0, pointHoverRadius: 4, fill: false },
                { label: 'History 1-Year (kW)', data: historyLastYear, borderColor: '#95A5A6', borderWidth: 1.5, borderDash: [4, 4], tension: 0.1, pointRadius: 0, pointHoverRadius: 4, fill: false }
            ]
        },
        options: baseOptions
    });


}

// ================================================
// FLEET ANALYTICS — 6 charts from /api/overview-analytics
// ================================================
let analyticsCharts = {};

function fetchOverviewAnalytics() {
    fetch('/api/overview-analytics')
        .then(r => r.json())
        .then(data => {
            document.getElementById('analytics-loading').style.display = 'none';
            document.getElementById('analytics-grid').style.display = 'block';
            renderAnalyticsCharts(data);
        })
        .catch(err => {
            document.getElementById('analytics-loading').innerHTML =
                '<p style="color:#C0392B;">Could not load analytics. Is the server running?</p>';
        });
}

function mkChart(id, config) {
    if (analyticsCharts[id]) analyticsCharts[id].destroy();
    analyticsCharts[id] = new Chart(document.getElementById(id).getContext('2d'), config);
}

const PALETTE = [
    '#2980B9','#27AE60','#8E44AD','#E67E22','#E74C3C',
    '#1ABC9C','#F39C12','#D35400','#16A085','#2C3E50',
    '#3498DB','#2ECC71','#9B59B6','#E67E22','#C0392B',
    '#1ABC9C','#F1C40F','#95A5A6','#7F8C8D'
];
const commonTooltip = { backgroundColor:'#fff', titleColor:'#2C3E50', bodyColor:'#7F8C8D', borderColor:'#E2E8F0', borderWidth:1 };
const commonScales  = (yLabel='') => ({
    x: { grid:{ display:false }, ticks:{ color:'#7F8C8D', font:{size:10} } },
    y: { grid:{ color:'#E2E8F0' }, ticks:{ color:'#7F8C8D', font:{size:10} },
         title: yLabel ? { display:true, text:yLabel, color:'#7F8C8D', font:{size:10} } : {} }
});

function renderAnalyticsCharts(d) {
    // 1. Active containers per day
    mkChart('chartActivePerDay', {
        type: 'line',
        data: {
            labels: d.active_per_day.dates.map(dt => {
                const dd = new Date(dt);
                return dd.toLocaleDateString([], { month: 'short', day: 'numeric' });
            }),
            datasets: [{
                label: 'Active Containers',
                data: d.active_per_day.counts,
                borderColor: '#2980B9',
                backgroundColor: 'rgba(41,128,185,0.08)',
                fill: true, tension: 0.3, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode:'index', intersect:false },
            plugins: { legend:{ display:false }, tooltip: commonTooltip },
            scales: { ...commonScales('Containers'), x: { grid:{display:false}, ticks:{ color:'#7F8C8D', font:{size:10}, maxTicksLimit:14, maxRotation:45 } } }
        }
    });

    // 2. Hardware distribution — group tail into "Other" to keep legend readable
    const TOP_N = 7;
    const sortedHW = [...d.hardware_types].sort((a, b) => b.cnt - a.cnt);
    const topHW = sortedHW.slice(0, TOP_N);
    const otherCount = sortedHW.slice(TOP_N).reduce((s, h) => s + h.cnt, 0);
    const hwLabels = [...topHW.map(h => h.hardware_type), ...(otherCount > 0 ? ['Other'] : [])];
    const hwCounts = [...topHW.map(h => h.cnt),          ...(otherCount > 0 ? [otherCount] : [])];
    const hwColors = PALETTE.slice(0, hwLabels.length);

    mkChart('chartHardware', {
        type: 'doughnut',
        data: {
            labels: hwLabels,
            datasets: [{ data: hwCounts, backgroundColor: hwColors, borderWidth: 2, borderColor:'#fff' }]
        },
        options: {
            responsive:true, maintainAspectRatio:false, cutout:'62%',
            plugins: {
                legend:{
                    position:'right',
                    labels:{ color:'#2C3E50', font:{size:11}, padding:10, boxWidth:14, boxHeight:14 }
                },
                tooltip: commonTooltip
            }
        }
    });

    // 3. Visit duration histogram
    mkChart('chartDuration', {
        type: 'bar',
        data: {
            labels: d.duration_hist.labels,
            datasets: [{ label: 'Visits', data: d.duration_hist.counts,
                backgroundColor: PALETTE.slice(0, d.duration_hist.labels.length), borderRadius: 4 }]
        },
        options: {
            responsive:true, maintainAspectRatio:false,
            plugins:{ legend:{display:false}, tooltip: commonTooltip },
            scales: commonScales('# Visits')
        }
    });

    // 4. Container size vs avg power (horizontal bar)
    mkChart('chartSizes', {
        type: 'bar',
        data: {
            labels: d.container_sizes.map(s => `${s.size}ft`),
            datasets: [{
                label: 'Avg Power (kW)', data: d.container_sizes.map(s => s.avg_power_kw),
                backgroundColor: ['#2980B9','#27AE60','#8E44AD'], borderRadius: 4
            }]
        },
        options: {
            responsive:true, maintainAspectRatio:false, indexAxis:'y',
            plugins:{ legend:{display:false}, tooltip: commonTooltip },
            scales: { x:{ grid:{color:'#E2E8F0'}, ticks:{color:'#7F8C8D',font:{size:10}}, title:{display:true,text:'kW',color:'#7F8C8D',font:{size:10}} },
                      y:{ grid:{display:false}, ticks:{color:'#7F8C8D',font:{size:10}} } }
        }
    });

    // 5. Monthly energy (bar)
    mkChart('chartMonthly', {
        type: 'bar',
        data: {
            labels: d.monthly_energy.map(m => m.month),
            datasets: [{ label: 'MWh', data: d.monthly_energy.map(m => m.total_mwh),
                backgroundColor: '#2980B9', borderRadius: 4 }]
        },
        options: {
            responsive:true, maintainAspectRatio:false,
            plugins:{ legend:{display:false}, tooltip: commonTooltip },
            scales: commonScales('MWh')
        }
    });

    // 6. SetPoint distribution (bar)
    mkChart('chartSetpoint', {
        type: 'bar',
        data: {
            labels: d.setpoint_dist.map(s => `${s.sp_bin}°C`),
            datasets: [{ label: 'Readings', data: d.setpoint_dist.map(s => s.cnt),
                backgroundColor: d.setpoint_dist.map(s => s.sp_bin < 0 ? '#2980B9' : '#E67E22'), borderRadius: 3 }]
        },
        options: {
            responsive:true, maintainAspectRatio:false,
            plugins:{ legend:{display:false}, tooltip: commonTooltip },
            scales: commonScales('# Readings')
        }
    });

    // 7. Hourly x DOW Matrix
    const hours = Array.from({length:24}, (_,i) => i.toString());
    const dows = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    const maxHourCnt = Math.max(1, ...d.hourly_heatmap.map(r=>r.count));
    
    mkChart('chartHourDow', {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Simultaneous Visits',
                data: d.hourly_heatmap.map(r => ({ x: r.hour.toString(), y: dows[r.dow], v: r.count })),
                backgroundColor: ctx => {
                    const v = ctx.dataset.data[ctx.dataIndex]?.v || 0;
                    return `rgba(41, 128, 185, ${Math.max(0.1, v/maxHourCnt)})`;
                },
                width: (ctx) => { const a = ctx.chart.chartArea; return a ? (a.right - a.left) / 24 - 1 : 0; },
                height: (ctx) => { const a = ctx.chart.chartArea; return a ? (a.bottom - a.top) / 7 - 1 : 0; },
                borderWidth: 1, borderColor: '#fff'
            }]
        },
        options: {
            responsive:true, maintainAspectRatio:false,
            plugins: {
                legend:{display:false},
                tooltip: { callbacks: { title: () => '', label: ctx => `${ctx.raw.v} active visits` }, ...commonTooltip }
            },
            scales: {
                x: { type: 'category', labels: hours, grid: {display:false}, ticks: {color:'#7F8C8D', font:{size:10}, callback: function(val, index) { return hours[index] + ':00'; }} },
                y: { type: 'category', labels: dows, grid: {display:false}, ticks: {color:'#7F8C8D', font:{size:10}} }
            }
        }
    });

    // 8. Annual Calendar Matrix
    const calMatrix = [];
    let maxCalCnt = 1;
    const startDate = new Date(d.active_per_day.dates[0]);
    const startDow = startDate.getDay();
    d.active_per_day.dates.forEach((dt, i) => {
        const v = d.active_per_day.counts[i];
        if(v > maxCalCnt) maxCalCnt = v;
        const dObj = new Date(dt);
        const dayOffset = Math.floor((dObj.getTime() - startDate.getTime())/86400000);
        const week = Math.floor((dayOffset + startDow) / 7);
        calMatrix.push({ x: week.toString(), y: dows[dObj.getDay()], v: v, date: dt });
    });
    
    const maxWeek = Math.max(...calMatrix.map(c => parseInt(c.x))) || 52;
    const weeks = Array.from({length: maxWeek+1}, (_,i) => i.toString());

    mkChart('chartCalendar', {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Active Containers',
                data: calMatrix,
                backgroundColor: ctx => {
                    const v = ctx.dataset.data[ctx.dataIndex]?.v || 0;
                    return `rgba(39, 174, 96, ${Math.max(0.1, v/maxCalCnt)})`; // Green pattern
                },
                width: (ctx) => { const a = ctx.chart.chartArea; return a ? (a.right - a.left) / (maxWeek + 1) - 1 : 0; },
                height: (ctx) => { const a = ctx.chart.chartArea; return a ? (a.bottom - a.top) / 7 - 1 : 0; },
                borderWidth: 1, borderColor: '#fff'
            }]
        },
        options: {
            responsive:true, maintainAspectRatio:false,
            plugins: {
                legend:{display:false},
                tooltip: { callbacks: { title: () => '', label: ctx => `${ctx.raw.date}: ${ctx.raw.v} active` }, ...commonTooltip }
            },
            scales: {
                x: { type: 'category', labels: weeks, grid: {display:false}, ticks: {display:false} },
                y: { type: 'category', labels: dows, grid: {display:false}, ticks: {color:'#7F8C8D', font:{size:10}} }
            }
        }
    });
}

// Kept for backwards compat if called directly:
function fetchAdvancedHistoricalData() { /* no-op: replaced by fetchOverviewAnalytics */ }
function renderAdvancedAnalytics() {}


// ===============================================
// PAGINATED CONTAINER LIST
// ===============================================

let allContainers    = [];  // current page data
let containerTotal   = 0;
let containerOffset  = 0;
const containerLimit = 50;
let sortCol = 'total_connected_hours';
let sortDir = -1; // -1 = desc, 1 = asc
let specificContainerInstance = null;

function loadContainerPage(query) {
    const q = query || document.getElementById('container-search').value || '';
    const url = `/api/containers?limit=${containerLimit}&offset=${containerOffset}&sort=${sortCol}&dir=${sortDir === 1 ? 'ASC' : 'DESC'}${ q ? '&q=' + encodeURIComponent(q) : ''}`;
    fetch(url)
        .then(r => r.json())
        .then(data => {
            allContainers  = data.containers;
            containerTotal = data.total;
            renderContainerTable(allContainers);
            updatePaginationBar();
        })
        .catch(() => {
            document.getElementById('container-list-body').innerHTML =
                '<tr><td colspan="5" style="text-align:center;color:#C0392B;">Could not reach backend server.</td></tr>';
        });
}

function updatePaginationBar() {
    const page = Math.floor(containerOffset / containerLimit) + 1;
    const pages = Math.ceil(containerTotal / containerLimit);
    document.getElementById('page-info').textContent = `Page ${page} of ${pages} (${containerTotal.toLocaleString()} containers)`;
    document.getElementById('btn-prev-page').disabled = containerOffset === 0;
    document.getElementById('btn-next-page').disabled = containerOffset + containerLimit >= containerTotal;
}

function changePage(dir) {
    containerOffset = Math.max(0, containerOffset + dir * containerLimit);
    loadContainerPage();
}

function onContainerSearch(query) {
    containerOffset = 0;  // reset to page 1 on new search
    loadContainerPage(query);
}

// Legacy alias kept for any inline HTML that still uses it
function filterContainerTable(query) { onContainerSearch(query); }
function populateInteractiveContainerList() { /* no-op: loadContainerPage called lazily */ }


function renderContainerTable(containers) {
    const tbody = document.getElementById('container-list-body');
    if (!containers.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-secondary);">No results.</td></tr>';
        return;
    }

    // Sort
    const sorted = [...containers].sort((a, b) => {
        const av = a[sortCol] ?? '';
        const bv = b[sortCol] ?? '';
        if (typeof av === 'number') return (av - bv) * sortDir;
        return String(av).localeCompare(String(bv)) * sortDir;
    });

    const arrow = col => col === sortCol ? (sortDir === 1 ? ' ↑' : ' ↓') : ' ⇅';
    const thStyle = 'cursor:pointer; user-select:none; white-space:nowrap;';

    // Rebuild thead with clickable headers
    document.querySelector('#container-list-table thead tr').innerHTML = `
        <th style="${thStyle}" onclick="sortTable('uuid')">Container UUID${arrow('uuid')}</th>
        <th style="${thStyle} text-align:center;" onclick="sortTable('num_visits')">Visits${arrow('num_visits')}</th>
        <th style="${thStyle}" onclick="sortTable('total_connected_hours')">Total Connected${arrow('total_connected_hours')}</th>
        <th style="${thStyle}" onclick="sortTable('avg_visit_hours')">Avg Visit Duration${arrow('avg_visit_hours')}</th>
        <th></th>
    `;

    let html = '';
    sorted.forEach(c => {
        const totalDays = (c.total_connected_hours / 24).toFixed(1);
        const avgDays  = (c.avg_visit_hours / 24).toFixed(1);
        html += `<tr>
            <td style="font-family:monospace; font-size:0.8rem;">${c.uuid}</td>
            <td style="color:var(--text-secondary); text-align:center;">${c.num_visits}</td>
            <td class="col-power">${c.total_connected_hours}h <span style="color:var(--text-secondary);font-size:0.8rem;">(${totalDays}d)</span></td>
            <td style="color:var(--text-secondary);">${c.avg_visit_hours}h <span style="font-size:0.78rem;">(${avgDays}d avg)</span></td>
            <td><button onclick="selectContainer('${c.uuid}')" class="toggle-btn active" style="padding:0.3rem 0.8rem; font-size:0.8rem; background:var(--eurogate-blue); color:#fff;">View</button></td>
        </tr>`;
    });
    tbody.innerHTML = html;
}

function sortTable(col) {
    if (sortCol === col) {
        sortDir *= -1;
    } else {
        sortCol = col;
        sortDir = col === 'uuid' ? 1 : -1;
    }
    containerOffset = 0;  // back to page 1
    loadContainerPage();
}

function filterContainerTable(query) {
    const lower = query.toLowerCase();
    const filtered = allContainers.filter(c => c.uuid.toLowerCase().includes(lower));
    renderContainerTable(filtered);
}

function selectContainer(uuid) {
    document.getElementById('container-search').value = uuid;
    fetchSpecificContainer(uuid);
    // Scroll to the chart
    document.getElementById('container-stats-banner').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function fetchSpecificContainer(uuid) {
    document.getElementById('container-empty-state').style.display = 'none';
    document.getElementById('container-graph-target').style.display = 'block';
    document.getElementById('container-stats-banner').style.display = 'none';

    fetch(`/api/data?uuid=${encodeURIComponent(uuid)}`)
        .then(response => response.json())
        .then(data => {
            // Build stats banner
            const banner = document.getElementById('container-stats-banner');
            const totalDays = data.total_connected_hours != null ? (data.total_connected_hours / 24).toFixed(1) : '—';
            const avgDays   = data.avg_visit_hours != null ? (data.avg_visit_hours / 24).toFixed(1) : '—';
            banner.innerHTML = `
                <div class="stat-pill highlight">
                    <span class="stat-label">Harbour Visits</span>
                    <span class="stat-value">${data.num_visits || '—'}</span>
                </div>
                <div class="stat-pill">
                    <span class="stat-label">Total Connected</span>
                    <span class="stat-value">${data.total_connected_hours}h <small style="color:var(--text-secondary)">(${totalDays}d)</small></span>
                </div>
                <div class="stat-pill">
                    <span class="stat-label">Avg. Visit Duration</span>
                    <span class="stat-value">${data.avg_visit_hours}h <small style="color:var(--text-secondary)">(${avgDays}d avg)</small></span>
                </div>
                <div class="stat-pill">
                    <span class="stat-label">Last Visit Start</span>
                    <span class="stat-value" style="font-size:0.9rem;">${data.last_visit_start || '—'}</span>
                </div>
            `;
            banner.style.display = 'flex';

            // Save globally and reset state
            currentContainerData = { timeline: data.timeline, visits: data.visits, uuid };
            activeVisit = 'all';
            activeResolution = 'hourly';

            // Render chart controls and chart
            buildVisitControls(data.visits);
            refreshContainerChart();
            renderVisitTable(data.visits);
        })
        .catch(err => {
            document.getElementById('container-empty-state').style.display = 'block';
            document.getElementById('container-empty-state').innerText = 'Error connecting to backend server.';
            document.getElementById('container-graph-target').style.display = 'none';
        });
}

// ==============================================
// VISIT-AWARE CONTAINER CHART ENGINE
// ==============================================

const VISIT_COLORS = ['#2980B9', '#27AE60', '#8E44AD', '#E67E22', '#E74C3C', '#1ABC9C', '#F39C12', '#D35400'];
let currentContainerData = null;
let activeVisit = 'all';
let activeResolution = 'hourly';

function buildVisitControls(visits) {
    let ctrl = document.getElementById('container-chart-controls');
    if (!ctrl) {
        ctrl = document.createElement('div');
        ctrl.id = 'container-chart-controls';
        document.getElementById('container-graph-target').insertAdjacentElement('beforebegin', ctrl);
    }

    let visitBtns = `<button class="toggle-btn active" id="vbtn-all" onclick="setActiveVisit('all')">All Visits</button>`;
    visits.forEach((v, i) => {
        visitBtns += `<button class="toggle-btn" id="vbtn-${i}" onclick="setActiveVisit('${v.visit_uuid}')">Visit ${i + 1}</button>`;
    });

    ctrl.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; flex-wrap:wrap; gap:0.75rem;">
            <div class="time-toggles">${visitBtns}</div>
            <div class="time-toggles">
                <button class="toggle-btn active" id="rbtn-hourly" onclick="setResolution('hourly')">Hourly</button>
                <button class="toggle-btn"         id="rbtn-daily"  onclick="setResolution('daily')">Daily Avg</button>
            </div>
        </div>`;
}

function setActiveVisit(visitUuid) {
    activeVisit = visitUuid;
    document.querySelectorAll('[id^="vbtn-"]').forEach(b => b.classList.remove('active'));
    if (visitUuid === 'all') {
        document.getElementById('vbtn-all').classList.add('active');
    } else {
        const idx = currentContainerData.visits.findIndex(v => v.visit_uuid === visitUuid);
        const btn = document.getElementById(`vbtn-${idx}`);
        if (btn) btn.classList.add('active');
    }
    refreshContainerChart();
}

function setResolution(res) {
    activeResolution = res;
    document.getElementById('rbtn-hourly').classList.toggle('active', res === 'hourly');
    document.getElementById('rbtn-daily').classList.toggle('active', res === 'daily');
    refreshContainerChart();
}

function aggregateDaily(timeline) {
    const byDay = {};
    timeline.forEach(row => {
        const day = row.time.substring(0, 10);
        if (!byDay[day]) byDay[day] = { power: [], ambient: [], setpoint: [], visit_uuid: row.visit_uuid };
        byDay[day].power.push(row.power_kw);
        if (row.ambient  != null) byDay[day].ambient.push(row.ambient);
        if (row.setpoint != null) byDay[day].setpoint.push(row.setpoint);
    });
    return Object.entries(byDay).sort().map(([day, d]) => ({
        time:       day,
        visit_uuid: d.visit_uuid,
        power_kw:   parseFloat((d.power.reduce((a, b) => a + b, 0) / d.power.length).toFixed(2)),
        ambient:    d.ambient.length  ? parseFloat((d.ambient.reduce((a, b)  => a + b, 0) / d.ambient.length).toFixed(1))  : null,
        setpoint:   d.setpoint.length ? parseFloat((d.setpoint.reduce((a, b) => a + b, 0) / d.setpoint.length).toFixed(1)) : null,
    }));
}

function refreshContainerChart() {
    const { timeline, visits } = currentContainerData;
    const isAll = activeVisit === 'all';

    // ---------- All Visits: scaffold full date range with gaps ----------
    if (isAll) {
        // Always use daily granularity for the "All" view — hourly across a full year is unreadable
        const dayMap = {};  // YYYY-MM-DD -> { power: [], ambient: [], setpoint: [], visit_uuid }
        timeline.forEach(row => {
            const day = row.time.substring(0, 10);
            if (!dayMap[day]) dayMap[day] = { power: [], ambient: [], setpoint: [], visit_uuid: row.visit_uuid };
            dayMap[day].power.push(row.power_kw);
            if (row.ambient  != null) dayMap[day].ambient.push(row.ambient);
            if (row.setpoint != null) dayMap[day].setpoint.push(row.setpoint);
        });

        // Build continuous date axis from very first to very last event
        const allDays = Object.keys(dayMap).sort();
        const startDate = new Date(allDays[0]);
        const endDate   = new Date(allDays[allDays.length - 1]);
        const dateAxis  = [];
        for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
            dateAxis.push(d.toISOString().substring(0, 10));
        }

        const labels = dateAxis.map(d => {
            const dt = new Date(d);
            return dt.toLocaleDateString([], { month: 'short', day: 'numeric' });
        });

        // One dataset per visit, each with null outside its own dates
        const visitUuidSet = visits.reduce((acc, v, i) => { acc[v.visit_uuid] = i; return acc; }, {});
        const powerDatasets = visits.map((v, i) => ({
            label: `Visit ${i + 1} Power (kW)`,
            data: dateAxis.map(day => {
                const d = dayMap[day];
                if (!d || d.visit_uuid !== v.visit_uuid) return null;
                return parseFloat((d.power.reduce((a, b) => a + b, 0) / d.power.length).toFixed(2));
            }),
            borderColor: VISIT_COLORS[i % VISIT_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.15,
            pointRadius: 0,
            pointHoverRadius: 4,
            spanGaps: false,   // null = visible gap in chart
            yAxisID: 'y',
        }));

        const ambientData = dateAxis.map(day => {
            const d = dayMap[day];
            if (!d || !d.ambient.length) return null;
            return parseFloat((d.ambient.reduce((a, b) => a + b, 0) / d.ambient.length).toFixed(1));
        });
        const setpointData = dateAxis.map(day => {
            const d = dayMap[day];
            if (!d || !d.setpoint.length) return null;
            return parseFloat((d.setpoint.reduce((a, b) => a + b, 0) / d.setpoint.length).toFixed(1));
        });

        const allPowers = dateAxis.flatMap(day => dayMap[day] ? [dayMap[day].power.reduce((a,b)=>a+b,0)/dayMap[day].power.length] : []);
        const minP = allPowers.length ? Math.min(...allPowers) : 0;
        const maxP = allPowers.length ? Math.max(...allPowers) : 10;
        const padP = Math.max((maxP - minP) * 0.1, 0.5);

        renderChart(labels, powerDatasets, ambientData, setpointData, minP, maxP, padP, true);
        return;
    }

    // ---------- Single Visit: filter + optional hourly/daily ----------
    let data = timeline.filter(r => r.visit_uuid === activeVisit);
    const isDaily = activeResolution === 'daily';
    if (isDaily) data = aggregateDaily(data);

    const labels = data.map(r => {
        const d = new Date(r.time);
        return isDaily
            ? d.toLocaleDateString([], { month: 'short', day: 'numeric' })
            : d.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit' });
    });

    const powerDatasets = [{
        label: 'Power (kW)',
        data: data.map(r => r.power_kw),
        borderColor: VISIT_COLORS[visits.findIndex(v => v.visit_uuid === activeVisit) % VISIT_COLORS.length],
        backgroundColor: 'rgba(41, 128, 185, 0.06)',
        fill: true,
        tension: 0.15,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 5,
        yAxisID: 'y',
    }];

    const ambientData  = data.map(r => r.ambient);
    const setpointData = data.map(r => r.setpoint);

    const powers = data.map(r => r.power_kw).filter(v => v != null);
    const minP = powers.length ? Math.min(...powers) : 0;
    const maxP = powers.length ? Math.max(...powers) : 10;
    const padP = Math.max((maxP - minP) * 0.1, 0.5);

    renderChart(labels, powerDatasets, ambientData, setpointData, minP, maxP, padP, isDaily);
}

function renderChart(labels, powerDatasets, ambientData, setpointData, minP, maxP, padP, isDaily) {
    const ctx = document.getElementById('specificContainerChart').getContext('2d');
    if (specificContainerInstance) specificContainerInstance.destroy();

    specificContainerInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                ...powerDatasets,
                {
                    label: 'SetPoint (°C)',
                    data: setpointData,
                    borderColor: '#C0392B',
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    tension: 0,
                    pointRadius: 0,
                    spanGaps: false,
                    yAxisID: 'y1',
                },
                {
                    label: 'Ambient Temp (°C)',
                    data: ambientData,
                    borderColor: '#F39C12',
                    borderWidth: 1.5,
                    tension: 0.3,
                    pointRadius: 0,
                    spanGaps: false,
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 20, boxHeight: 2, padding: 20, font: { size: 12 }, color: '#2C3E50' },
                },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    titleColor: '#2C3E50',
                    bodyColor: '#7F8C8D',
                    borderColor: '#E2E8F0',
                    borderWidth: 1,
                    padding: 10,
                    filter: item => item.raw != null,  // hide null entries from tooltip
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#7F8C8D', font: { size: 10 }, maxRotation: 45, maxTicksLimit: isDaily ? 24 : 20 },
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    min: Math.max(0, minP - padP),
                    max: maxP + padP,
                    title: { display: true, text: 'Power (kW)', color: '#2980B9', font: { size: 11 } },
                    grid: { color: '#E2E8F0' },
                    ticks: { color: '#7F8C8D', font: { size: 10 } },
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Temperature (°C)', color: '#7F8C8D', font: { size: 11 } },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#7F8C8D', font: { size: 10 } },
                },
            },
        },
    });
}

function renderVisitTable(visits) {
    let existing = document.getElementById('visit-breakdown-table');
    if (!existing) {
        existing = document.createElement('div');
        existing.id = 'visit-breakdown-table';
        document.getElementById('container-graph-target').insertAdjacentElement('afterend', existing);
    }
    if (!visits || !visits.length) { existing.innerHTML = ''; return; }
    const days = v => (v.duration_hours / 24).toFixed(1);
    let html = `<h3 style="margin-top:2rem; margin-bottom:0.75rem; font-size:1.1rem; font-weight:600; color:var(--text-primary);">Visit Breakdown</h3>
    <table class="premium-table">
        <thead><tr>
            <th>#</th>
            <th>Visit UUID</th>
            <th>Visit Start</th>
            <th>Visit End</th>
            <th>Duration</th>
            <th>Readings</th>
            <th>Avg Power</th>
            <th>Hardware</th>
            <th>Size</th>
        </tr></thead><tbody>`;
    visits.forEach((v, i) => {
        const color = VISIT_COLORS[i % VISIT_COLORS.length];
        html += `<tr>
            <td style="color:${color}; font-weight:600;">Visit ${i + 1}</td>
            <td style="font-family:monospace; font-size:0.75rem; color:var(--text-secondary);">${v.visit_uuid}</td>
            <td>${v.visit_start}</td>
            <td>${v.visit_end}</td>
            <td style="color:${color}; font-weight:500;">${v.duration_hours}h <span style="color:var(--text-secondary);font-size:0.8rem;">(${days(v)}d)</span></td>
            <td style="color:var(--text-secondary);">${v.reading_count}</td>
            <td style="color:${color}; font-weight:500;">${v.avg_power_kw} kW</td>
            <td style="color:var(--text-secondary);">${v.hardware_type || '—'}</td>
            <td style="color:var(--text-secondary);">${v.container_size || '—'}</td>
        </tr>`;
    });
}
