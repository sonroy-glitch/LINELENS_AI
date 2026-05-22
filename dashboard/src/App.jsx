import React, { useEffect, useState } from 'react';
import { supabase } from './supabaseClient';
import { Loader2 } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
} from 'chart.js';
import { Bar, Doughnut, Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
);

function App() {
  const [data, setData] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchData = async () => {
    try {
      const { data: records, error } = await supabase
        .from('linelens_reports')
        .select('report_data, created_at')
        .order('created_at', { ascending: false })
        .limit(1);

      if (error) throw error;
      
      if (records && records.length > 0) {
        setData(records[0].report_data);
        setLastUpdated(new Date(records[0].created_at).toLocaleTimeString());
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (!data) {
    return (
      <div className="dashboard-container">
        <div className="empty-state">
          <Loader2 size={48} color="#4ecdc4" />
          <h2>Waiting for LineLens Data...</h2>
          <p>Please ensure the camera is running and data is being uploaded.</p>
        </div>
      </div>
    );
  }

  // Parse data
  const cycles = data.cycles || [];
  const idle_data = data.idle_data || [];
  const ergo_violations = data.ergo_violations || [];
  const events = data.events || [];
  const near_misses = data.near_misses || [];
  const queue_events = data.queue_events || [];
  const walk_positions = data.walk_positions || { walk_m: 0 };
  const sop_result = data.sop_result || { drift_pct: 0 };
  const heatmap_base64 = data.heatmap_base64;
  const spaghetti_base64 = data.spaghetti_base64;
  const fixes = data.fixes || [];

  // Stats
  const num_cycles = cycles.length;
  const avg_cycle = num_cycles > 0 ? (cycles.reduce((acc, c) => acc + (c.dur || 0), 0) / num_cycles).toFixed(1) : 0;
  const total_idle = idle_data.reduce((acc, e) => acc + ((e.end_time || 0) - (e.start_time || 0)), 0).toFixed(0);
  const total_walk = walk_positions.walk_m ? walk_positions.walk_m.toFixed(1) : 0;
  const num_violations = ergo_violations.length;
  const num_near_misses = near_misses.length;
  const num_queues = queue_events.length;
  const sop_drift = sop_result.drift_pct ? sop_result.drift_pct.toFixed(0) : 0;

  // Chart defaults
  const T = '#4ecdc4', O = '#ffd93d', R = '#ff6b6b', B = '#4a90d9', P = '#a78bfa', gC = 'rgba(255,255,255,0.07)';
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#ccc' } } },
    scales: {
      x: { ticks: { color: '#aaa' }, grid: { color: gC } },
      y: { ticks: { color: '#aaa' }, grid: { color: gC } }
    }
  };

  // 1. Cycle Times
  const cycleData = {
    labels: cycles.map((_, i) => `Cycle ${i+1}`),
    datasets: [{
      label: 'Duration (s)',
      data: cycles.map(c => c.dur),
      backgroundColor: T,
      borderRadius: 5
    }]
  };

  // 2. Ergo violations
  const bendCount = ergo_violations.filter(v => v.type === 'bend').length;
  const squatCount = ergo_violations.filter(v => v.type === 'squat').length;
  const overheadCount = ergo_violations.filter(v => v.type === 'overhead').length;
  
  const ergoData = {
    labels: ['Bend', 'Squat', 'Overhead'],
    datasets: [{
      data: [bendCount, squatCount, overheadCount],
      backgroundColor: [R, '#ff9f43', O],
      borderWidth: 2,
      borderColor: '#0b172e'
    }]
  };

  // 3. Idle Durations
  const idleChartData = {
    labels: idle_data.map((_, i) => `Idle ${i+1}`),
    datasets: [{
      label: 'Idle (s)',
      data: idle_data.map(e => (e.end_time - e.start_time)),
      backgroundColor: O,
      borderRadius: 5
    }]
  };

  // 4. Zone visits
  const zoneCounts = {};
  events.filter(e => e.type === 'zone_entry').forEach(e => {
    zoneCounts[e.zone] = (zoneCounts[e.zone] || 0) + 1;
  });
  
  const zoneData = {
    labels: Object.keys(zoneCounts),
    datasets: [{
      label: 'Entries',
      data: Object.values(zoneCounts),
      backgroundColor: B,
      borderRadius: 5
    }]
  };
  const zoneOptions = { ...baseOptions, indexAxis: 'y' };

  // 5. Near Misses
  const t0 = near_misses.length > 0 ? Math.min(...near_misses.map(nm => nm.ts)) : 0;
  const nmData = {
    labels: near_misses.map(nm => `T+${((nm.ts - t0)/60).toFixed(1)}m`),
    datasets: [{
      label: 'Distance (px)',
      data: near_misses.map(nm => nm.dist),
      borderColor: R,
      backgroundColor: 'rgba(255,107,107,0.15)',
      fill: true,
      tension: 0.35,
      pointRadius: 5,
      pointBackgroundColor: R
    }]
  };

  // 6. Queue Events
  const queueChartData = {
    labels: queue_events.map((_, i) => `Q${i+1}`),
    datasets: [{
      label: 'People queued',
      data: queue_events.map(q => q.count),
      backgroundColor: P,
      borderRadius: 5
    }]
  };

  // 7. SOP Drift Gauge
  const driftNum = parseFloat(sop_drift);
  const sopData = {
    labels: ['Drift', 'On-track'],
    datasets: [{
      data: [driftNum, Math.max(0, 100 - driftNum)],
      backgroundColor: [driftNum > 20 ? R : T, 'rgba(255,255,255,0.07)'],
      borderWidth: 0
    }]
  };
  const sopOptions = {
    responsive: true,
    maintainAspectRatio: false,
    circumference: 180,
    rotation: -90,
    cutout: '72%',
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${ctx.parsed.toFixed(1)}%` } }
    }
  };

  return (
    <div className="dashboard-container">
      <header>
        <h1>🏭 LineLens AI — Shift Report</h1>
        <div className="last-updated">
          <span className="polling-badge">
            <span className="polling-dot"></span> Live
          </span>
          Last updated: {lastUpdated}
        </div>
      </header>

      {/* Stats Grid */}
      <div className="grid">
        <div className="card"><div className="label">Total Cycles</div><div className="number green">{num_cycles}</div></div>
        <div className="card"><div className="label">Avg Cycle Time</div><div className="number">{avg_cycle}s</div></div>
        <div className="card"><div className="label">Total Idle Time</div><div className="number yellow">{total_idle}s</div></div>
        <div className="card"><div className="label">Walk Distance</div><div className="number">{total_walk}m</div></div>
        <div className="card"><div className="label">Ergo Violations</div><div className={`number ${num_violations > 5 ? 'red' : 'yellow'}`}>{num_violations}</div></div>
        <div className="card"><div className="label">Near Misses</div><div className="number red">{num_near_misses}</div></div>
        <div className="card"><div className="label">Queue Events</div><div className="number yellow">{num_queues}</div></div>
        <div className="card"><div className="label">SOP Drift</div><div className={`number ${driftNum > 20 ? 'red' : 'green'}`}>{sop_drift}%</div></div>
      </div>

      {/* Analytics Charts */}
      <div className="section">
        <h2>📊 Detection Analytics</h2>
        <div className="chart-grid">
          
          <div className="chart-card">
            <h3>⏱ Cycle Times</h3>
            <div className="chart-container">
              {cycles.length > 0 ? <Bar data={cycleData} options={baseOptions} /> : <p className="no-data">No cycles completed this session</p>}
            </div>
          </div>

          <div className="chart-card">
            <h3>🦺 Ergonomic Violations</h3>
            <div className="chart-container">
              {num_violations > 0 ? <Doughnut data={ergoData} options={{...baseOptions, maintainAspectRatio: false}} /> : <p className="no-data">No ergonomic violations — great shift!</p>}
            </div>
          </div>

          <div className="chart-card">
            <h3>😴 Idle Events</h3>
            <div className="chart-container">
              {idle_data.length > 0 ? <Bar data={idleChartData} options={baseOptions} /> : <p className="no-data">No idle events recorded</p>}
            </div>
          </div>

          <div className="chart-card">
            <h3>📍 Zone Activity</h3>
            <div className="chart-container">
              {Object.keys(zoneCounts).length > 0 ? <Bar data={zoneData} options={zoneOptions} /> : <p className="no-data">No zone data</p>}
            </div>
          </div>

          <div className="chart-card">
            <h3>⚠️ Near-Miss Proximity</h3>
            <div className="chart-container">
              {near_misses.length > 0 ? <Line data={nmData} options={baseOptions} /> : <p className="no-data">No near-miss incidents — safe shift!</p>}
            </div>
          </div>

          <div className="chart-card">
            <h3>🚦 Queue / Bottleneck Events</h3>
            <div className="chart-container">
              {queue_events.length > 0 ? <Bar data={queueChartData} options={baseOptions} /> : <p className="no-data">No queue events</p>}
            </div>
          </div>

        </div>

        {/* SOP Gauge */}
        <div className="sop-row">
          <div className="chart-card">
            <h3>📋 SOP Drift Gauge</h3>
            <div className="chart-container" style={{ height: '150px' }}>
              <Doughnut data={sopData} options={sopOptions} />
            </div>
            <p style={{ textAlign: 'center', color: '#888', fontSize: '0.78em', marginTop: '4px' }}>{sop_drift}% deviation</p>
          </div>
          <div className="sop-hint">
            <b className="primary">SOP Compliance</b> tracks whether workers visited zones in the correct expected sequence.<br/>
            <b className="perfect">0%</b> = perfect compliance &nbsp;|&nbsp; <b className="bad">&gt;20%</b> = retraining recommended
          </div>
        </div>
      </div>

      {/* Visual Maps */}
      {(heatmap_base64 || spaghetti_base64) && (
        <div className="section">
          <h2>🗺 Visual Maps</h2>
          <div className="img-grid">
            {heatmap_base64 && (
              <div className="img-card">
                <h3>🌡 Occupancy Heatmap</h3>
                <img src={`data:image/png;base64,${heatmap_base64}`} alt="Occupancy Heatmap" />
              </div>
            )}
            {spaghetti_base64 && (
              <div className="img-card">
                <h3>🍝 Walk Spaghetti Diagram</h3>
                <img src={`data:image/png;base64,${spaghetti_base64}`} alt="Spaghetti Diagram" />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Process Fixes */}
      <div className="section">
        <h2>🔧 Top 3 Process Fixes</h2>
        {fixes && fixes.length > 0 ? (
          fixes.map((f, i) => (
            <div key={i} className="fix">
              <h3>#{i+1}: {f.title}</h3>
              <p>{f.description}</p>
              <p className="impact">▲ Impact: {f.impact}</p>
            </div>
          ))
        ) : (
          <p className="no-data">No AI recommendations generated this session.</p>
        )}
      </div>

      <footer>Generated by LineLens AI &nbsp;·&nbsp; Privacy-first on-device analytics</footer>
    </div>
  );
}

export default App;
