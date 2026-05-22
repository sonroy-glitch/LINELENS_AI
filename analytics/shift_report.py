# shift report generator
# reads all json data files from a session and builds a rich, self-contained HTML report
# charts for every detection type via Chart.js (CDN)
# top 3 process fixes via Groq LLM API
# heatmap + spaghetti diagram embedded as base64

import json, os, time, base64, requests
from datetime import datetime
from groq import Groq

# ── Groq LLM config ──────────────────────────────────────────────────────────
GROQ_API_KEY = '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'

client=Groq(api_key=GROQ_API_KEY)
def get_fixes_from_groq(stats: dict) -> list:
    """Call Groq and return a list of (title, description, impact) tuples."""
    prompt = f"""You are an Industrial Engineering AI assistant analyzing live factory floor session data.

Session metrics:
- Total work cycles completed : {stats['cycles']}
- Average cycle time          : {stats['avg_cycle']:.1f} s
- Total idle time             : {stats['total_idle']:.0f} s
- Total walk distance         : {stats['total_walk']:.1f} m
- Ergonomic violations        : {stats['violations']} (bend / squat / overhead)
- Near-miss incidents         : {stats['near_misses']}
- Queue / bottleneck events   : {stats['queues']}
- SOP (process order) drift   : {stats['sop_drift']:.0f}%

Based on this data, give exactly 3 specific, actionable improvement recommendations.
Respond ONLY with a valid JSON array (no markdown, no explanation), like:
[
  {{"title": "...", "description": "...", "impact": "..."}},
  {{"title": "...", "description": "...", "impact": "..."}},
  {{"title": "...", "description": "...", "impact": "..."}}
]"""

    try:
        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages= [{"role": "user", "content": prompt}],
                            temperature=1, max_completion_tokens=600
                        )
        resp = completion.choices[0].message.content
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        fixes_raw = json.loads(raw)
        return [(f["title"], f["description"], f["impact"]) for f in fixes_raw[:3]]
    except Exception as e:
        print(f"[Groq] API call failed - using rule-based fallback. Error: {e}")
        fallback = []
        if stats["violations"] > 5:
            fallback.append(("Reduce Ergonomic Violations",
                "Workers exceeded safe posture limits multiple times. Schedule a posture and ergonomics briefing.",
                f"{stats['violations']} violations detected this shift"))
        if stats["near_misses"] > 0:
            fallback.append(("Address Near-Miss Incidents",
                "Workers came within the safety distance threshold. Review aisle markings and add floor signage.",
                f"{stats['near_misses']} near-miss events recorded"))
        if stats["total_idle"] > 60:
            fallback.append(("Reduce Idle Time",
                "Significant idle time recorded. Review workstation layout to ensure tools are within easy reach.",
                f"{stats['total_idle']:.0f}s of idle time this shift"))
        if stats["sop_drift"] > 20:
            fallback.append(("Improve SOP Adherence",
                f"Process order deviation was {stats['sop_drift']:.0f}%. Retrain workers on the correct zone visit sequence.",
                "Deviation from standard work sequence"))
        if stats["queues"] > 0:
            fallback.append(("Resolve Bottleneck / Queue",
                "Multiple workers were idle simultaneously, indicating a bottleneck. Balance workload between stations.",
                f"{stats['queues']} queue events"))
        return fallback[:3]


# ── load data ─────────────────────────────────────────────────────────────────
def load_json(path, default=[]):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"Warning: {path} not found, using default.")
    return default

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "..", "data")

cycles       = load_json(os.path.join(data_dir, "cycles.json"), [])
idles        = load_json(os.path.join(data_dir, "idle.json"), [])
walk         = load_json(os.path.join(data_dir, "walk_positions.json"), {"walk_m": 0})
events       = load_json(os.path.join(data_dir, "events.json"), [])
violations   = load_json(os.path.join(data_dir, "ergo_violations.json"), [])
near_misses  = load_json(os.path.join(data_dir, "near_misses.json"), [])
queue_events = load_json(os.path.join(data_dir, "queue_events.json"), [])
sop          = load_json(os.path.join(data_dir, "sop_result.json"), {"drift_pct": 0})

# ── stats ─────────────────────────────────────────────────────────────────────
num_cycles      = len(cycles)
avg_cycle       = sum(c.get("dur", 0) for c in cycles) / max(num_cycles, 1)
total_idle      = sum(e.get("end_time", 0) - e.get("start_time", 0) for e in idles)
total_walk      = walk.get("walk_m", 0) if isinstance(walk, dict) else 0
num_violations  = len(violations)
num_near_misses = len(near_misses)
num_queues      = len(queue_events)
sop_drift       = sop.get("drift_pct", 0) if isinstance(sop, dict) else 0

now = datetime.now()

# ── chart data prep ───────────────────────────────────────────────────────────
cycle_durs   = [round(c.get("dur", 0), 1) for c in cycles]
cycle_labels = [f"Cycle {i+1}" for i in range(len(cycle_durs))]

bend_count     = sum(1 for v in violations if v.get("type") == "bend")
squat_count    = sum(1 for v in violations if v.get("type") == "squat")
overhead_count = sum(1 for v in violations if v.get("type") == "overhead")

idle_durs   = [round(e.get("end_time", 0) - e.get("start_time", 0), 1) for e in idles]
idle_labels = [f"Idle {i+1}" for i in range(len(idle_durs))]

zone_counts = {}
for e in events:
    if e.get("type") == "zone_entry":
        z = e.get("zone", "unknown")
        zone_counts[z] = zone_counts.get(z, 0) + 1
zone_names  = list(zone_counts.keys())
zone_values = [zone_counts[z] for z in zone_names]

_sess_t0     = min((nm["ts"] for nm in near_misses), default=time.time())
nm_labels    = [f"T+{round((nm['ts'] - _sess_t0) / 60, 1)}m" for nm in near_misses]
nm_dists     = [round(nm.get("dist", 0), 1) for nm in near_misses]

queue_labels = [f"Q{i+1}" for i in range(len(queue_events))]
queue_cnts   = [q.get("count", 0) for q in queue_events]

def _b64(p):
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

heatmap_b64   = _b64(os.path.join(data_dir, "heatmap.png"))
spaghetti_b64 = _b64(os.path.join(data_dir, "spaghetti_diagram.png"))

# ── inject Python chart data as JS variables (f-string: {{ }} → literal { }) ─
_js_data = f"""<script>
const _cycles = {{labels:{json.dumps(cycle_labels)},data:{json.dumps(cycle_durs)}}};
const _ergo   = {{labels:['Bend','Squat','Overhead'],data:[{bend_count},{squat_count},{overhead_count}]}};
const _idle   = {{labels:{json.dumps(idle_labels)},data:{json.dumps(idle_durs)}}};
const _zones  = {{labels:{json.dumps(zone_names)},data:{json.dumps(zone_values)}}};
const _nm     = {{labels:{json.dumps(nm_labels)},data:{json.dumps(nm_dists)}}};
const _queues = {{labels:{json.dumps(queue_labels)},data:{json.dumps(queue_cnts)}}};
const _drift  = {sop_drift:.1f};
</script>"""

# ── Chart.js rendering code (plain string — no Python f-string, so {{ }} not needed) ──
_js_render = """<script>
const T='#4ecdc4',O='#ffd93d',R='#ff6b6b',B='#4a90d9',P='#a78bfa';
const gC='rgba(255,255,255,0.07)';
const base={
  responsive:true,maintainAspectRatio:true,
  plugins:{legend:{labels:{color:'#ccc'}}},
  scales:{x:{ticks:{color:'#aaa'},grid:{color:gC}},y:{ticks:{color:'#aaa'},grid:{color:gC}}}
};
function noData(id,msg){
  const c=document.getElementById(id); if(!c)return;
  c.insertAdjacentHTML('afterend','<p class="no-data">'+msg+'</p>');
  c.remove();
}

// 1. Cycle Times
if(_cycles.data.length){
  new Chart(document.getElementById('cCycles'),{type:'bar',
    data:{labels:_cycles.labels,datasets:[{label:'Duration (s)',data:_cycles.data,
      backgroundColor:T,borderRadius:5,borderSkipped:false}]},
    options:base});
}else noData('cCycles','No cycles completed this session');

// 2. Ergo breakdown doughnut
if(_ergo.data.reduce((a,b)=>a+b,0)>0){
  new Chart(document.getElementById('cErgo'),{type:'doughnut',
    data:{labels:_ergo.labels,datasets:[{data:_ergo.data,
      backgroundColor:[R,'#ff9f43',O],borderWidth:2,borderColor:'#0b172e'}]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#ccc'}}}}});
}else noData('cErgo','No ergonomic violations — great shift!');

// 3. Idle durations
if(_idle.data.length){
  new Chart(document.getElementById('cIdle'),{type:'bar',
    data:{labels:_idle.labels,datasets:[{label:'Idle (s)',data:_idle.data,
      backgroundColor:O,borderRadius:5,borderSkipped:false}]},
    options:base});
}else noData('cIdle','No idle events recorded');

// 4. Zone visits (horizontal)
if(_zones.data.length){
  new Chart(document.getElementById('cZones'),{type:'bar',
    data:{labels:_zones.labels,datasets:[{label:'Entries',data:_zones.data,
      backgroundColor:B,borderRadius:5,borderSkipped:false}]},
    options:{...base,indexAxis:'y'}});
}else noData('cZones','No zone data — run zone calibration first');

// 5. Near-miss proximity over session
if(_nm.data.length){
  new Chart(document.getElementById('cNM'),{type:'line',
    data:{labels:_nm.labels,datasets:[{label:'Distance (px)',data:_nm.data,
      borderColor:R,backgroundColor:'rgba(255,107,107,0.15)',fill:true,tension:0.35,pointRadius:5,pointBackgroundColor:R}]},
    options:base});
}else noData('cNM','No near-miss incidents — safe shift!');

// 6. Queue/bottleneck events
if(_queues.data.length){
  new Chart(document.getElementById('cQueues'),{type:'bar',
    data:{labels:_queues.labels,datasets:[{label:'People queued',data:_queues.data,
      backgroundColor:P,borderRadius:5,borderSkipped:false}]},
    options:base});
}else noData('cQueues','No queue / bottleneck events');

// 7. SOP drift half-gauge
new Chart(document.getElementById('cSOP'),{type:'doughnut',
  data:{labels:['Drift','On-track'],datasets:[{
    data:[_drift,Math.max(0,100-_drift)],
    backgroundColor:[_drift>20?R:T,'rgba(255,255,255,0.07)'],
    borderWidth:0
  }]},
  options:{
    responsive:true,circumference:180,rotation:-90,cutout:'72%',
    plugins:{
      legend:{display:false},
      tooltip:{callbacks:{label:ctx=>ctx.label+': '+ctx.parsed.toFixed(1)+'%'}}
    }
  }
});
</script>"""

# ── assemble HTML ─────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LineLens AI — Shift Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f0f1a;color:#eee;padding:28px;line-height:1.5}}
h1{{color:#4ecdc4;text-align:center;font-size:2em;margin-bottom:6px;letter-spacing:-.01em}}
.meta{{text-align:center;color:#555;margin-bottom:30px;font-size:0.85em}}
/* stat cards */
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px}}
.card{{background:#16213e;border-radius:12px;padding:20px;text-align:center;border:1px solid rgba(255,255,255,0.05);transition:transform .2s}}
.card:hover{{transform:translateY(-2px)}}
.card .number{{font-size:2em;font-weight:700;margin:8px 0}}
.card .label{{color:#667;font-size:0.75em;text-transform:uppercase;letter-spacing:.07em}}
.green{{color:#4ecdc4}}.yellow{{color:#ffd93d}}.red{{color:#ff6b6b}}
/* sections */
.section{{background:#16213e;border-radius:12px;padding:24px;margin-bottom:20px;border:1px solid rgba(255,255,255,0.05)}}
.section h2{{color:#4ecdc4;margin-bottom:18px;font-size:0.9em;text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:10px}}
/* chart grid */
.chart-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-bottom:16px}}
.chart-card{{background:#0b172e;border-radius:10px;padding:16px;border:1px solid rgba(255,255,255,0.06)}}
.chart-card h3{{color:#7a8fa0;font-size:0.73em;text-transform:uppercase;letter-spacing:.09em;margin-bottom:14px}}
.chart-card canvas{{max-height:200px!important}}
.no-data{{color:#3a3a4a;font-size:0.8em;text-align:center;padding:40px 0;font-style:italic}}
/* SOP gauge row */
.sop-row{{display:grid;grid-template-columns:260px 1fr;gap:18px;align-items:center}}
.sop-hint{{color:#666;font-size:0.88em;line-height:2;padding-left:8px}}
/* fixes */
.fix{{background:#0f3460;border-left:3px solid #ffd93d;padding:14px 16px;margin:10px 0;border-radius:6px}}
.fix h3{{color:#ffd93d;margin-bottom:5px;font-size:0.92em}}
.fix p{{color:#bbb;font-size:0.86em;line-height:1.6}}
.impact{{color:#ff6b6b!important;font-weight:700;margin-top:6px;font-size:0.8em}}
/* images */
.img-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}
.img-card{{background:#0b172e;border-radius:10px;padding:14px;border:1px solid rgba(255,255,255,0.06)}}
.img-card h3{{color:#7a8fa0;font-size:0.73em;text-transform:uppercase;letter-spacing:.09em;margin-bottom:10px}}
.img-card img{{width:100%;border-radius:6px;display:block}}
footer{{text-align:center;color:#252535;margin-top:28px;padding-top:14px;border-top:1px solid #1a1a2a;font-size:0.72em}}
@media(max-width:860px){{
  .grid{{grid-template-columns:repeat(2,1fr)}}
  .chart-grid,.img-grid,.sop-row{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>

<h1>🏭 LineLens AI — Shift Report</h1>
<div class="meta">{now.strftime('%B %d, %Y &nbsp;&nbsp;|&nbsp;&nbsp; %H:%M')}</div>

<!-- ── stat overview cards ── -->
<div class="grid">
  <div class="card"><div class="label">Total Cycles</div><div class="number green">{num_cycles}</div></div>
  <div class="card"><div class="label">Avg Cycle Time</div><div class="number">{avg_cycle:.1f}s</div></div>
  <div class="card"><div class="label">Total Idle Time</div><div class="number yellow">{total_idle:.0f}s</div></div>
  <div class="card"><div class="label">Walk Distance</div><div class="number">{total_walk:.1f}m</div></div>
  <div class="card"><div class="label">Ergo Violations</div><div class="number {'red' if num_violations > 5 else 'yellow'}">{num_violations}</div></div>
  <div class="card"><div class="label">Near Misses</div><div class="number red">{num_near_misses}</div></div>
  <div class="card"><div class="label">Queue Events</div><div class="number yellow">{num_queues}</div></div>
  <div class="card"><div class="label">SOP Drift</div><div class="number {'red' if sop_drift > 20 else 'green'}">{sop_drift:.0f}%</div></div>
</div>

<!-- ── detection analytics charts ── -->
<div class="section">
  <h2>📊 Detection Analytics</h2>
  <div class="chart-grid">

    <div class="chart-card">
      <h3>⏱ Cycle Times</h3>
      <canvas id="cCycles"></canvas>
    </div>

    <div class="chart-card">
      <h3>🦺 Ergonomic Violations — Bend / Squat / Overhead</h3>
      <canvas id="cErgo"></canvas>
    </div>

    <div class="chart-card">
      <h3>😴 Idle Events (duration per event)</h3>
      <canvas id="cIdle"></canvas>
    </div>

    <div class="chart-card">
      <h3>📍 Zone Activity (entries per zone)</h3>
      <canvas id="cZones"></canvas>
    </div>

    <div class="chart-card">
      <h3>⚠️ Near-Miss Proximity over Session</h3>
      <canvas id="cNM"></canvas>
    </div>

    <div class="chart-card">
      <h3>🚦 Queue / Bottleneck Events (people count)</h3>
      <canvas id="cQueues"></canvas>
    </div>

  </div>

  <!-- SOP drift half-gauge -->
  <div class="sop-row">
    <div class="chart-card">
      <h3>📋 SOP Drift Gauge</h3>
      <canvas id="cSOP" style="max-height:150px!important"></canvas>
      <p style="text-align:center;color:#888;font-size:0.78em;margin-top:4px">{sop_drift:.1f}% deviation from standard order</p>
    </div>
    <div class="sop-hint">
      <b style="color:#4ecdc4">SOP Compliance</b> tracks whether workers visited zones in the correct expected sequence.<br>
      <b style="color:#ffd93d">0%</b> = perfect compliance &nbsp;|&nbsp;
      <b style="color:#ff6b6b">&gt;20%</b> = retraining recommended
    </div>
  </div>
</div>
"""

# ── visual maps (heatmap + spaghetti — embedded as base64) ───────────────────
if heatmap_b64 or spaghetti_b64:
    html += '<div class="section"><h2>🗺 Visual Maps</h2><div class="img-grid">\n'
    if heatmap_b64:
        html += (f'<div class="img-card"><h3>🌡 Occupancy Heatmap</h3>'
                 f'<img src="data:image/png;base64,{heatmap_b64}" alt="Occupancy Heatmap"></div>\n')
    if spaghetti_b64:
        html += (f'<div class="img-card"><h3>🍝 Walk Spaghetti Diagram</h3>'
                 f'<img src="data:image/png;base64,{spaghetti_b64}" alt="Spaghetti Diagram"></div>\n')
    html += '</div></div>\n'

# ── top 3 process fixes via Groq ─────────────────────────────────────────────
html += '<div class="section"><h2>🔧 Top 3 Process Fixes</h2>\n'

print("  Calling Groq for process fix recommendations...")
fixes = get_fixes_from_groq({
    "cycles":      num_cycles,
    "avg_cycle":   avg_cycle,
    "total_idle":  total_idle,
    "total_walk":  total_walk,
    "violations":  num_violations,
    "near_misses": num_near_misses,
    "queues":      num_queues,
    "sop_drift":   sop_drift,
})

if fixes:
    for i, (title, desc, impact) in enumerate(fixes):
        html += f"""<div class="fix">
  <h3>#{i+1}: {title}</h3>
  <p>{desc}</p>
  <p class="impact">▲ Impact: {impact}</p>
</div>\n"""
else:
    html += "<p class='no-data'>No recommendations generated this session.</p>\n"

html += "</div>\n"

# ── footer + inject JS ────────────────────────────────────────────────────────
html += '<footer>Generated by LineLens AI &nbsp;·&nbsp; Privacy-first on-device analytics</footer>\n'
html += _js_data + "\n" + _js_render + "\n</body>\n</html>"

# ── save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(script_dir, "..")
fname   = os.path.join(out_dir, f"shift_report_{now.strftime('%Y%m%d_%H%M')}.html")
with open(fname, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nReport saved: {os.path.abspath(fname)}")
print("Open it in a browser to see the full report with charts.")
