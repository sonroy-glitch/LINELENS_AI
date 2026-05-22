import os
import json
import base64
import requests
import datetime
import time

def get_fixes_from_groq(stats: dict) -> list:
    """Call Groq and return a list of (title, description, impact) tuples."""
    try:
        from groq import Groq
    except ImportError:
        print("Warning: groq package not installed. Skipping LLM fixes.")
        return []
        
    GROQ_API_KEY = ''
    if not GROQ_API_KEY: return []
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
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
        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages= [{"role": "user", "content": prompt}],
                            temperature=1, max_completion_tokens=600
                        )
        resp = completion.choices[0].message.content
        raw = resp.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        fixes_raw = json.loads(raw)
        return [{"title": f["title"], "description": f["description"], "impact": f["impact"]} for f in fixes_raw[:3]]
    except Exception as e:
        print(f"[Groq] API call failed - using rule-based fallback. Error: {e}")
        fallback = []
        if stats["violations"] > 5:
            fallback.append({"title": "Reduce Ergonomic Violations",
                "description": "Workers exceeded safe posture limits multiple times. Schedule a posture and ergonomics briefing.",
                "impact": f"{stats['violations']} violations detected this shift"})
        if stats["near_misses"] > 0:
            fallback.append({"title": "Address Near-Miss Incidents",
                "description": "Workers came within the safety distance threshold. Review aisle markings and add floor signage.",
                "impact": f"{stats['near_misses']} near-miss events recorded"})
        if stats["total_idle"] > 60:
            fallback.append({"title": "Reduce Idle Time",
                "description": "Significant idle time recorded. Review workstation layout to ensure tools are within easy reach.",
                "impact": f"{stats['total_idle']:.0f}s of idle time this shift"})
        if stats["sop_drift"] > 20:
            fallback.append({"title": "Improve SOP Adherence",
                "description": f"Process order deviation was {stats['sop_drift']:.0f}%. Retrain workers on the correct zone visit sequence.",
                "impact": "Deviation from standard work sequence"})
        if stats["queues"] > 0:
            fallback.append({"title": "Resolve Bottleneck / Queue",
                "description": "Multiple workers were idle simultaneously, indicating a bottleneck. Balance workload between stations.",
                "impact": f"{stats['queues']} queue events"})
        return fallback[:3]


def upload_to_supabase(data_dir="data"):
    SUPABASE_URL = os.getenv("VITE_SUPABASE_URL", "https://smhppukwdhopgaaqxzur.supabase.co")
    SUPABASE_KEY = os.getenv("VITE_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNtaHBwdWt3ZGhvcGdhYXF4enVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkyMDgxMjUsImV4cCI6MjA5NDc4NDEyNX0.BIjB89yf_iaaJoI-lR5ueVm_05WD_87h30n4dp4lUbY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Warning: VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY not found in environment.")
        print("Set them to enable Supabase uploading.")
        return
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "idle_data": [],
        "walk_positions": {},
        "ergo_violations": [],
        "near_misses": [],
        "queue_events": [],
        "events": [],
        "cycles": [],
        "sop_result": {},
        "heatmap_base64": "",
        "spaghetti_base64": "",
        "fixes": []
    }
    
    # Read json files
    for key, filename in [
        ("idle_data", "idle.json"),
        ("walk_positions", "walk_positions.json"),
        ("ergo_violations", "ergo_violations.json"),
        ("near_misses", "near_misses.json"),
        ("queue_events", "queue_events.json"),
        ("events", "events.json"),
        ("cycles", "cycles.json"),
        ("sop_result", "sop_result.json"),
    ]:
        try:
            with open(os.path.join(data_dir, filename), "r") as f:
                report_data[key] = json.load(f)
        except Exception as e:
            pass
            
    # Read images as base64
    for key, filename in [
        ("heatmap_base64", "heatmap.png"),
        ("spaghetti_base64", "spaghetti_diagram.png")
    ]:
        try:
            with open(os.path.join(data_dir, filename), "rb") as f:
                report_data[key] = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            pass
            
    # --- Generate fixes via LLM ---
    num_cycles = len(report_data.get("cycles", []))
    avg_cycle = sum(c.get("dur", 0) for c in report_data.get("cycles", [])) / max(num_cycles, 1)
    total_idle = sum(e.get("end_time", 0) - e.get("start_time", 0) for e in report_data.get("idle_data", []))
    walk_positions = report_data.get("walk_positions", {})
    total_walk = walk_positions.get("walk_m", 0) if isinstance(walk_positions, dict) else 0
    num_violations = len(report_data.get("ergo_violations", []))
    num_near_misses = len(report_data.get("near_misses", []))
    num_queues = len(report_data.get("queue_events", []))
    sop = report_data.get("sop_result", {})
    sop_drift = sop.get("drift_pct", 0) if isinstance(sop, dict) else 0

    stats = {
        "cycles": num_cycles,
        "avg_cycle": avg_cycle,
        "total_idle": total_idle,
        "total_walk": total_walk,
        "violations": num_violations,
        "near_misses": num_near_misses,
        "queues": num_queues,
        "sop_drift": sop_drift,
    }
    print("Generating AI Process Fixes...")
    report_data["fixes"] = get_fixes_from_groq(stats)
            
    payload = {"report_data": report_data}
    url = f"{SUPABASE_URL}/rest/v1/linelens_reports"
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("Data successfully uploaded to Supabase.")
    except Exception as e:
        print(f"Failed to upload to Supabase: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(e.response.text)

if __name__ == "__main__":
    upload_to_supabase()
