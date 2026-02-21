# -*- coding: utf-8 -*-
"""
Alice Smart System — REST API Server
FastAPI-powered smart system API interface

Endpoints:
  GET  /                    : Welcome page (redirect to dashboard)
  GET  /api/status          : System status
  GET  /api/brain           : Brain state snapshot
  GET  /api/vitals          : ★ Vital signs real-time data
  GET  /api/waveforms       : ★ Waveform data (heartbeat/temperature/pain/left-right brain)
  GET  /api/oscilloscope    : ★ Oscilloscope data (channel waveforms/reflection/standing waves)
  POST /api/perceive        : Perceive stimulus
  POST /api/think           : Think/reason
  POST /api/act             : Action selection
  POST /api/learn           : Learning feedback
  POST /api/stabilize       : ★ Stabilize system (gentle homeostatic reset)
  POST /api/time-scale      : ★ Time accelerator (adjust tick multiplier)
  POST /api/dream-input     : ★ Dream input (inject language during REM)
  POST /api/administer-drug : ★ Pharmacology (administer drug by name)
  GET  /api/introspect      : Full introspection report
  GET  /api/working-memory  : Working memory contents
  GET  /api/causal-graph    : Causal graph
  GET  /api/stats           : Full statistics
  WS   /ws/stream           : WebSocket real-time streaming
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, RedirectResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


# ============================================================================
# Pydantic models
# ============================================================================

if HAS_FASTAPI:

    class PerceiveRequest(BaseModel):
        signal: List[float] = Field(..., description="Sensory stimulus signal (numeric array)")
        modality: str = Field("visual", description="Modality: visual/auditory/tactile/internal")
        priority: str = Field("NORMAL", description="Priority: BACKGROUND/NORMAL/HIGH/CRITICAL")
        context: Optional[str] = Field(None, description="Context tag")

    class ThinkRequest(BaseModel):
        question: str = Field(..., description="Question to think about")
        context_vars: Optional[Dict[str, float]] = Field(None, description="Context variables")

    class ActRequest(BaseModel):
        state: str = Field(..., description="Current state")
        available_actions: List[str] = Field(..., description="List of available actions")

    class LearnRequest(BaseModel):
        state: str
        action: str
        reward: float
        next_state: str
        next_actions: Optional[List[str]] = None


# ============================================================================
# Application factory
# ============================================================================


def create_app(alice: Optional[AliceBrain] = None) -> Any:
    """Create FastAPI application"""
    if not HAS_FASTAPI:
        raise ImportError("fastapi and uvicorn are required: pip install fastapi uvicorn")

    if alice is None:
        alice = AliceBrain()

    app = FastAPI(
        title="Alice Smart System",
        description="Biologically-inspired digital brain smart system based on Γ-Net architecture",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # WebSocket connection management
    ws_clients: List[WebSocket] = []

    async def broadcast(data: Dict):
        dead = []
        for ws in ws_clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.remove(ws)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=RedirectResponse)
    async def root():
        return RedirectResponse(url="/dashboard")

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return _get_dashboard_html()

    @app.get("/api/status")
    async def status():
        return {
            "system": "Alice Smart System",
            "version": "1.0.0",
            "status": "running",
            "uptime": round(time.time() - alice._start_time, 1),
            "cycles": alice._cycle_count,
        }

    @app.get("/api/brain")
    async def brain_state():
        return alice.fusion_brain.get_brain_state()

    # ★ Neuro-Physics vital signs endpoint
    @app.get("/api/vitals")
    async def vitals():
        return alice.get_vitals()

    @app.get("/api/waveforms")
    async def waveforms(last_n: int = 60):
        return alice.get_waveforms(last_n)

    @app.post("/api/stabilize")
    async def stabilize():
        """Gentle homeostatic stabilization — gradually return vitals toward baseline."""
        alice.emergency_reset()
        result = alice.get_vitals()
        await broadcast({"event": "stabilize", "data": result})
        return result

    @app.post("/api/time-scale")
    async def time_scale(multiplier: float = 1.0, ticks: int = 100):
        """Time accelerator — run N ticks at accelerated pace.

        Args:
            multiplier: Display label only (physics is tick-based, so 1× or 100× use the same equations).
            ticks: Number of ticks to advance (max 10000).
        """
        ticks = min(ticks, 10000)
        results = []
        for i in range(ticks):
            # Run a quiet perceive cycle (internal tick)
            r = alice.perceive(np.zeros(10), Modality.INTERNAL, Priority.BACKGROUND, "time_accel")
            if i % max(1, ticks // 10) == 0:
                results.append({
                    "tick": i,
                    "vitals": {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in alice.get_vitals().items()},
                })
        final = alice.get_vitals()
        await broadcast({"event": "time_scale", "data": {"ticks": ticks, "multiplier": multiplier, "final_vitals": final}})
        return {
            "ticks_advanced": ticks,
            "multiplier_label": f"{multiplier}×",
            "final_vitals": final,
            "trajectory": results,
        }

    @app.post("/api/dream-input")
    async def dream_input(text: str = "mama", language: str = "proto"):
        """Inject language stimulus during REM sleep for language acquisition verification.

        If Alice is not sleeping, this gently nudges sleep pressure up first.
        The stimulus is delivered as an auditory signal during REM phase.
        """
        is_sleeping = alice.sleep_cycle.is_sleeping()
        # Encode text as simple frequency pattern
        encoded = np.array([float(ord(c)) / 127.0 for c in text[:50]], dtype=np.float64)
        if not is_sleeping:
            # Nudge toward sleep so the dream can be received
            alice.sleep_cycle.sleep_pressure = max(alice.sleep_cycle.sleep_pressure, 0.75)

        result = alice.perceive(encoded, Modality.AUDITORY, Priority.BACKGROUND, f"dream_{language}")
        output = {
            "text": text,
            "language": language,
            "was_sleeping": is_sleeping,
            "signal_length": len(encoded),
            "vitals": alice.get_vitals(),
        }
        await broadcast({"event": "dream_input", "data": output})
        return output

    @app.post("/api/administer-drug")
    async def administer_drug(drug_name: str = "SSRI", alpha: Optional[float] = None):
        """Administer a drug via the pharmacology engine.

        Predefined drugs: L-DOPA, Valproate, Carbamazepine, SSRI.
        Or specify custom alpha for research.
        """
        from alice.brain.pharmacology import DrugProfile, ALL_CHANNELS

        DRUG_PRESETS = {
            "L-DOPA": DrugProfile(name="L-DOPA", alpha=-0.35,
                target_channels=["basal_ganglia", "motor", "prefrontal"],
                onset_delay=10, half_life=200),
            "Valproate": DrugProfile(name="Valproate", alpha=-0.25,
                target_channels=["temporal", "hippocampus", "thalamus"],
                onset_delay=30, half_life=400),
            "Carbamazepine": DrugProfile(name="Carbamazepine", alpha=-0.20,
                target_channels=["temporal", "hippocampus"],
                onset_delay=20, half_life=350),
            "SSRI": DrugProfile(name="SSRI", alpha=-0.20,
                target_channels=["amygdala", "prefrontal", "hippocampus", "insular"],
                onset_delay=300, half_life=600),
        }

        drug_key = drug_name.upper().replace("-", "_").replace(" ", "_")
        # Try preset match
        preset = None
        for k, v in DRUG_PRESETS.items():
            if k.upper().replace("-", "_") == drug_key:
                preset = v
                break

        if preset:
            drug = preset
        else:
            # Custom drug with user-specified alpha
            _alpha = alpha if alpha is not None else -0.15
            drug = DrugProfile(name=drug_name, alpha=_alpha,
                target_channels=list(ALL_CHANNELS)[:5],
                onset_delay=10, half_life=300)

        alice.pharmacology.administer(drug)
        output = {
            "drug": drug.name,
            "alpha": drug.alpha,
            "targets": drug.target_channels,
            "onset_delay": drug.onset_delay,
            "half_life": drug.half_life,
            "active_drugs": len(alice.pharmacology.active_drugs),
            "vitals": alice.get_vitals(),
        }
        await broadcast({"event": "administer_drug", "data": output})
        return output

    @app.post("/api/perceive")
    async def perceive(req: PerceiveRequest):
        signal = np.array(req.signal)
        modality = Modality(req.modality)
        priority = Priority[req.priority]
        result = alice.perceive(signal, modality, priority, req.context)
        await broadcast({"event": "perceive", "data": _sanitize(result)})
        return _sanitize(result)

    @app.post("/api/think")
    async def think(req: ThinkRequest):
        result = alice.think(req.question, req.context_vars)
        await broadcast({"event": "think", "data": _sanitize(result)})
        return _sanitize(result)

    @app.post("/api/act")
    async def act(req: ActRequest):
        result = alice.act(req.state, req.available_actions)
        await broadcast({"event": "act", "data": _sanitize(result)})
        return _sanitize(result)

    @app.post("/api/learn")
    async def learn(req: LearnRequest):
        result = alice.learn_from_feedback(
            req.state, req.action, req.reward, req.next_state, req.next_actions
        )
        await broadcast({"event": "learn", "data": _sanitize(result)})
        return _sanitize(result)

    @app.get("/api/introspect")
    async def introspect():
        return _sanitize(alice.introspect())

    @app.get("/api/working-memory")
    async def working_memory():
        return alice.working_memory.get_contents()

    @app.get("/api/causal-graph")
    async def causal_graph():
        return alice.causal.get_causal_graph_summary()

    @app.get("/api/oscilloscope")
    async def oscilloscope():
        """Oscilloscope data: input waveform + channel waveforms + reflection coefficients + perception results"""
        return _sanitize(alice.get_oscilloscope_data())

    @app.get("/api/stats")
    async def stats():
        return _sanitize(alice.introspect()["subsystems"])

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        await websocket.accept()
        ws_clients.append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"pong": True})
                elif data == "status":
                    await websocket.send_json(_sanitize(alice.introspect()))
        except WebSocketDisconnect:
            if websocket in ws_clients:
                ws_clients.remove(websocket)

    return app


# ============================================================================
# Utility functions
# ============================================================================


def _sanitize(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable types"""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _get_dashboard_html() -> str:
    """Embedded Web dashboard HTML — Oscilloscope edition"""
    return """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alice — Oscilloscope</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            background: #020810;
            color: #88ccaa;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* === Header === */
        .header {
            background: #040c14;
            padding: 8px 20px;
            border-bottom: 1px solid #0a2a1a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 15px;
            color: #33ff99;
            letter-spacing: 3px;
            text-shadow: 0 0 12px rgba(51,255,153,0.4);
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 11px;
        }
        .heartbeat-indicator {
            width: 10px; height: 10px;
            border-radius: 50%;
            background: #33ff99;
            box-shadow: 0 0 8px rgba(51,255,153,0.6);
            animation: pulse 1s infinite;
        }
        .heartbeat-indicator.danger {
            background: #ff3344;
            box-shadow: 0 0 12px rgba(255,51,68,0.8);
            animation: pulse-fast 0.4s infinite;
        }
        .heartbeat-indicator.frozen {
            background: #333;
            box-shadow: none;
            animation: none;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.4); opacity: 0.6; }
        }
        @keyframes pulse-fast {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.6); opacity: 0.4; }
        }
        .hr-display {
            font-size: 18px;
            font-weight: bold;
            color: #33ff99;
            min-width: 70px;
            text-align: right;
        }
        .hr-display.danger { color: #ff3344; }
        .hr-display.frozen { color: #444; }

        /* === Scope Grid === */
        .scope-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr auto;
            gap: 1px;
            padding: 1px;
            height: calc(100vh - 42px);
            background: #0a1a12;
        }

        /* === Scope Panel (each channel) === */
        .scope-panel {
            background: #020810;
            border: 1px solid #0a2a1a;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .scope-label {
            position: absolute;
            top: 4px;
            left: 8px;
            font-size: 10px;
            color: #336644;
            letter-spacing: 1px;
            z-index: 2;
            text-transform: uppercase;
        }
        .scope-info {
            position: absolute;
            top: 4px;
            right: 8px;
            font-size: 10px;
            color: #3a6a4a;
            text-align: right;
            z-index: 2;
        }
        .scope-canvas-wrap {
            flex: 1;
            min-height: 0;
            position: relative;
        }
        .scope-canvas-wrap canvas {
            width: 100%;
            height: 100%;
            display: block;
        }

        /* === Bottom bar === */
        .bottom-bar {
            grid-column: 1 / -1;
            background: #040c14;
            border: 1px solid #0a2a1a;
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 8px;
            padding: 6px 10px;
            max-height: 160px;
        }

        /* === Controls === */
        .controls {
            display: flex;
            flex-direction: column;
            gap: 3px;
            min-width: 170px;
        }
        .controls h3 {
            font-size: 9px;
            color: #336644;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .btn-row {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 4px 10px;
            border: 1px solid #1a3a2a;
            border-radius: 2px;
            background: #061210;
            color: #88ccaa;
            cursor: pointer;
            font-family: inherit;
            font-size: 10px;
            transition: all 0.15s;
        }
        .btn:hover { border-color: #33ff99; color: #33ff99;
            box-shadow: 0 0 6px rgba(51,255,153,0.3); }
        .btn.danger { border-color: #ff3344; color: #ff3344; }
        .btn.danger:hover { background: #ff3344; color: #000; }
        .btn.heal { border-color: #22aaff; color: #22aaff; }
        .btn.heal:hover { background: #22aaff; color: #000; }
        .btn.storm { border-color: #ff8800; color: #ff8800; }
        .btn.storm:hover { background: #ff8800; color: #000; }

        /* === Vitals mini === */
        .vitals-mini {
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: 180px;
        }
        .vm-row {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 10px;
        }
        .vm-label { color: #336644; width: 55px; }
        .vm-bar-bg {
            flex: 1;
            height: 8px;
            background: #0a1610;
            border-radius: 1px;
            overflow: hidden;
        }
        .vm-bar {
            height: 100%;
            border-radius: 1px;
            transition: width 0.3s;
        }
        .vm-val {
            min-width: 38px;
            text-align: right;
            font-size: 10px;
            font-weight: bold;
        }

        /* === Event log === */
        .event-log-wrap {
            overflow-y: auto;
            font-size: 9px;
            line-height: 1.5;
            color: #3a5a4a;
        }
        .event-log-wrap .pain { color: #ff3344; }
        .event-log-wrap .warn { color: #ff9900; }
        .event-log-wrap .info { color: #22aaff; }
        .event-log-wrap .ok { color: #33ff99; }

        /* === Freeze overlay === */
        .freeze-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .freeze-overlay.active { display: flex; }
        .freeze-text {
            font-size: 42px;
            color: #ff3344;
            letter-spacing: 8px;
            animation: flicker 0.5s infinite;
        }
        .freeze-sub {
            font-size: 13px;
            color: #666;
            margin-top: 10px;
        }
        @keyframes flicker {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.2; }
        }
    </style>
</head>
<body>
    <div class="freeze-overlay" id="freeze-overlay">
        <div class="freeze-text">SYSTEM FREEZE</div>
        <div class="freeze-sub">Consciousness below threshold — only CRITICAL signals can penetrate</div>
        <button class="btn heal" onclick="stabilize()" style="margin-top:16px;font-size:14px;padding:10px 20px;">
            STABILIZE
        </button>
    </div>

    <div class="header">
        <h1>ALICE OSCILLOSCOPE</h1>
        <div class="header-right">
            <span id="sys-info" style="color:#336644;">Cycle 0 | 0s</span>
            <div class="heartbeat-indicator" id="hb-dot"></div>
            <div class="hr-display" id="hr-display">-- bpm</div>
        </div>
    </div>

    <div class="scope-grid">
        <!-- CH1: Input Signal -->
        <div class="scope-panel">
            <div class="scope-label">CH1 — Input Signal</div>
            <div class="scope-info" id="ch1-info">--</div>
            <div class="scope-canvas-wrap"><canvas id="scope-ch1"></canvas></div>
        </div>

        <!-- CH2: Coaxial Channels (Standing Waves) -->
        <div class="scope-panel">
            <div class="scope-label">CH2 — Coaxial Channels</div>
            <div class="scope-info" id="ch2-info">--</div>
            <div class="scope-canvas-wrap"><canvas id="scope-ch2"></canvas></div>
        </div>

        <!-- CH3: Perception / Resonance -->
        <div class="scope-panel">
            <div class="scope-label">CH3 — Perception &amp; Resonance</div>
            <div class="scope-info" id="ch3-info">--</div>
            <div class="scope-canvas-wrap"><canvas id="scope-ch3"></canvas></div>
        </div>

        <!-- CH4: Vital Signs -->
        <div class="scope-panel">
            <div class="scope-label">CH4 — Vital Signs</div>
            <div class="scope-info" id="ch4-info">--</div>
            <div class="scope-canvas-wrap"><canvas id="scope-ch4"></canvas></div>
        </div>

        <!-- Bottom bar: controls + vitals + log -->
        <div class="bottom-bar">
            <div class="controls">
                <h3>Controls</h3>
                <div class="btn-row">
                    <button class="btn" onclick="sendStimulus('NORMAL',1.0)">Normal</button>
                    <button class="btn danger" onclick="sendStimulus('CRITICAL',3.0)">CRITICAL</button>
                    <button class="btn" onclick="sendThink()">Think</button>
                    <button class="btn" onclick="sendAct()">Act</button>
                </div>
                <div class="btn-row">
                    <button class="btn heal" onclick="stabilize()">Stabilize</button>
                    <button class="btn" onclick="timeScale(100)">Accel×100</button>
                    <button class="btn" onclick="dreamInput()">Dream</button>
                    <button class="btn" onclick="administerDrug()">Drug</button>
                </div>
                <div class="vitals-mini" style="margin-top:4px;">
                    <div class="vm-row">
                        <span class="vm-label">Temp</span>
                        <div class="vm-bar-bg"><div class="vm-bar" id="vm-temp" style="width:0%;background:#ff3344;"></div></div>
                        <span class="vm-val" id="vv-temp">0</span>
                    </div>
                    <div class="vm-row">
                        <span class="vm-label">Pain</span>
                        <div class="vm-bar-bg"><div class="vm-bar" id="vm-pain" style="width:0%;background:#ff6600;"></div></div>
                        <span class="vm-val" id="vv-pain">0</span>
                    </div>
                    <div class="vm-row">
                        <span class="vm-label">Stab.</span>
                        <div class="vm-bar-bg"><div class="vm-bar" id="vm-stab" style="width:100%;background:#33ff99;"></div></div>
                        <span class="vm-val" id="vv-stab">1.0</span>
                    </div>
                    <div class="vm-row">
                        <span class="vm-label">Consc.</span>
                        <div class="vm-bar-bg"><div class="vm-bar" id="vm-cons" style="width:100%;background:#22aaff;"></div></div>
                        <span class="vm-val" id="vv-cons">1.0</span>
                    </div>
                </div>
            </div>
            <div class="event-log-wrap" id="event-log"></div>
            <div style="display:flex;flex-direction:column;gap:2px;min-width:120px;">
                <div style="font-size:9px;color:#336644;letter-spacing:1px;text-transform:uppercase;">Standing Wave</div>
                <canvas id="mini-standing" width="240" height="100" style="width:120px;height:50px;background:#020810;border:1px solid #0a2a1a;border-radius:2px;"></canvas>
                <div style="font-size:9px;color:#336644;margin-top:2px;">
                    Γ: <span id="sw-gamma" style="color:#ff8800;">0.0</span>
                    | Refl: <span id="sw-refl" style="color:#ff3344;">0%</span>
                </div>
                <div style="font-size:9px;color:#336644;">
                    Ticks: <span id="v-ticks" style="color:#88ccaa;">0</span>
                    | Freezes: <span id="v-freezes" style="color:#ff3344;">0</span>
                </div>
            </div>
        </div>
    </div>

<script>
// ============= Config =============
const API = '';
let ws = null;
let lastVitals = {};
let scopeData = {};

// ============= Logging =============
function log(cls, msg) {
    const el = document.getElementById('event-log');
    const t = new Date().toLocaleTimeString('en-US', {hour12:false});
    el.innerHTML = '<div class="'+cls+'">['+t+'] '+msg+'</div>' + el.innerHTML;
    if (el.children.length > 150) el.lastChild.remove();
}

// ============= API helpers =============
async function api(url, opts) {
    try {
        const r = await fetch(API + url, opts);
        return await r.json();
    } catch(e) { log('warn', 'API error: '+e.message); return null; }
}

// ============= Controls =============
async function sendStimulus(priority, intensity) {
    const signal = Array.from({length:20}, ()=> (Math.random()-0.5)*intensity*2);
    const d = await api('/api/perceive', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({signal, modality:'tactile', priority, context: priority.toLowerCase()+'_stim'})});
    if (d) {
        if (d.status === 'FROZEN') log('pain', 'BLOCKED — System FROZEN');
        else log(priority==='CRITICAL'?'warn':'info', 'Perceive #'+(d.cycle||'?')+' ('+( d.elapsed_ms||'?')+'ms)');
    }
    refresh();
}
async function stabilize() {
    await api('/api/stabilize', {method:'POST'});
    log('ok', 'Stabilized');
    refresh();
}
async function timeScale(ticks) {
    log('info', 'Time accelerating ×'+ticks+'...');
    const d = await api('/api/time-scale?multiplier='+ticks+'&ticks='+ticks, {method:'POST'});
    if (d) log('ok', 'Advanced '+d.ticks_advanced+' ticks');
    refresh();
}
async function dreamInput() {
    const text = prompt('Dream language input:', 'mama');
    if (!text) return;
    const d = await api('/api/dream-input?text='+encodeURIComponent(text), {method:'POST'});
    if (d) log('info', 'Dream: "'+d.text+'" (sleeping='+d.was_sleeping+')');
    refresh();
}
async function administerDrug() {
    const name = prompt('Drug name (L-DOPA / Valproate / Carbamazepine / SSRI):', 'SSRI');
    if (!name) return;
    const d = await api('/api/administer-drug?drug_name='+encodeURIComponent(name), {method:'POST'});
    if (d) log('ok', 'Administered: '+d.drug+' (α='+d.alpha+')');
    refresh();
}
async function sendThink() {
    const d = await api('/api/think', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question:'why does pain cause freeze?', context_vars:{pain_level:0.8}})});
    if (d) log('info', 'Think: '+(d.type||'?'));
    refresh();
}
async function sendAct() {
    const d = await api('/api/act', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({state:'stress', available_actions:['fight','flee','freeze','recover']})});
    if (d) log('info', 'Act: '+d.chosen_action);
    refresh();
}

// ============= Oscilloscope Drawing Engine =============

/** Draw CRT phosphor graticule */
function drawGraticule(ctx, W, H, pad) {
    const cW = W-pad.l-pad.r, cH = H-pad.t-pad.b;
    // Dark background
    ctx.fillStyle = '#020810';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = '#0a2a1a';
    ctx.lineWidth = 0.5;
    const divX = 10, divY = 8;
    for (let i = 0; i <= divX; i++) {
        const x = pad.l + (cW/divX)*i;
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H-pad.b); ctx.stroke();
    }
    for (let i = 0; i <= divY; i++) {
        const y = pad.t + (cH/divY)*i;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
    }

    // Center cross (brighter)
    ctx.strokeStyle = '#0f3a2a';
    ctx.lineWidth = 0.8;
    const cx = pad.l + cW/2, cy = pad.t + cH/2;
    ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(W-pad.r, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, H-pad.b); ctx.stroke();

    // Ticks on center lines
    ctx.strokeStyle = '#0f3a2a';
    ctx.lineWidth = 0.5;
    const tickLen = 3;
    for (let i = 0; i <= divX*5; i++) {
        const x = pad.l + (cW/(divX*5))*i;
        ctx.beginPath(); ctx.moveTo(x, cy-tickLen); ctx.lineTo(x, cy+tickLen); ctx.stroke();
    }
    for (let i = 0; i <= divY*5; i++) {
        const y = pad.t + (cH/(divY*5))*i;
        ctx.beginPath(); ctx.moveTo(cx-tickLen, y); ctx.lineTo(cx+tickLen, y); ctx.stroke();
    }
}

/** Draw a phosphor trace */
function drawTrace(ctx, data, W, H, pad, color, yRange, lineWidth) {
    if (!data || data.length < 2) return;
    const cW = W-pad.l-pad.r, cH = H-pad.t-pad.b;
    const n = data.length;

    // Glow (blurred wider line)
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 4;
    ctx.strokeStyle = color;
    ctx.lineWidth = (lineWidth||1.5) + 1;
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = pad.l + (i/(n-1))*cW;
        const yNorm = (data[i]-yRange[0])/(yRange[1]-yRange[0]);
        const y = pad.t + cH*(1 - Math.max(0, Math.min(1, yNorm)));
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.restore();

    // Main trace
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth || 1.5;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = pad.l + (i/(n-1))*cW;
        const yNorm = (data[i]-yRange[0])/(yRange[1]-yRange[0]);
        const y = pad.t + cH*(1 - Math.max(0, Math.min(1, yNorm)));
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;
}

/** Compute standing wave: forward + Γ × reverse */
function computeStandingWave(waveform, gamma) {
    const n = waveform.length;
    const standing = new Array(n);
    for (let i = 0; i < n; i++) {
        standing[i] = waveform[i] + gamma * waveform[n-1-i];
    }
    return standing;
}

/** Setup HiDPI canvas */
function setupCanvas(canvasId) {
    const canvas = document.getElementById(canvasId);
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, W: rect.width, H: rect.height };
}

/** Get waveform range for auto-scale */
function autoRange(data) {
    if (!data || data.length === 0) return [-1, 1];
    let mn = Infinity, mx = -Infinity;
    for (const v of data) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const margin = (mx - mn) * 0.1 || 0.5;
    return [mn - margin, mx + margin];
}

// ============= Channel Renderers =============

/** CH1: Input signal waveform */
function renderCH1(scope) {
    const {ctx, W, H} = setupCanvas('scope-ch1');
    const pad = {l:8, r:8, t:18, b:4};
    drawGraticule(ctx, W, H, pad);

    const wave = scope.input_waveform || [];
    const info = document.getElementById('ch1-info');

    if (wave.length > 0) {
        const range = autoRange(wave);
        drawTrace(ctx, wave, W, H, pad, '#33ff99', range, 1.5);
        const freq = scope.input_freq || 0;
        const band = scope.input_band || '?';
        info.textContent = freq.toFixed(1)+'Hz | '+band;
    } else {
        info.textContent = 'No signal';
        // Draw flat line at center
        const cy = pad.t + (H-pad.t-pad.b)/2;
        ctx.strokeStyle = '#1a3a2a';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(W-pad.r, cy); ctx.stroke();
    }
}

/** CH2: Coaxial channel waveforms + standing waves */
function renderCH2(scope) {
    const {ctx, W, H} = setupCanvas('scope-ch2');
    const pad = {l:8, r:8, t:18, b:14};
    drawGraticule(ctx, W, H, pad);

    const channels = scope.channels || {};
    const chNames = Object.keys(channels);
    const info = document.getElementById('ch2-info');

    if (chNames.length === 0) {
        info.textContent = 'No channels';
        return;
    }

    // Color palette for channels
    const colors = ['#33ff99','#22aaff','#ff8800','#ff55aa'];
    const labels = [];
    let allWaves = [];

    chNames.forEach((name, idx) => {
        const ch = channels[name];
        if (!ch || !ch.waveform || ch.waveform.length < 2) return;
        allWaves = allWaves.concat(ch.waveform);
    });

    const range = allWaves.length > 0 ? autoRange(allWaves) : [-1,1];

    chNames.forEach((name, idx) => {
        const ch = channels[name];
        if (!ch || !ch.waveform || ch.waveform.length < 2) return;
        const color = colors[idx % colors.length];

        // Incident waveform (dimmed)
        ctx.globalAlpha = 0.4;
        drawTrace(ctx, ch.waveform, W, H, pad, color, range, 1);
        ctx.globalAlpha = 1.0;

        // Standing wave = incident + Γ × reflected
        if (Math.abs(ch.gamma) > 0.001) {
            const standing = computeStandingWave(ch.waveform, ch.gamma);
            drawTrace(ctx, standing, W, H, pad, color, range, 2);
        } else {
            drawTrace(ctx, ch.waveform, W, H, pad, color, range, 1.5);
        }

        const shortName = name.replace('sensory','S').replace('prefrontal','PF').replace('limbic','LM').replace('motor','M');
        labels.push(shortName + ' Γ='+ch.gamma.toFixed(2));
    });

    // Labels along bottom
    ctx.font = '8px monospace';
    let lx = pad.l + 2;
    labels.forEach((lbl, idx) => {
        ctx.fillStyle = colors[idx % colors.length];
        ctx.fillText(lbl, lx, H - 3);
        lx += ctx.measureText(lbl).width + 10;
    });

    info.textContent = chNames.length + ' channels';
}

/** CH3: Perception resonance */
function renderCH3(scope) {
    const {ctx, W, H} = setupCanvas('scope-ch3');
    const pad = {l:8, r:8, t:18, b:14};
    drawGraticule(ctx, W, H, pad);

    const perc = scope.perception;
    const info = document.getElementById('ch3-info');

    if (!perc) {
        info.textContent = 'No perception';
        return;
    }

    const cW = W-pad.l-pad.r, cH = H-pad.t-pad.b;

    // Draw resonance meter - Left brain (right-tuner = holistic)
    const leftRes = perc.right_resonance || 0;  // right tuner = left visual field → holistic
    const rightRes = perc.left_resonance || 0;  // left tuner = right visual field → detail

    // Bar charts for resonance
    const barW = cW * 0.15;
    const gap = cW * 0.05;
    const cx = pad.l + cW/2;

    // Left hemisphere bar (right side of brain = holistic)
    const lBarH = cH * leftRes;
    ctx.fillStyle = 'rgba(85,170,255,0.3)';
    ctx.fillRect(cx - barW - gap, pad.t + cH - lBarH, barW, lBarH);
    ctx.strokeStyle = '#55aaff';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#55aaff';
    ctx.shadowBlur = 4;
    ctx.strokeRect(cx - barW - gap, pad.t + cH - lBarH, barW, lBarH);
    ctx.shadowBlur = 0;

    // Right hemisphere bar (left side of brain = detail)
    const rBarH = cH * rightRes;
    ctx.fillStyle = 'rgba(255,85,170,0.3)';
    ctx.fillRect(cx + gap, pad.t + cH - rBarH, barW, rBarH);
    ctx.strokeStyle = '#ff55aa';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#ff55aa';
    ctx.shadowBlur = 4;
    ctx.strokeRect(cx + gap, pad.t + cH - rBarH, barW, rBarH);
    ctx.shadowBlur = 0;

    // Labels for hemispheres
    ctx.font = '9px monospace';
    ctx.fillStyle = '#55aaff';
    ctx.fillText('R-holo', cx-barW-gap, H-3);
    ctx.fillStyle = '#ff55aa';
    ctx.fillText('L-detail', cx+gap, H-3);

    // Attention strength indicator (center arc)
    const attn = perc.attention_strength || 0;
    const arcR = Math.min(cW, cH) * 0.2;
    const arcCy = pad.t + cH * 0.4;
    ctx.beginPath();
    ctx.arc(cx, arcCy, arcR, Math.PI, Math.PI + Math.PI * attn, false);
    ctx.strokeStyle = '#33ff99';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#33ff99';
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Concept label
    const concept = perc.concept || '?';
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = '#33ff99';
    ctx.textAlign = 'center';
    ctx.fillText(concept, cx, arcCy + arcR + 14);
    ctx.textAlign = 'left';

    // Attention band
    ctx.font = '9px monospace';
    ctx.fillStyle = '#3a6a4a';
    ctx.fillText('Attn: '+(perc.attention_band||'?')+' ('+attn.toFixed(2)+')', pad.l+2, H-3);

    info.textContent = (perc.left_band||'?')+'/'+( perc.right_band||'?')+' → '+(perc.attention_band||'?');
}

/** CH4: Vital signs time series */
function renderCH4(scope) {
    const {ctx, W, H} = setupCanvas('scope-ch4');
    const pad = {l:8, r:8, t:18, b:14};
    drawGraticule(ctx, W, H, pad);

    const vitals = scope.vitals || {};
    const info = document.getElementById('ch4-info');

    // Use vitals waveform data (128 point time series)
    const hr = vitals.heart_rate || [];
    const temp = vitals.temperature || [];
    const pain = vitals.pain || [];
    const cons = vitals.consciousness || [];

    if (hr.length > 1) {
        drawTrace(ctx, hr, W, H, pad, '#33ff99', [0, 200], 1.5);
    }
    if (temp.length > 1) {
        drawTrace(ctx, temp, W, H, pad, '#ff3344', [0, 1], 1.2);
    }
    if (pain.length > 1) {
        drawTrace(ctx, pain, W, H, pad, '#ff8800', [0, 1], 1.2);
    }

    // Labels
    ctx.font = '8px monospace';
    let lx = pad.l + 2;
    const ldata = [
        {label:'HR', color:'#33ff99', data:hr},
        {label:'Temp', color:'#ff3344', data:temp},
        {label:'Pain', color:'#ff8800', data:pain},
    ];
    ldata.forEach(d => {
        const last = d.data.length > 0 ? d.data[d.data.length-1] : 0;
        const txt = d.label+':'+last.toFixed(1);
        ctx.fillStyle = d.color;
        ctx.fillText(txt, lx, H-3);
        lx += ctx.measureText(txt).width + 8;
    });

    info.textContent = hr.length+' pts';
}

/** Mini standing wave display */
function renderMiniStanding(scope) {
    const canvas = document.getElementById('mini-standing');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 240, 100);
    ctx.fillStyle = '#020810';
    ctx.fillRect(0, 0, 240, 100);

    // Pick first channel with non-zero gamma
    const channels = scope.channels || {};
    let bestCh = null, bestGamma = 0;
    for (const [name, ch] of Object.entries(channels)) {
        if (ch && ch.waveform && ch.waveform.length > 1 && Math.abs(ch.gamma) > bestGamma) {
            bestCh = ch;
            bestGamma = Math.abs(ch.gamma);
        }
    }

    const gammaEl = document.getElementById('sw-gamma');
    const reflEl = document.getElementById('sw-refl');

    if (!bestCh) {
        gammaEl.textContent = '0.0';
        reflEl.textContent = '0%';
        return;
    }

    gammaEl.textContent = bestCh.gamma.toFixed(3);
    reflEl.textContent = (bestCh.reflected_ratio * 100).toFixed(1) + '%';

    const wave = bestCh.waveform;
    const standing = computeStandingWave(wave, bestCh.gamma);
    const range = autoRange(standing);
    const n = standing.length;
    const pad = {l:2, r:2, t:2, b:2};
    const cW = 236, cH = 96;

    // Grid
    ctx.strokeStyle = '#0a2a1a';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, 50); ctx.lineTo(238, 50); ctx.stroke();

    // Standing wave
    ctx.strokeStyle = '#ff8800';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#ff8800';
    ctx.shadowBlur = 3;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = pad.l + (i/(n-1))*cW;
        const yNorm = (standing[i]-range[0])/(range[1]-range[0]);
        const y = pad.t + cH*(1 - Math.max(0, Math.min(1, yNorm)));
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
}

// ============= Refresh loop =============
async function refresh() {
    const [vitals, scopeRaw, intro] = await Promise.all([
        api('/api/vitals'),
        api('/api/oscilloscope'),
        api('/api/introspect'),
    ]);

    if (scopeRaw) scopeData = scopeRaw;

    if (vitals) {
        lastVitals = vitals;

        // Mini vitals bars
        setVitalBar('vm-temp', 'vv-temp', vitals.ram_temperature, 1);
        setVitalBar('vm-pain', 'vv-pain', vitals.pain_level, 1);
        setVitalBar('vm-stab', 'vv-stab', vitals.stability_index, 1);
        setVitalBar('vm-cons', 'vv-cons', vitals.consciousness, 1);

        document.getElementById('v-ticks').textContent = vitals.total_ticks;
        document.getElementById('v-freezes').textContent = vitals.freeze_events;

        // Heart rate
        const hrEl = document.getElementById('hr-display');
        const dotEl = document.getElementById('hb-dot');
        hrEl.textContent = vitals.heart_rate.toFixed(0)+' bpm';
        const overlay = document.getElementById('freeze-overlay');

        if (vitals.is_frozen) {
            hrEl.className = 'hr-display frozen';
            dotEl.className = 'heartbeat-indicator frozen';
            overlay.classList.add('active');
        } else if (vitals.pain_level > 0.3) {
            hrEl.className = 'hr-display danger';
            dotEl.className = 'heartbeat-indicator danger';
            overlay.classList.remove('active');
        } else {
            hrEl.className = 'hr-display';
            dotEl.className = 'heartbeat-indicator';
            overlay.classList.remove('active');
        }
    }

    // Render oscilloscope channels
    renderCH1(scopeData);
    renderCH2(scopeData);
    renderCH3(scopeData);
    renderCH4(scopeData);
    renderMiniStanding(scopeData);

    if (intro) {
        document.getElementById('sys-info').textContent =
            'Cycle '+(intro.cycle_count||0)+' | '+(intro.uptime_seconds||0).toFixed(0)+'s';
    }
}

function setVitalBar(barId, valId, value, max) {
    const pct = Math.min(100, (value/max)*100);
    document.getElementById(barId).style.width = pct+'%';
    const valEl = document.getElementById(valId);
    valEl.textContent = value.toFixed(2);
    // Color code
    if (barId==='vm-temp'||barId==='vm-pain') {
        valEl.style.color = value>0.7?'#ff3344':value>0.4?'#ff8800':'#33ff99';
    } else {
        valEl.style.color = value<0.3?'#ff3344':value<0.6?'#ff8800':'#33ff99';
    }
}

// WebSocket
function connectWS() {
    try {
        const proto = location.protocol==='https:'?'wss:':'ws:';
        ws = new WebSocket(proto+'//'+location.host+'/ws/stream');
        ws.onmessage = (e) => {
            const d = JSON.parse(e.data);
            if (d.event==='stabilize') log('ok','Stabilized');
            else if (d.event==='time_scale') log('ok','Time: +'+d.data?.ticks+' ticks');
            else if (d.event==='dream_input') log('info','Dream: "'+d.data?.text+'"');
            else if (d.event==='administer_drug') log('ok','Drug: '+d.data?.drug);
            else if (d.event) log('info','Event: '+d.event);
            refresh();
        };
        ws.onclose = () => setTimeout(connectWS, 3000);
    } catch(e) {}
}

// Init
refresh();
connectWS();
setInterval(refresh, 1000);
</script>
</body>
</html>"""
