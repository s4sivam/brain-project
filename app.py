# app.py
# FINAL VERSION with RELIABLE Persistent Memory (Lifespan + Thread Event)
# With Render.com Health Check Fix + HTMLResponse

import importlib.util, threading, time, json, os, asyncio, base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, Body
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse # Import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Lock, Thread, Event
from contextlib import asynccontextmanager

# --- Global State & Config ---
NEURO_FILE = "neuron_net_final.py"
CHECKPOINT_DIR = "checkpoints"
SNAPSHOT_DIR = "snapshots"
# Use /data for persistent storage on Render (Free tier)
RENDER_DATA_DIR = "/data"
if os.path.exists(RENDER_DATA_DIR):
    CHECKPOINT_DIR = os.path.join(RENDER_DATA_DIR, "checkpoints")
    SNAPSHOT_DIR = os.path.join(RENDER_DATA_DIR, "snapshots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "default.json")
clients = set()
state_lock = Lock()
stop_event = Event() 

brain_runtime = {"module": None, "net": None, "thread": None} 
latest_state = {
    "tick": 0, "spikes": [], "weights_hist": {}, "labels": [], 
    "params": {}, "twin_text": "Initializing...", "twin_sentiment": 0.0
}
shared_control = {
    "A_PLUS": None, "A_MINUS": None, "NO_REWARD_SCALE": None, 
    "OUTGOING_SUM_LIMIT": None, "REWARD_WINDOW": None, "cmd": None
}

# --- Brain Runner Thread ---
def runner_loop():
    try:
        spec = importlib.util.spec_from_file_location("brain_module", NEURO_FILE)
        brain = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(brain)
    except Exception as e:
        print(f"CRITICAL: Could not load {NEURO_FILE}: {e}")
        return

    net = brain.Network(brain.N_INPUT, brain.N_HIDDEN, brain.N_OUTPUT)
    
    brain_runtime["module"] = brain
    brain_runtime["net"] = net
    
    try: 
        if os.path.exists(DEFAULT_CHECKPOINT):
            brain.load_checkpoint(net, DEFAULT_CHECKPOINT)
            print(f"✅ Brain memory loaded from {DEFAULT_CHECKPOINT}")
        else:
            print(f"No default checkpoint found at {DEFAULT_CHECKPOINT}. Starting fresh.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    spike_hist = []
    
    while not stop_event.is_set():
        try:
            for param, val in shared_control.items():
                if val is not None and hasattr(brain, param):
                    if getattr(brain, param) != float(val):
                         setattr(brain, param, float(val))

            cmd = shared_control.get("cmd")
            if cmd:
                try:
                    if cmd == "reset":
                        print("[CMD] Resetting brain network...")
                        net = brain.Network(brain.N_INPUT, brain.N_HIDDEN, brain.N_OUTPUT)
                        brain_runtime["net"] = net 
                        if os.path.exists(DEFAULT_CHECKPOINT):
                            os.remove(DEFAULT_CHECKPOINT)
                except Exception as e: print(f"Cmd Error: {e}")
                finally: shared_control["cmd"] = None

            fired, reward = net.tick()
            
            spike_hist.append(list(fired))
            if len(spike_hist) > 400: spike_hist.pop(0)
            
            if net.time % 5 == 0:
                h_o = [s.w for s in net.synapses if s.pre in net.hidden and s.post in net.outputs]
                hist = {}
                for w in h_o:
                    b = f"{round(w,2):.2f}"
                    hist[b] = hist.get(b, 0) + 1
                
                with state_lock:
                    latest_state["tick"] = net.time
                    latest_state["spikes"] = spike_hist.copy()
                    latest_state["weights_hist"] = hist
                    latest_state["labels"] = [f"I{i}" for i in range(brain.N_INPUT)] + [f"H{i}" for i in range(brain.N_HIDDEN)] + [f"O{i}" for i in range(brain.N_OUTPUT)]
                    latest_state["params"] = {
                        "A_PLUS": getattr(brain, "A_PLUS", 0),
                        "NO_REWARD_SCALE": getattr(brain, "NO_REWARD_SCALE", 0)
                    }
            
            time.sleep(0.02)
        
        except Exception as e:
            print(f"Error in runner_loop: {e}")
            time.sleep(1) 

    print("Brain thread received stop signal and is exiting gracefully.")

# --- LIFESPAN EVENT HANDLER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("Starting brain simulation thread...")
    brain_thread = Thread(target=runner_loop, daemon=False) 
    brain_runtime["thread"] = brain_thread
    brain_thread.start()
    
    yield
    
    # --- SHUTDOWN LOGIC ---
    print("\nShutdown signal received. Stopping brain thread...")
    stop_event.set() 
    brain_thread.join(timeout=5.0) 

    print(f"Saving brain state to {DEFAULT_CHECKPOINT}...")
    brain = brain_runtime.get("module")
    net = brain_runtime.get("net")
    
    if brain and net and hasattr(brain, "save_checkpoint"):
        try:
            brain.save_checkpoint(net.synapses, fname=DEFAULT_CHECKPOINT)
            print(f"✅ Brain memory saved.")
        except Exception as e:
            print(f"Error saving checkpoint on shutdown: {e}")
    else:
        print("Could not find brain runtime to save.")

# --- Initialize FastAPI App ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- API Endpoints ---
@app.api_route("/", methods=["GET", "HEAD"])  # <-- FIX 1
async def index():
    html_path = "monitor_final.html"
    if not os.path.exists(html_path): # <-- FIX 2 (Better Errors)
        print(f"CRITICAL ERROR: {html_path} not found!")
        return JSONResponse(
            status_code=404, 
            content={"error": f"{html_path} not found on server. Make sure it is in your GitHub repo."}
        )
    with open(html_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("WebSocket connection established.")
    clients.add(ws)
    try:
        while True:
            await asyncio.sleep(0.1)
            with state_lock:
                await ws.send_text(json.dumps({"type":"snapshot", "state": latest_state}))
            
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                payload = json.loads(msg)
                if payload.get("type") == "set":
                    shared_control.update(payload.get("params", {}))
                elif payload.get("type") == "cmd":
                    shared_control["cmd"] = payload.get("cmd")
            except: pass
    except: 
        print("WebSocket connection closed.")
    finally: clients.remove(ws)

@app.post("/twin")
async def post_twin_thought(payload: dict = Body(...)):
    with state_lock:
        latest_state["twin_text"] = payload.get("text", "")
        latest_state["twin_sentiment"] = payload.get("sentiment", 0.0)
    return {"status": "ok"}

@app.post("/set_params")
async def inject_params(payload: dict = Body(...)):
    print(f"[INJECTION] Twin modifying params: {payload}")
    shared_control.update(payload)
    return {"status": "ok"}

@app.post("/save_snapshot")
async def save_snapshot(name: str = Form(...), data_url: str = Form(...)):
    try:
        data = base64.b64decode(data_url.split(",",1)[1])
        with open(os.path.join(SNAPSHOT_DIR, f"{name}.png"), "wb") as f: f.write(data)
        return {"ok": True}
    except Exception as e: return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    print("Starting Brain 2.0 Server with Persistent Memory...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
