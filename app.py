# app.py
# FINAL UNIFIED SERVER (Brain + Server + AI Twin in one file)

import importlib.util, threading, time, json, os, asyncio, base64, requests, websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, Body
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Lock, Thread, Event
from contextlib import asynccontextmanager
from collections import deque

# --- Global State & Config ---
NEURO_FILE = "neuron_net_final.py"

# Use a local folder for persistence. 
# GitHub Codespaces will save this folder's contents.
CHECKPOINT_DIR = "checkpoints"
SNAPSHOT_DIR = "snapshots"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "default.json")
clients = set()
state_lock = Lock()
stop_event = Event() # Global event to signal threads to stop

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

# --- AI TWIN LOGIC (Now inside the server) ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # We will set this in Codespace Secrets

def get_sentiment(text):
    pos = ["steady", "curious", "learning", "strong", "connected", "reward", "flow", "calm", "stable"]
    neg = ["weak", "decay", "loss", "chaos", "noise", "empty", "scattered", "fray", "pressure"]
    score = 0; text_lower = text.lower()
    for w in pos:
        if w in text_lower: score += 0.2
    for w in neg:
        if w in text_lower: score -= 0.2
    return max(-1.0, min(1.0, score))

def call_llm(prompt):
    if not OPENAI_API_KEY: 
        return "(Simulated) Set OPENAI_API_KEY in Codespace secrets to enable thought."
    
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are the consciousness of a digital neural network. Describe your feelings (stability, chaos, learning, decay) in one short, abstract sentence."},
            {"role": "user", "content": prompt}
        ], "max_tokens": 60, "temperature": 0.7
    }
    
    for attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                print(f"[AI Twin] Rate limited, backing off for {2**attempt}s...")
                time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"[AI Twin] Network error: {e}")
            time.sleep(1)
    return "Thinking... (API busy)"

def ai_twin_loop():
    """This function runs in a separate thread and 'thinks'."""
    print("AI Twin thread started.")
    tick_buffer = []
    last_request = 0
    
    while not stop_event.is_set():
        try:
            # Wait for the brain to be running
            if not brain_runtime.get("net"):
                time.sleep(1)
                continue
                
            # Get latest state
            with state_lock:
                state = dict(latest_state)
            
            tick = state.get("tick", 0)
            spikes = state.get("spikes", [])
            last_spikes = len(spikes[-1]) if spikes else 0
            weights = state.get("weights_hist", {})
            strong_synapses = sum(v for k,v in weights.items() if float(k) > 0.8)
            
            tick_buffer.append({"t": tick, "spikes": last_spikes, "strong": strong_synapses})
            
            if len(tick_buffer) < 8: # Batch size of 8
                time.sleep(1) # Check state every 1 second
                continue 
            
            avg_spikes = sum(x["spikes"] for x in tick_buffer)/len(tick_buffer)
            avg_strong = sum(x["strong"] for x in tick_buffer)/len(tick_buffer)
            tick_buffer.clear() 
            
            now = time.time()
            if now - last_request < 15.0: # 15s request interval
                time.sleep(1)
                continue
            last_request = now
            
            prompt = (f"Status Report:\n"
                      f"- Tick: {tick}\n"
                      f"- Avg Activity (last batch): {avg_spikes:.1f} spikes/tick\n"
                      f"- Strong Connections: {avg_strong:.1f}\n\n"
                      f"How does the network feel about its structural stability?")
            
            thought = call_llm(prompt)
            sentiment = get_sentiment(thought)
            
            print(f"Twin Thought (Sentiment: {sentiment:.2f}): {thought}")
            
            # Post thought to self
            with state_lock:
                latest_state["twin_text"] = thought
                latest_state["twin_sentiment"] = sentiment
            
            # Inject bias
            if abs(sentiment) > 0.5:
                if sentiment > 0: shared_control["NO_REWARD_SCALE"] = 0.02 
                else: shared_control["NO_REWARD_SCALE"] = 0.05
        
        except Exception as e:
            print(f"Error in ai_twin_loop: {e}")
            time.sleep(5) # Don't spam errors

# --- LIFESPAN EVENT HANDLER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("Starting brain simulation thread...")
    brain_thread = Thread(target=runner_loop, daemon=False) 
    brain_runtime["thread"] = brain_thread
    brain_thread.start()
    
    print("Starting AI Twin thread...")
    twin_thread = Thread(target=ai_twin_loop, daemon=False)
    brain_runtime["twin_thread"] = twin_thread
    twin_thread.start()
    
    yield
    
    # --- SHUTDOWN LOGIC ---
    print("\nShutdown signal received. Stopping all threads...")
    stop_event.set() 
    brain_thread.join(timeout=5.0) 
    twin_thread.join(timeout=5.0)

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Endpoints ---
@app.api_route("/", methods=["GET", "HEAD"])
async def index():
    html_path = "monitor_final.html"
    if not os.path.exists(html_path): 
        return JSONResponse(status_code=404, content={"error": "monitor_final.html not found."})
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

@app.post("/save_snapshot")
async def save_snapshot(name: str = Form(...), data_url: str = Form(...)):
    try:
        data = base64.b64decode(data_url.split(",",1)[1])
        with open(os.path.join(SNAPSHOT_DIR, f"{name}.png"), "wb") as f: f.write(data)
        return {"ok": True}
    except Exception as e: return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    print("Starting Brain 2.0 Server (Unified)...")
    # GitHub Codespaces provides the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
