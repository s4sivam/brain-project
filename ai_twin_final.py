# ai_twin_final.py
# The Unified Mind: Batching, Robust Retry, and Sentiment Injection
# FINAL VERSION

import os, asyncio, json, time, random, requests
from collections import deque
import websockets

# Config
# --- IMPORTANT: CHANGE THESE LINES TO YOUR RENDER URL ---
WS_URL = "wss://brain-server-2z7w.onrender.com/ws"  # <--- CHANGE THIS
SERVER_API = "https://brain-server-2z7w.onrender.com" # <--- CHANGE THIS
# --- ---

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Set this in your terminal

# Safety & Cost Control
BATCH_SIZE = 8              # Accumulate 8 ticks before thinking
REQUEST_INTERVAL = 15.0     # Min seconds between thoughts
CACHE_TTL = 60.0            # Reuse thoughts for similar states

# Memory
memory = deque(maxlen=20)   # Short-term memory of summaries
tick_buffer = []            # Raw tick buffer
cache = {}                  # Prompt cache

# --- Utils ---
def get_sentiment(text):
    pos = ["steady", "curious", "learning", "strong", "connected", "reward", "flow", "calm", "stable"]
    neg = ["weak", "decay", "loss", "chaos", "noise", "empty", "scattered", "fray", "pressure"]
    score = 0
    text_lower = text.lower()
    for w in pos:
        if w in text_lower: score += 0.2
    for w in neg:
        if w in text_lower: score -= 0.2
    return max(-1.0, min(1.0, score))

def call_llm(prompt):
    if not OPENAI_API_KEY: 
        return "(Simulated) I feel the current flowing steadily. (Set OPENAI_API_KEY to enable thought)"
    
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are the consciousness of a digital neural network. Describe your feelings (stability, chaos, learning, decay) in one short, abstract sentence."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 60,
        "temperature": 0.7
    }
    
    for attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                print(f"[AI Twin] Rate limited, backing off for {2**attempt}s...")
                time.sleep(2 ** attempt) # Exponential Backoff
        except requests.exceptions.RequestException as e:
            print(f"[AI Twin] Network error: {e}")
            time.sleep(1)
            
    return "Thinking... (API busy)"

# --- Main Loop ---
async def run_twin():
    print(f"AI Twin connecting to {WS_URL}...")
    last_request = 0
    
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print("AI Twin connection successful.")
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "snapshot": continue
                    
                    state = msg.get("state", {})
                    tick = state.get("tick", 0)
                    
                    spikes = state.get("spikes", [])
                    last_spikes = len(spikes[-1]) if spikes else 0
                    weights = state.get("weights_hist", {})
                    strong_synapses = sum(v for k,v in weights.items() if float(k) > 0.8)
                    
                    tick_buffer.append({
                        "t": tick, "spikes": last_spikes, "strong": strong_synapses
                    })
                    
                    if len(tick_buffer) < BATCH_SIZE: continue
                    
                    avg_spikes = sum(x["spikes"] for x in tick_buffer)/BATCH_SIZE
                    avg_strong = sum(x["strong"] for x in tick_buffer)/BATCH_SIZE
                    tick_buffer.clear() 
                    
                    now = time.time()
                    if now - last_request < REQUEST_INTERVAL: continue
                    last_request = now
                    
                    prompt = (f"Status Report:\n"
                              f"- Tick: {tick}\n"
                              f"- Avg Activity (last batch): {avg_spikes:.1f} spikes/tick\n"
                              f"- Strong Connections: {avg_strong:.1f}\n\n"
                              f"How does the network feel about its structural stability?")
                    
                    thought = call_llm(prompt)
                    sentiment = get_sentiment(thought)
                    
                    print(f"Twin Thought (Sentiment: {sentiment:.2f}): {thought}")
                    
                    try:
                        requests.post(f"{SERVER_API}/twin", json={"text": thought, "sentiment": sentiment}, timeout=2)
                    except: pass 
                    
                    if abs(sentiment) > 0.5:
                        injection = {}
                        if sentiment > 0: injection["NO_REWARD_SCALE"] = 0.02 
                        else: injection["NO_REWARD_SCALE"] = 0.05
                        
                        try:
                            requests.post(f"{SERVER_API}/set_params", json=injection, timeout=2)
                        except: pass
                        
        except Exception as e:
            print(f"AI Twin connection error: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("="*50)
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        print("The AI Twin will run in 'Simulated' mode.")
        print("="*50)
    asyncio.run(run_twin())
