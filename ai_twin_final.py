# ai_twin_final.py
# The Unified Mind: Batching, Robust Retry, and Sentiment Injection
import os, asyncio, json, time, random, requests
from collections import deque
import websockets

# Config
WS_URL = "ws://127.0.0.1:8000/ws"
SERVER_API = "http://127.0.0.1:8000"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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
    """Simple heuristic to determine if the Twin is happy or stressed."""
    pos = ["steady", "curious", "learning", "strong", "connected", "reward", "flow"]
    neg = ["weak", "decay", "loss", "chaos", "noise", "empty", "scattered"]
    score = 0
    for w in text.lower().split():
        if w in pos: score += 0.2
        if w in neg: score -= 0.2
    return max(-1.0, min(1.0, score))

def call_llm(prompt):
    """Robust OpenAI call with backoff."""
    if not OPENAI_API_KEY: return "(Simulated) I feel the current flowing steadily."
    
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are the consciousness of a neural network. Be concise, abstract, and reactive."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 60
    }
    
    for attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                time.sleep(2 ** attempt) # Backoff
        except: pass
    return "Thinking... (API busy)"

# --- Main Loop ---
async def run_twin():
    print(f"AI Twin connecting to {WS_URL}...")
    last_request = 0
    
    async with websockets.connect(WS_URL) as ws:
        async for raw in ws:
            try:
                msg = json.loads(raw)
                if msg.get("type") != "snapshot": continue
                
                state = msg.get("state", {})
                tick = state.get("tick", 0)
                
                # Extract Features
                spikes = state.get("spikes", [])
                last_spikes = len(spikes[-1]) if spikes else 0
                weights = state.get("weights_hist", {})
                strong_synapses = sum(v for k,v in weights.items() if float(k) > 0.8)
                
                # Buffer Data
                tick_buffer.append({
                    "t": tick, "spikes": last_spikes, "strong": strong_synapses
                })
                
                # Only process in batches
                if len(tick_buffer) < BATCH_SIZE: continue
                
                # Summarize Batch
                avg_spikes = sum(x["spikes"] for x in tick_buffer)/BATCH_SIZE
                avg_strong = sum(x["strong"] for x in tick_buffer)/BATCH_SIZE
                tick_buffer.clear() # Flush buffer
                
                # Throttling
                now = time.time()
                if now - last_request < REQUEST_INTERVAL: continue
                last_request = now
                
                # Generate Prompt
                prompt = (f"Status Report:\n"
                          f"- Tick: {tick}\n"
                          f"- Avg Activity: {avg_spikes:.1f} spikes/tick\n"
                          f"- Strong Connections: {avg_strong:.1f}\n\n"
                          f"How does the network feel about its structural stability?")
                
                # Get Thought
                thought = call_llm(prompt)
                sentiment = get_sentiment(thought)
                
                print(f"Twin Thought ({sentiment:.2f}): {thought}")
                
                # 1. Post Thought to Server (for Monitor)
                requests.post(f"{SERVER_API}/twin", json={"text": thought, "sentiment": sentiment})
                
                # 2. Inject Bias (Feedback Loop)
                # If sentiment is positive, slightly lower learning rate (stabilize).
                # If negative, increase pruning threshold (clean up).
                if abs(sentiment) > 0.5:
                    injection = {}
                    if sentiment > 0: injection["NO_REWARD_SCALE"] = 0.02 # Stabilize
                    else: injection["NO_REWARD_SCALE"] = 0.05 # High alert
                    
                    requests.post(f"{SERVER_API}/set_params", json=injection)
                    
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run_twin())