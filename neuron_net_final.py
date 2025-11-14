# neuron_net_final.py
# The Core SNN Engine (Optimized V7 Super Bootstrap)
print("LOADING NEURON_NET_FINAL... (Global functions check)") 

import random, math, json, os
from collections import deque

# --- Configuration ---
TICKS = 6000
SEED = 42
random.seed(SEED)

N_INPUT = 3
N_HIDDEN = 8
N_OUTPUT = 1

THETA = 1.0
LEAK = 0.03
REFRACT = 2

# Aggressive initial weights for faster bootstrapping
INIT_W = (0.18, 0.9)    
DELAY_RANGE = (1, 2)
INPUT_PULSE = 3.0

PATTERN_PERIOD = 40
PATTERN_BEATS = [0, 5, 10]

# STDP & Learning
TAU_STDP = 30.0
A_PLUS = 0.035
A_MINUS = 0.04
STDP_WINDOW = 60
W_MIN = 0.01
W_MAX = 2.0

# Reward & Plasticity
REWARD_GAIN = 1.0
NO_REWARD_SCALE = 0.22
REWARD_WINDOW = 14

DECAY_INTERVAL = 200
DECAY_FACTOR = 0.995

# Structural Plasticity
PRUNE_INTERVAL = 400
PRUNE_THRESH = 0.02
PRUNE_PROB = 0.4
SPROUT_K = 14
CORR_WINDOW = 400

NORMALIZATION_MODE = "clamp"
OUTGOING_SUM_LIMIT = 2.4

# --- Core Classes ---
class Event:
    __slots__ = ("tick", "amp")
    def __init__(self, tick, amp):
        self.tick = tick
        self.amp = amp

class Neuron:
    def __init__(self, idx):
        self.idx = idx
        self.v = 0.0
        self.theta = THETA
        self.refractory = 0
        self.inbox = deque()
        self.outgoing = []
        self.last_spike = -10**9
        self.spike_times = deque()

    def receive(self, ev: Event):
        self.inbox.append(ev)

    def step(self, t, spikes_out):
        while self.inbox and self.inbox[0].tick <= t:
            ev = self.inbox.popleft()
            self.v += ev.amp
        
        if self.v > 0.0:
            self.v -= LEAK * self.v
            
        if self.refractory > 0:
            self.refractory -= 1
            return False
            
        if self.v >= self.theta:
            self.v = 0.0
            self.refractory = REFRACT
            self.last_spike = t
            self.spike_times.append(t)
            # Cleanup old spike history for correlation
            while self.spike_times and (t - self.spike_times[0] > CORR_WINDOW):
                self.spike_times.popleft()
            spikes_out.append(self.idx)
            for s in self.outgoing:
                s.propagate(t)
            return True
        return False

class Synapse:
    def __init__(self, pre, post, w=None, delay=None):
        self.pre = pre
        self.post = post
        self.w = w if w is not None else random.uniform(*INIT_W)
        self.delay = delay if delay is not None else random.randint(*DELAY_RANGE)
        self.last_pre = -10**9
        self.last_post = -10**9

    def propagate(self, t_now):
        self.post.receive(Event(t_now + self.delay, self.w))
        self.last_pre = t_now

    def on_post_spike(self, t_now):
        self.last_post = t_now

    def apply_stdp(self, reward_scale=1.0):
        dt = self.last_post - self.last_pre
        if self.last_pre < -1e8 or self.last_post < -1e8: return
        if abs(dt) > STDP_WINDOW: return
        
        if dt > 0: # Potentiation
            dw = A_PLUS * math.exp(-dt / TAU_STDP) * reward_scale
            self.w = min(W_MAX, self.w + dw)
        elif dt < 0: # Depression
            dw = A_MINUS * math.exp(dt / TAU_STDP) * reward_scale
            self.w = max(W_MIN, self.w - dw)

class Network:
    def __init__(self, n_in, n_hidden, n_out):
        self.time = 0
        self.neurons = [Neuron(i) for i in range(n_in + n_hidden + n_out)]
        self.inputs = self.neurons[:n_in]
        self.hidden = self.neurons[n_in:n_in+n_hidden]
        self.outputs = self.neurons[n_in+n_hidden:]
        self.synapses = []
        self.expected_windows = []
        self.pattern_windows_info = [] # For debugging
        self.wire_default()

    def wire_default(self):
        def connect(src_layer, dst_layer, prob):
            for u in src_layer:
                for v in dst_layer:
                    if u is v: continue
                    if random.random() < prob:
                        s = Synapse(u, v)
                        u.outgoing.append(s)
                        self.synapses.append(s)
        # Aggressive connectivity
        connect(self.inputs, self.hidden, 0.98)
        connect(self.hidden, self.hidden, 0.2)
        connect(self.hidden, self.outputs, 0.9)
        connect(self.outputs, self.hidden, 0.05)

    def stimulate_pattern(self, t):
        base = t % PATTERN_PERIOD
        if base in PATTERN_BEATS:
            idx = PATTERN_BEATS.index(base)
            if idx < len(self.inputs):
                self.inputs[idx].receive(Event(t, INPUT_PULSE))
            self.expected_windows.append((t, t + REWARD_WINDOW))

    def tick(self):
        t = self.time
        self.stimulate_pattern(t)
        
        fired = []
        for n in self.neurons:
            n.step(t, fired)
            
        if fired:
            fired_set = set(fired)
            for s in self.synapses:
                if s.post.idx in fired_set:
                    s.on_post_spike(t)

        # Check Rewards
        reward = 0.0
        matched = False
        out_fired = any(n.idx in fired for n in self.outputs)
        
        # Process windows
        active_windows = []
        for (start, end) in self.expected_windows:
            if t < start: active_windows.append((start, end))
            elif start <= t <= end:
                if out_fired: matched = True
                else: active_windows.append((start, end))
        self.expected_windows = active_windows

        if matched: reward = REWARD_GAIN
        
        # Apply STDP
        gain = reward if reward > 0 else NO_REWARD_SCALE
        for s in self.synapses:
            s.apply_stdp(reward_scale=gain)
            
        # Normalization & Homeostasis
        self.normalize_outgoing()
        if t > 0 and t % DECAY_INTERVAL == 0:
            for s in self.synapses:
                s.w = max(W_MIN, min(W_MAX, s.w * DECAY_FACTOR))
                
        self.time += 1
        return fired, reward

    def normalize_outgoing(self):
        if NORMALIZATION_MODE == "clamp":
            for n in self.neurons:
                outs = n.outgoing
                if not outs: continue
                ssum = sum(max(W_MIN, s.w) for s in outs)
                if ssum > OUTGOING_SUM_LIMIT:
                    factor = OUTGOING_SUM_LIMIT / ssum
                    for s in outs: s.w = max(W_MIN, s.w * factor)

# --- Persistence Methods ---
# These are strictly OUTSIDE the class Network
def save_checkpoint(synapses, fname):
    data = [{"pre":s.pre.idx, "post":s.post.idx, "w":s.w, "delay":s.delay} for s in synapses]
    with open(fname, "w") as f: json.dump(data, f)

def load_checkpoint(net, fname):
    if not os.path.exists(fname): return
    with open(fname, "r") as f: data = json.load(f)
    key_map = {(s.pre.idx, s.post.idx): s for s in net.synapses}
    for item in data:
        k = (item["pre"], item["post"])
        if k in key_map:
            key_map[k].w = item["w"]
            key_map[k].delay = item.get("delay", key_map[k].delay)