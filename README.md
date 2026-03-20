# 🧠 Embodied Agent

## A lightweight world model for real-time goal-directed control in an embodied agent

<p align="left">
  <a href="https://github.com/tonyt-ai/embodied-agent"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-tonyt--ai%2Fembodied--agent-111827?logo=github"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-22c55e.svg">
  <img alt="World Model" src="https://img.shields.io/badge/Focus-World%20Models-60a5fa">
  <img alt="Embodied AI" src="https://img.shields.io/badge/Embodied-AI-34d399">
</p>

> A minimal real-time agent that **sees**, **predicts**, **plans**, and **speaks**.

---

# ✨ Abstract

We present a lightweight embodied agent that performs real-time, goal-directed behavior through continuous prediction in a learned world model. The system encodes multimodal observations into a compact state, maintains a belief over the environment, and predicts future states under candidate actions. A simple planner evaluates these imagined futures to select actions that reduce uncertainty and maximize progress toward the goal.

Unlike reactive multimodal assistants, this system explicitly models counterfactual outcomes and maintains structured beliefs over object locations and task state. This enables reasoning under partial observability, informative action selection, and grounded explanations through language.

---

# 🔁 Core loop

```text
perception → state → prediction → planning → action → speech
```

---

# 🧩 System overview

![Embodied Agent architecture](./architecture.svg)

---

# 🚀 What the system does

* streams live camera input
* detects and tracks objects (YOLO)
* builds a compact state (geometry + appearance)
* predicts future states under actions
* plans with **2-step rollouts**
* selects an action (`left`, `right`, `up`, `down`, `wait`, `stop`)
* speaks via **Gemini Live + avatar**

Result: a **real-time predictive control loop**.

---

# 🧠 Core idea

```text
f(s_t, a_t) → s_{t+1}
```

Planner:

```text
(s_t, a1) → s_{t+1}
(s_{t+1}, a2) → s_{t+2}
```

Select best first action.

---

# 🏗️ Architecture

## Perception

* YOLO → object detection
* DINOv2 → appearance embedding

## State

```python
state = [x, y, vx, vy, z]
```

## Dynamics

* MLP transition model
* trained from self-collected transitions

## Planning

* discrete action set
* 2-step rollout
* value-based selection

## Embodiment

* action → Gemini Live → avatar speech

---

# 🖥️ Demo UI

![Demo UI](./public/demo-ui.png)

---

# 📦 Repository structure

```text
embodied-agent/
│
├── app/                        # Next.js frontend (UI + avatar + camera)
│   ├── page.tsx
│   ├── layout.tsx
│   ├── globals.css
│
├── public/                     # Static assets (optional)
│
├── world_model/                # 🔥 Core backend (Python)
│   │
│   ├── server.py               # WebSocket server (YOLO + planning)
│   ├── world_state.py          # Object tracking + state vector
│   ├── planner.py              # 1-step / 2-step planning
│   ├── dynamics_model.py       # MLP transition model
│   ├── dino_encoder.py         # DINO embedding
│   ├── train_dynamics.py       # Training script
│   ├── clean_transitions.py    # Dataset filtering
│   │
│   ├── models/                 # trained + external models
│   │   ├── dynamics_model.pt
│   │   └── yolov8n.pt
│   │
│   ├── data/                   # collected dataset
│   │   ├── transitions.jsonl
│   │   └── transitions_clean.jsonl
│   │
│   └── __init__.py
│
├── server/                     # Node bridge (Gemini Live)
│   └── live-bridge.mjs
│
├── .env.local
├── .gitignore
│
├── package.json
├── package-lock.json
├── tsconfig.json
├── next.config.ts
├── postcss.config.mjs
├── eslint.config.mjs
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup

```bash
pip install -r requirements.txt
npm install
```

### Start backend

```bash
python world_model/server.py
```

### Start Gemini bridge

```bash
export GEMINI_API_KEY=your_key
node server/live-bridge.mjs
```

### Start frontend

```bash
npm run dev
```

---

# 📊 Training

```bash
python world_model/train_dynamics.py
```

---

# ⚠️ Limitations

* short planning horizon (2-step)
* single-object focus
* no long-term memory
* no learned reward
* dynamics sensitive to data

---

# 🔭 Next steps

* longer-horizon planning
* richer latent state
* multi-object reasoning
* memory
* better predictive learning

---

# 💡 Key insight

> A small system can already combine perception, prediction, planning, and language into a real-time embodied loop.

---

# 📄 License

MIT

---

# 📌 Citation / sharing

* ⭐ star the repo
* share / discuss
* open issues

