"""Planning helper that simulates short action rollouts using a dynamics
model to choose simple, goal-directed actions.

The module loads a compact `DynamicsModel` (if available) then provides
utilities to score predicted states and simulate a 2-step action
rollout over a small discrete action space.
It is used to produce explainable, short-horizon plans for the agent.
"""

import torch
import math
import os

from dynamics_model import DynamicsModel

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "dynamics_model.pt")

# Goal specification and acceptance radius
GOAL_X = 0.5
GOAL_Y = 0.5
GOAL_THRESHOLD = 0.06

# Expected vector sizes
STATE_DIM = 36
ACTION_DIM = 2

# Instantiate the dynamics predictor and attempt to load trained weights.
model = DynamicsModel(STATE_DIM, ACTION_DIM)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Loaded trained dynamics model.")
except FileNotFoundError:
    # If no model file exists, fall back to untrained/random weights
    print("No trained model found. Using random weights.")

model.eval()

# Discrete, human-readable action set used for planning/simulation
ACTIONS = {
    "left": [-1.0, 0.0],
    "right": [1.0, 0.0],
    "up": [0.0, -1.0],
    "down": [0.0, 1.0],
}


def is_state_empty(state_vec, eps=1e-6):
    """Return True if `state_vec` is missing or effectively zero.

    This guards planners from trying to plan when no valid object is present.
    """
    return state_vec is None or len(state_vec) < 2 or sum(abs(v) for v in state_vec) < eps


def distance_to_goal(state_vec):
    """Euclidean distance from object's (x,y) to the configured goal."""
    x, y = state_vec[0], state_vec[1]
    dx = x - GOAL_X
    dy = y - GOAL_Y
    return math.sqrt(dx * dx + dy * dy)


def score_state(predicted_state):
    """Score a predicted state for planning.

    Higher scores are better; the function returns a large negative
    penalty for invalid/empty predictions and otherwise returns the
    negative distance to the goal (so closer states have higher score).
    """
    cup_x = predicted_state[0]
    cup_y = predicted_state[1]

    # Treat near-zero predictions as invalid
    if abs(cup_x) < 1e-6 and abs(cup_y) < 1e-6:
        return -1e6

    dx = cup_x - GOAL_X
    dy = cup_y - GOAL_Y
    dist = math.sqrt(dx * dx + dy * dy)
    return -dist


def predict_next_from_vector(state_vec, action):
    """Run the dynamics model once to predict next state from a vector.

    Converts Python lists to tensors, performs a no-grad forward pass,
    and returns a plain Python list as the predicted state vector.
    """
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
    action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        next_state = model(state_tensor, action_tensor)

    return next_state.squeeze(0).tolist()


def plan_action(state, action):
    """Convenience wrapper: get current state vector and predict one step.

    Returns a small dict with the input state, action, and predicted state
    for downstream code to use or display.
    """
    state_vec = state.get_state_vector()
    predicted = predict_next_from_vector(state_vec, action)

    return {
        "state_vector": state_vec,
        "action_vector": action,
        "predicted_state": predicted,
    }


def simulate_all_actions(state):
    """Simulate a 2-step rollout for every pair of discrete actions.

    The function returns the simulation results, scores for each
    sequence, the best first action, and an explanation string when
    the move is considered meaningful. Behavior when the state is
    empty or the goal is already reached is handled early.
    """
    base_state = state.get_state_vector()

    # If no valid object visible, return an explanatory message
    if is_state_empty(base_state):
        return {
            "simulations": {},
            "scores": {},
            "best_action": None,
            "best_sequence": [],
            "goal_reached": False,
            "rollout_depth": 2,
            "message": "No stable objects detected yet.",
            "explanation": "wait",
        }

    current_dist = distance_to_goal(base_state)

    # If already close to the goal, nothing to do
    if current_dist < GOAL_THRESHOLD:
        return {
            "simulations": {},
            "scores": {},
            "best_action": None,
            "best_sequence": [],
            "goal_reached": True,
            "rollout_depth": 2,
            "message": "Target reached.",
            "explanation": "stop",
        }

    results = {}
    scores = {}

    # Enumerate all 2-step action sequences and predict resulting state
    for a1_name, a1_vec in ACTIONS.items():
        s1 = predict_next_from_vector(base_state, a1_vec)

        for a2_name, a2_vec in ACTIONS.items():
            s2 = predict_next_from_vector(s1, a2_vec)
            key = f"{a1_name}->{a2_name}"
            score = score_state(s2)

            # Store the rollout details for diagnostics/UI
            results[key] = {
                "sequence": [a1_name, a2_name],
                "state_vector": base_state,
                "action_vector": a1_vec,
                "step1_state": s1,
                "step2_action_vector": a2_vec,
                "predicted_state": s2,
                "predicted_state_2": s2,
                "score": score,
            }
            scores[key] = score

    # Pick the sequence with the highest score (closest to goal)
    best_key = max(scores, key=scores.get)
    best_item = results[best_key]
    best_sequence = best_item["sequence"]
    best_first_action = best_sequence[0]

    next_dist = distance_to_goal(best_item["predicted_state"])
    progress = current_dist - next_dist

    # Thresholds to control when to produce an explanation/direction
    MIN_PROGRESS = 0.01     # ignore tiny improvements (jitter)
    STRONG_PROGRESS = 0.03  # confident move
    STAGNATION_EPS = 0.005  # basically no movement

    # Determine whether the chosen action makes meaningful progress
    if abs(progress) < STAGNATION_EPS:
        explanation = None
    elif progress < MIN_PROGRESS:
        # Small improvement — stay silent to avoid noisy feedback
        explanation = None
    elif progress > STRONG_PROGRESS:
        # Strong improvement — report the chosen direction
        explanation = best_first_action
    else:
        # Medium improvement — optionally report direction
        explanation = best_first_action

    return {
        "simulations": results,
        "scores": scores,
        "best_action": best_first_action,
        "best_sequence": best_sequence,
        "best_sequence_key": best_key,
        "goal_reached": False,
        "rollout_depth": 2,
        "message": "Simulated 2-step futures for all action pairs.",
        "explanation": explanation,
    }