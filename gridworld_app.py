import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# configuration & constant variables
st.set_page_config(page_title="Gridworld Q-Learning Sandbox", layout="wide")

ROWS, COLS = 6, 6  # fixed grid size maybe in future could add adjustable slider
N_ACTIONS = 4
ACTIONS = ["↑", "→", "↓", "←"]
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Cell types
EMPTY, WALL, START, GOAL, PIT, REWARD = 0, 1, 2, 3, 4, 5
CELL_LABELS = {
    EMPTY: "⬜",
    WALL: "⬛",
    START: "🟩",
    GOAL: "⭐",
    PIT: "🟥",
    REWARD: "🟦",
}
CELL_COLORS = {
    EMPTY: "white",
    WALL: "black",
    START: "green",
    GOAL: "gold",
    PIT: "red",
    REWARD: "blue",
}

# Rewards
STEP_REWARD = -0.04
GOAL_REWARD = 1.0
PIT_REWARD = -1.0
REWARD_BONUS = 0.5   # extra reward when stepping on a reward tile maybe in future could make adjustable

# helper functions
def create_default_grid():
    grid = np.zeros((ROWS, COLS), dtype=int)
    grid[:, :] = EMPTY
    grid[ROWS - 1, 0] = START     # bottom-left
    grid[0, COLS - 1] = GOAL      # top-right
    grid[2, 2] = PIT
    grid[3, 3] = WALL
    return grid

def get_reward_positions(grid):
    """Return list of (r, c) that are reward tiles and mapping to bit indices."""
    positions = [(r, c) for r in range(ROWS) for c in range(COLS) if grid[r, c] == REWARD]
    index = {pos: i for i, pos in enumerate(positions)}
    return positions, index


def get_start_pos(grid):
    starts = np.argwhere(grid == START)
    if len(starts) == 0:
        return (ROWS - 1, 0)
    r, c = starts[0]
    return (int(r), int(c))


def ensure_single_start_goal(grid, r, c, new_type):
    """If setting a cell to START or GOAL, make sure it's unique."""
    if new_type == START:
        grid[grid == START] = EMPTY
    elif new_type == GOAL:
        grid[grid == GOAL] = EMPTY
    grid[r, c] = new_type
    return grid


def step_env(state, action, grid, reward_index):
    """
    One step in the gridworld

    State = (row, col, mask) where mask bit i indicates that reward tile i
    (from reward_index) has been collected in this episode (1 = collected)
    """
    r, c, mask = state
    dr, dc = DELTAS[action]
    nr, nc = r + dr, c + dc

    # stay if hit wall or boundary
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or grid[nr, nc] == WALL:
        nr, nc = r, c

    cell = grid[nr, nc]
    reward = STEP_REWARD
    done = False
    new_mask = mask

    if cell == GOAL:
        reward = GOAL_REWARD
        done = True
    elif cell == PIT:
        reward = PIT_REWARD
        done = True
    elif cell == REWARD:
        idx = reward_index.get((nr, nc))
        if idx is not None:
            if (mask & (1 << idx)) == 0:
                reward += REWARD_BONUS
                new_mask = mask | (1 << idx)

    next_state = (nr, nc, new_mask)
    return next_state, reward, done


def epsilon_greedy_action(state, Q, epsilon):
    r, c, mask = state
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    qs = [Q.get((r, c, mask, a), 0.0) for a in range(N_ACTIONS)]
    return int(np.argmax(qs))


def greedy_action(state, Q):
    return epsilon_greedy_action(state, Q, 0.0)

def compute_value_grid(grid, Q):
    """
    Compute V(r, c) = max_{mask, a} Q((r, c, mask), a)

    This shows, for each cell, the best value the agent has learned for any mask state
    """
    value_grid = np.full((ROWS, COLS), np.nan)

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r, c] == WALL:
                continue

            # Collect all Q-values for this (r, c), any mask, any action
            qs = [
                q
                for (rr, cc, m, a), q in Q.items()
                if rr == r and cc == c
            ]

            if qs:
                value_grid[r, c] = max(qs)
            else:
                value_grid[r, c] = 0.0 

    return value_grid




def make_value_heatmap_figure(grid, Q, title=None):
    """Base value heatmap figure."""
    value_grid = compute_value_grid(grid, Q)

    fig = go.Figure(
        data=go.Heatmap(
            z=value_grid,
            colorscale="Viridis",
            colorbar=dict(title="Value"),
            zmin=np.nanmin(value_grid) if np.any(~np.isnan(value_grid)) else 0.0,
            zmax=np.nanmax(value_grid) if np.any(~np.isnan(value_grid)) else 1.0,
        )
    )

    annotations = []
    for r in range(ROWS):
        for c in range(COLS):
            cell = grid[r, c]
            if cell != WALL:
                text = ""
                if cell == START:
                    text = "S"
                elif cell == GOAL:
                    text = "G"
                elif cell == PIT:
                    text = "P"
                elif cell == REWARD:
                    text = "R"
                if text:
                    annotations.append(
                        dict(
                            x=c,
                            y=r,
                            text=text,
                            showarrow=False,
                            font=dict(color="white", size=14),
                        )
                    )

    fig.update_layout(
        xaxis=dict(title="Col", dtick=1, range=(-0.5, COLS - 0.5)),
        yaxis=dict(title="Row", dtick=1, range=(ROWS - 0.5, -0.5)),
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        annotations=annotations,
        title=title,
    )

    return fig


def make_value_heatmap_with_path(grid, Q, path, title=None):
    """Value heatmap + path + current agent position."""
    fig = make_value_heatmap_figure(grid, Q, title=title)

    if len(path) == 0:
        return fig

    xs = [c for (r, c, mask) in path]
    ys = [r for (r, c, mask) in path]

    if len(xs) > 1:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name="Path",
                line=dict(width=2),
                marker=dict(size=6),
            )
        )

    ax, ay = xs[-1], ys[-1]
    fig.add_trace(
        go.Scatter(
            x=[ax],
            y=[ay],
            mode="markers",
            name="Agent",
            marker=dict(size=14, color="red", symbol="circle"),
        )
    )

    return fig


# Session State Initialization
if "grid" not in st.session_state:
    st.session_state.grid = create_default_grid()

grid = st.session_state.grid

# figure out reward tiles & reset Q if the count changed (MDP changed)
reward_positions, reward_index = get_reward_positions(grid)
new_num_rewards = len(reward_positions)
if "num_rewards" not in st.session_state:
    st.session_state.num_rewards = new_num_rewards
elif st.session_state.num_rewards != new_num_rewards:
    # reset Q and history (new MDP)
    st.session_state.Q = {}
    st.session_state.episode_rewards = []
    st.session_state.num_rewards = new_num_rewards

if "Q" not in st.session_state:
    st.session_state.Q = {}  # dictionary: (r,c,mask,a) -> Q-value

if "episode_rewards" not in st.session_state:
    st.session_state.episode_rewards = []

Q = st.session_state.Q 

# Sidebar Controls
st.sidebar.title("Q-Learning Settings")

alpha = st.sidebar.slider("Learning rate (α)", 0.01, 1.0, 0.3, 0.01)
gamma = st.sidebar.slider("Discount factor (γ)", 0.5, 0.99, 0.95, 0.01)
epsilon = st.sidebar.slider("Exploration (ε)", 0.0, 1.0, 0.2, 0.01)
episodes_to_run = st.sidebar.slider("Episodes per training run / timelapse", 1, 1000, 100, 5)
max_steps = st.sidebar.slider("Max steps per episode", 10, 500, 100, 10)
frame_delay = st.sidebar.slider("Animation speed (s per frame)", 0.05, 1.0, 0.2, 0.05)

if st.sidebar.button("Reset Q-table & history"):
    st.session_state.Q = {}
    st.session_state.episode_rewards = []
    st.sidebar.success("Reset complete!")

# Main Layout
st.title("Interactive Gridworld Q-Learning Sandbox")
st.markdown(
    "Draw an environment on the left, then train a **model-free Q-learning** agent. "
    "Reward tiles 🟦 give a bonus **only the first time** they are visited in an episode, "
    "and this is encoded in the Markov state via a bitmask of collected rewards."
)

left_col, right_col = st.columns([1, 1.2])
grid = st.session_state.grid
Q = st.session_state.Q

# Left: Grid Editor
with left_col:
    st.subheader("Environment Editor")

    brush = st.radio(
        "Brush",
        ["Empty ⬜", "Wall ⬛", "Start 🟩", "Goal ⭐", "Pit 🟥", "Reward 🟦"],
        horizontal=True,
    )

    brush_map = {
        "Empty ⬜": EMPTY,
        "Wall ⬛": WALL,
        "Start 🟩": START,
        "Goal ⭐": GOAL,
        "Pit 🟥": PIT,
        "Reward 🟦": REWARD,
    }
    current_type = brush_map[brush]

    for r in range(ROWS):
        cols = st.columns(COLS)
        for c in range(COLS):
            cell_type = grid[r, c]
            label = CELL_LABELS[cell_type]
            key = f"cell-{r}-{c}"
            if cols[c].button(label, key=key):
                grid = ensure_single_start_goal(grid, r, c, current_type)
                st.session_state.grid = grid
                # environment changed -> recompute rewards & reset Q next run if needed
                st.rerun()

    st.caption(
        "Click cells to paint them with the selected brush. "
        "There should be exactly one 🟩 Start and (ideally) one ⭐ Goal. "
        "Reward tiles 🟦 give a +0.5 bonus the **first time** you step on them each episode."
    )

# Q-learning Training Functions
def run_q_learning(episodes, alpha, gamma, epsilon, max_steps, reward_index):
    grid = st.session_state.grid
    Q = st.session_state.Q

    for _ in range(episodes):
        start_r, start_c = get_start_pos(grid)
        state = (start_r, start_c, 0)  # mask = 0, no rewards collected
        G = 0.0

        for t in range(max_steps):
            a = epsilon_greedy_action(state, Q, epsilon)
            next_state, reward, done = step_env(state, a, grid, reward_index)

            r, c, mask = state
            nr, nc, nmask = next_state

            # Max over next actions
            if done:
                best_next = 0.0
            else:
                qs_next = [Q.get((nr, nc, nmask, a2), 0.0) for a2 in range(N_ACTIONS)]
                best_next = max(qs_next)

            old_q = Q.get((r, c, mask, a), 0.0)
            td_target = reward + gamma * best_next
            new_q = old_q + alpha * (td_target - old_q)
            Q[(r, c, mask, a)] = new_q

            G += reward
            state = next_state

            if done:
                break

        st.session_state.episode_rewards.append(G)

    st.session_state.Q = Q


def animate_training_episode(alpha, gamma, epsilon, max_steps, reward_index, frame_delay, placeholder):
    """Run one training episode and animate the agent's path step-by-step (with learning)."""
    grid = st.session_state.grid
    Q = st.session_state.Q

    start_r, start_c = get_start_pos(grid)
    state = (start_r, start_c, 0)
    G = 0.0
    path = [state]

    for t in range(max_steps):
        a = epsilon_greedy_action(state, Q, epsilon)
        next_state, reward, done = step_env(state, a, grid, reward_index)

        r, c, mask = state
        nr, nc, nmask = next_state

        if done:
            best_next = 0.0
        else:
            qs_next = [Q.get((nr, nc, nmask, a2), 0.0) for a2 in range(N_ACTIONS)]
            best_next = max(qs_next)

        old_q = Q.get((r, c, mask, a), 0.0)
        td_target = reward + gamma * best_next
        new_q = old_q + alpha * (td_target - old_q)
        Q[(r, c, mask, a)] = new_q

        G += reward
        state = next_state
        path.append(state)

        fig = make_value_heatmap_with_path(
            grid,
            Q,
            path,
            title=f"Animated Training Episode — Step {t+1}",
        )
        placeholder.plotly_chart(fig, width="stretch")

        time.sleep(frame_delay)

        if done:
            break

    st.session_state.Q = Q
    st.session_state.episode_rewards.append(G)
    st.success(f"Animated one training episode (return = {G:.2f}).")



def animate_greedy_episode(max_steps, reward_index, frame_delay, placeholder):
    """
    Animate a 'learned optimal path' (under current Q): greedy actions only, no learning.
    """
    grid = st.session_state.grid
    Q = st.session_state.Q

    start_r, start_c = get_start_pos(grid)
    state = (start_r, start_c, 0)
    G = 0.0
    path = [state]

    for t in range(max_steps):
        a = greedy_action(state, Q)
        next_state, reward, done = step_env(state, a, grid, reward_index)

        G += reward
        state = next_state
        path.append(state)

        fig = make_value_heatmap_with_path(
            grid,
            Q,
            path,
            title=f"Greedy Episode (No Learning) — Step {t+1}",
        )
        placeholder.plotly_chart(fig, width="stretch")
        time.sleep(frame_delay)

        if done:
            break

    st.info(f"Greedy episode finished (return = {G:.2f}, no learning).")


def animate_episode_timelapse(num_episodes, alpha, gamma, epsilon, max_steps, reward_index, frame_delay, placeholder):
    """
    Timelapse over episodes: show final path of each episode after training it.
    """
    grid = st.session_state.grid
    Q = st.session_state.Q

    for ep in range(num_episodes):
        start_r, start_c = get_start_pos(grid)
        state = (start_r, start_c, 0)
        G = 0.0
        path = [state]

        for t in range(max_steps):
            a = epsilon_greedy_action(state, Q, epsilon)
            next_state, reward, done = step_env(state, a, grid, reward_index)

            r, c, mask = state
            nr, nc, nmask = next_state

            if done:
                best_next = 0.0
            else:
                qs_next = [Q.get((nr, nc, nmask, a2), 0.0) for a2 in range(N_ACTIONS)]
                best_next = max(qs_next)

            old_q = Q.get((r, c, mask, a), 0.0)
            td_target = reward + gamma * best_next
            new_q = old_q + alpha * (td_target - old_q)
            Q[(r, c, mask, a)] = new_q

            G += reward
            state = next_state
            path.append(state)

            if done:
                break

        st.session_state.episode_rewards.append(G)

        fig = make_value_heatmap_with_path(
            grid,
            Q,
            path,
            title=f"Episode {ep+1} / {num_episodes} (return = {G:.2f})",
        )
        placeholder.plotly_chart(fig, width="stretch")
        time.sleep(frame_delay)

    st.session_state.Q = Q
    st.success(f"Animated {num_episodes} training episodes.")

# Training & Visualizations
with right_col:
    st.subheader("Training & Visualizations")

    # Two horizontal slots for animations
    anim_col1, anim_col2 = st.columns(2)
    with anim_col1:
        train_placeholder = st.empty()   # left: training / timelapse
    with anim_col2:
        greedy_placeholder = st.empty()  # right: greedy episode

    # Buttons row
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("Run Training (batch)"):
            run_q_learning(episodes_to_run, alpha, gamma, epsilon, max_steps, reward_index)
            st.success(f"Ran {episodes_to_run} episodes of Q-learning.")
    with col_btn2:
        if st.button("Watch one training episode (animated)"):
            animate_training_episode(alpha, gamma, epsilon, max_steps, reward_index, frame_delay, train_placeholder)
    with col_btn3:
        if st.button("Watch learned optimal path (no learning)"):
            animate_greedy_episode(max_steps, reward_index, frame_delay, greedy_placeholder)

    # Timelapse uses the left animation slot
    if st.button("Train with timelapse (paths per episode)"):
        animate_episode_timelapse(episodes_to_run, alpha, gamma, epsilon, max_steps, reward_index, frame_delay, train_placeholder)

    # Plot episode returns
    if len(st.session_state.episode_rewards) > 0:
        st.markdown("**Episode Returns (sum of rewards per episode)**")
        rewards = st.session_state.episode_rewards
        fig_rewards = go.Figure()
        fig_rewards.add_trace(go.Scatter(y=rewards, mode="lines", name="Return"))
        fig_rewards.update_layout(
            xaxis_title="Episode",
            yaxis_title="Return",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_rewards, width="stretch")

    # Static value heatmap
    st.markdown("**State Value Heatmap (maxₐ Q(s, a) with no rewards collected)**")
    fig_val = make_value_heatmap_figure(grid, Q, title="Current Value Function (mask = 0)")
    st.plotly_chart(fig_val, width="stretch")

    st.caption(
        "Colors show the learned value of each state, assuming no rewards have been collected yet (mask = 0). "
        "S = Start, G = Goal, P = Pit, R = Reward tile (one-time per episode). "
        "Use the animated buttons above to watch the agent explore, learn, and follow its greedy policy."
    )
