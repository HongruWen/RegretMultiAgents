import os, sys
import numpy as np

# -------------------------------------------------------------------
#  Make sure the parent directory is on the import path
#  (handy when you run from a sub‑folder or an IDE).
# -------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model_manager import ModelManager
from LanguageAgent import WaterworldAgent
from OnlineRegretTracker import OnlineRegretTracker   # ← all‑lower‑case

# -------------------------------------------------------------------
#  helper that wraps simulation in regret tracking
# -------------------------------------------------------------------
def run_with_regret(agents, env, regret_tracker, planning_horizon=5, seed=42):
    print("=== Starting Simulation (with regret tracker) ===")

    # deterministic reset handled by tracker
    obs = regret_tracker.start_episode(seed=seed)
    if isinstance(obs, tuple):          # PettingZoo variant
       obs = obs[0]   

    # prime LLM prompts
    for agent in agents.values():
        agent.reset()

    done = {aid: False for aid in env.agents}

    # ---------- heartbeat so you see progress -----------
    print("[heartbeat] entering main while‑loop – first LLM query coming…")
    # -----------------------------------------------------

    while not all(done.values()):
        joint_action = {}
        for aid in env.agents:
            if not done[aid]:
                agents[aid].observe(obs[aid], reward=0, term=False,
                                    trunc=False, info={})
                joint_action[aid] = agents[aid].act()   # <‑‑ can block
            else:
                joint_action[aid] = None

        obs, rewards, terminations, truncations, infos = env.step(joint_action)
        regret_tracker.record_step(joint_action, rewards)

        done = {
            aid: terminations.get(aid, False) or truncations.get(aid, False)
            for aid in env.agents
        }

    print("=== Simulation Completed ===")

    regret_tracker.compute_regrets()
    regret_tracker.plot_regret_trajectories(show=True,
                                            save_path="regret_plot.png")
    env.close()


# -------------------------------------------------------------------
#  (original teammate helper – kept for reference)
# -------------------------------------------------------------------
def main(agents, env, planning_horizon=5):
    print("=== Starting Simulation ===")
    env.reset()
    for a in agents.values():
        a.reset()
    for agent_name in env.agent_iter():
        obs, rew, term, trunc, info = env.last()
        agents[agent_name].observe(obs, rew, term, trunc, info)
        env.step(None if (term or trunc) else agents[agent_name].act())
    print("=== Simulation Completed ===")
    env.close()


# -------------------------------------------------------------------
#  entry point
# -------------------------------------------------------------------
def waterworld():
    from pettingzoo.sisl import waterworld_v4

    print("Setting up WaterWorld environment…")
    env = waterworld_v4.parallel_env(
        n_pursuers=1,
        n_evaders=5,
        n_poisons=10,
        n_sensors=20,
        sensor_range=0.2,
        radius=0.015,
        obstacle_radius=0.2,
        n_obstacles=1,
        obstacle_coord=[(0.5, 0.5)],
        pursuer_max_accel=0.01,
        evader_speed=0.01,
        poison_speed=0.01,
        poison_reward=-1.0,
        food_reward=10.0,
        encounter_reward=0.01,
        thrust_penalty=-0.5,
        speed_features=True,
        max_cycles=30,
        render_mode="human",
    )
    print("Environment created successfully")

    # ---------- model manager ----------
    mm = ModelManager()
    mm.add_model(
        name="deepseek-qwen-32b",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        temperature=0.2,
        max_length=256,
    )

    # ---------- LLM agents -------------
    planning_horizon = 5
    agents = {
        aid: WaterworldAgent(
            name=aid,
            model=mm.get_next_model(),
            env=env,
            planning_horizon=planning_horizon,
        )
        for aid in env.possible_agents
    }

    # ---------- baseline + tracker -----
    baseline_policies = {aid: "greedy" for aid in env.possible_agents}
    tracker = OnlineRegretTracker(env, baseline_policies)

    print("\nStarting main simulation loop (with regret tracker)…")
    run_with_regret(agents, env, tracker,
                    planning_horizon=planning_horizon, seed=42)


if __name__ == "__main__":
    waterworld()
