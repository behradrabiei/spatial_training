"""Multi-object (sequential goal) habitat worker, built for OneMap's benchmark
("One Map to Find Them All", ICRA 2025; episodes converted by
tools/convert_onemap_episodes.py).

Episodes carry their goal sequence in episode.info['object_goals']. STOP is
intercepted instead of forwarded to habitat: a correct stop on a non-final goal
advances to the next goal in place (the episode continues from wherever the
agent stands), a correct stop on the final goal or a wrong stop passes through
and ends the episode. Each leg has its own step budget, and per-leg SPL uses the
geodesic distance from the leg's actual start pose, mirroring OneMap's evaluator.

Sub-goal success mirrors habitat's ObjectNav criterion: geodesic distance from
the agent to the current category's goal view-points below success_distance.
"""

import os
import time

import numpy as np

from longnav.env.habitat import HabitatEnvActor, HabitatWorker

# Final-step info keys (registered into summary_schema so the flush-time filter
# keeps them in the results JSONL).
MULTI_SUMMARY_KEYS = (
    "progress", "legs_succeeded", "all_success", "ppl", "leg_results",
    "leg_spls", "leg_steps_list", "leg_path_lens", "leg_best_dists",
    "goal_sequence", "done_reason", "onemap_episode_id", "goal_idx",
)


class MultiObjectHabitatWorker(HabitatWorker):
    def __init__(self, *args, leg_max_steps=500, leg_success_dist=None, **kwargs):
        self.leg_max_steps = leg_max_steps
        self.leg_success_dist = leg_success_dist
        self._multi = None
        self._pending = None
        super().__init__(*args, **kwargs)

    def _success_dist(self):
        if self.leg_success_dist is not None:
            return self.leg_success_dist
        return self.config_env.habitat.task.measurements.success.success_distance

    def _goal_view_positions(self, category):
        # habitat_env.current_episode is the full episode; the gym wrapper's
        # current_episode() returns a stripped BaseEpisode without info/goals
        episode = self.env.habitat_env.current_episode
        key = f"{os.path.basename(episode.scene_id)}_{category}"
        goals = self.env.habitat_env._dataset.goals_by_category[key]
        return [vp.agent_state.position for g in goals for vp in g.view_points]

    def _geodesic_to_goal(self, category):
        sim = self.env.habitat_env.sim
        pos = sim.get_agent_state().position
        # episode arg deliberately omitted: passing it would poison the episode's
        # goal-1 _shortest_path_cache used by the distance_to_goal measure.
        return sim.geodesic_distance(pos, self._goal_view_positions(category))

    def assign_shard(self, assigned_episode_labels=None):
        super().assign_shard(assigned_episode_labels)
        self.summary_schema.update({k: True for k in MULTI_SUMMARY_KEYS})

    def _reset(self, episode_id=None, output_schema=None, logging_schema=None):
        self._multi = None
        self._pending = None
        return super()._reset(episode_id, output_schema, logging_schema)

    def step(self, action: int, supplementary_logs={}):
        action, guard_extras = self._apply_stop_guards(action)
        if action != 0:
            self._pending = "move"
            return super().step(action, supplementary_logs, _stop_guard_extras=guard_extras)
        m = self._multi
        d = self._geodesic_to_goal(m["goals"][m["goal_idx"]])
        success = np.isfinite(d) and d < self._success_dist()
        if success and m["goal_idx"] < len(m["goals"]) - 1:
            self._pending = "success_advance"
            return self._fabricated_stop_step(supplementary_logs, guard_extras)
        self._pending = "success_final" if success else "wrong_stop"
        return super().step(0, supplementary_logs, _stop_guard_extras=guard_extras)

    def _apply_stop_guards(self, action):
        """Apply stop guards against the active sub-goal, not Habitat's goal 1."""
        extras = {
            "+fp_stop": -99999 * int(not self.fp_guard),
            "+fn_stop": -99999 * int(not self.fn_guard),
        }
        m = self._multi
        distance = self._geodesic_to_goal(m["goals"][m["goal_idx"]])
        inside_goal = np.isfinite(distance) and distance < self._success_dist()
        if action == 0 and not inside_goal:
            if self.fp_guard:
                action = int(np.random.choice([1, 2, 3]))
            extras["+fp_stop"] = 1
        elif action != 0 and inside_goal:
            if self.fn_guard:
                action = 0
            extras["+fn_stop"] = 1
        return action, extras

    def _fabricated_stop_step(self, supplementary_logs={}, guard_extras=None):
        """A stop that advances the goal without stepping habitat (which would end
        the episode). Deliberately mirrors the tail of HabitatWorker.step so the
        cached lists stay aligned with real steps — the logger and video renderer
        index by them."""
        self.reset_flag = False
        step_dict = {
            "obs": dict(self.last_step["obs"]),
            "reward": 0.0,
            "done": False,
            "info": dict(self.last_step["info"]),
        }
        extras = guard_extras or {
            "+fp_stop": -99999 * int(not self.fp_guard),
            "+fn_stop": -99999 * int(not self.fn_guard),
        }
        if self.postprocess:
            step_dict = self._postprocess_step(step_dict)
            extras["+stuck"] = False
        if self.enable_caching:
            extras["timestamp"] = time.time()
            supplementary_logs = {
                (f"sup/{k}" if not k.startswith("sup/") else k): v
                for k, v in (supplementary_logs or {}).items()
            }
            self._cache_step(self._apply_schema(step_dict | extras, self.logging_schema) | supplementary_logs)
            self.steps["action"].append(0)
        self.last_step = step_dict
        return self._apply_schema(step_dict | extras, self.output_schema)

    def _postprocess_step(self, step_dict):
        step_dict = super()._postprocess_step(step_dict)
        obs, info = step_dict["obs"], step_dict["info"]
        pos = np.array(info["+pos_rots"][:3])
        if self._multi is None:  # first postprocess of the episode (reset path)
            ep_info = self.env.habitat_env.current_episode.info or {}
            if "object_goals" not in ep_info:
                raise RuntimeError(
                    "multi_object worker needs episodes with info['object_goals'] "
                    "(build the dataset with tools/convert_onemap_episodes.py)")
            self._multi = {
                "goals": list(ep_info["object_goals"]),
                "goal_idx": 0,
                "leg_steps": 0,
                "leg_path_len": 0.0,
                "leg_best_dist": self._geodesic_to_goal(ep_info["object_goals"][0]),
                "last_pos": pos,
                "legs": [],
                "onemap_episode_id": ep_info.get("onemap_episode_id", -1),
            }
        elif self._pending is not None:  # a real agent step (fabricated stops included)
            m = self._multi
            m["leg_path_len"] += float(np.linalg.norm(pos - m["last_pos"]))
            m["last_pos"] = pos
            m["leg_steps"] += 1
            kind = self._pending
            if kind == "move" and m["leg_steps"] >= self.leg_max_steps:
                self._record_leg(m, success=False, result="oot")
                self._finalize(info, "oot")
                step_dict["done"] = True  # habitat wasn't stopped; end the episode ourselves
            elif kind == "success_advance":
                self._record_leg(m, success=True, result="success")
                m["goal_idx"] += 1
                m["leg_steps"] = 0
                m["leg_path_len"] = 0.0
                m["leg_best_dist"] = self._geodesic_to_goal(m["goals"][m["goal_idx"]])
            elif kind == "success_final":
                self._record_leg(m, success=True, result="success")
                self._finalize(info, "all_success")
            elif kind == "wrong_stop":
                self._record_leg(m, success=False, result="wrong_stop")
                self._finalize(info, "wrong_stop")
        m = self._multi
        obs["+instr_or_goal"] = m["goals"][m["goal_idx"]]
        obs["+goal_idx"] = m["goal_idx"]
        obs["+goal_sequence"] = list(m["goals"])
        info["+goal_idx"] = m["goal_idx"]
        self._pending = None
        return step_dict

    @staticmethod
    def _record_leg(m, success, result):
        path_len, best = m["leg_path_len"], m["leg_best_dist"]
        if success and np.isfinite(best):
            # OneMap's per-leg SPL, guarded for zero-length legs
            spl = min(1.0, best / max(path_len, best)) if max(path_len, best) > 0 else 1.0
        else:
            spl = 0.0
        m["legs"].append({
            "category": m["goals"][m["goal_idx"]],
            "result": result,
            "steps": m["leg_steps"],
            "path_len": path_len,
            "best_dist": float(best) if np.isfinite(best) else -1.0,
            "spl": spl,
        })

    def _finalize(self, info, done_reason):
        m = self._multi
        n_goals = len(m["goals"])
        n_success = sum(leg["result"] == "success" for leg in m["legs"])
        info["+progress"] = n_success / n_goals
        info["+legs_succeeded"] = n_success
        info["+all_success"] = float(n_success == n_goals)
        info["+ppl"] = sum(leg["spl"] for leg in m["legs"]) / n_goals
        info["+leg_results"] = ",".join(leg["result"] for leg in m["legs"])
        info["+leg_spls"] = ",".join(f"{leg['spl']:.4f}" for leg in m["legs"])
        info["+leg_steps_list"] = ",".join(str(leg["steps"]) for leg in m["legs"])
        info["+leg_path_lens"] = ",".join(f"{leg['path_len']:.3f}" for leg in m["legs"])
        info["+leg_best_dists"] = ",".join(f"{leg['best_dist']:.3f}" for leg in m["legs"])
        info["+goal_sequence"] = ",".join(m["goals"])
        info["+done_reason"] = done_reason
        info["+onemap_episode_id"] = m["onemap_episode_id"]


class MultiObjectHabitatEnvActor(HabitatEnvActor, MultiObjectHabitatWorker):
    # MRO: HabitatEnvActor -> LoggingHabitatWorker -> MultiObjectHabitatWorker -> HabitatWorker
    pass
