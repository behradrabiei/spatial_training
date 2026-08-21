import numpy as np
import random
from typing import Dict, Any
def get_dummy_state() -> Dict[str, Any]:
    '''
    generate dummy state dict for smoke tests.
    '''
    return {
            "obs":{
                "instr_or_goal": "dummy_instruction" #initial system prompt
                },
            "done": random.random() < 0.05,
            "reward": random.random(), 
            "is_exhausted": False,
            "info": {}
            }
class DummyEnvActor:
    '''
    Dummy Environment that demonstrates the interface but uses dummy RGB data.
    Reward is a trivial bandit problem: reward of 1 for stop, small negative reward otherwise. Episode ends when stop is taken or randomly with small probability.
    '''
    def step(self, action: int, supplementary_logs: Dict[str, Any] = None):
        # generate random RGB data and state dict
        print(f"step {self.sc} of dummy env")
        self.sc += 1
        rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state = get_dummy_state()
        state['reward'] = -0.01 if action != 0 else 1.0 # reward of 1 for stop, small negative reward otherwise
        state['done'] = state['done'] or action==0 # end episode
        return rgb, state
    def reset(self):
        self.sc = 0
        print("resetting dummy env")
        rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state = get_dummy_state()
        state['done'] = False # ensure not done at reset
        return rgb, state
    
    def assign_shard(self, episodes: list[str]|None = None):
        '''
        assign a list of episodes identified via strings to the actor.
        if None is passed, load all available episodes.
        '''
        pass
    
    def flush_logs_to_disk(self):
        '''
        flush any internal logging. returns either None or a path pointing to a json file.
        '''
        pass
    
    def is_exhausted(self):
        '''
        returns True if the actor has exhausted its assigned episodes.
        '''
        return False


class MultiGoalDummyEnvActor(DummyEnvActor):
    '''
    Dummy env for the multi-object conversation path: switches instr_or_goal /
    goal_idx every `steps_per_goal` steps (the way MultiObjectHabitatEnvActor
    does on a correct stop). Ends on a stop at the final goal or once every
    goal's budget is spent. Deterministic, unlike DummyEnvActor.
    '''
    def __init__(self, goals=("red cube", "green ball", "blue chair"), steps_per_goal=4):
        self.goals = list(goals)
        self.steps_per_goal = steps_per_goal

    def _state(self):
        state = get_dummy_state()
        goal_idx = min(self.sc // self.steps_per_goal, len(self.goals) - 1)
        state['obs'] = {
            "instr_or_goal": self.goals[goal_idx],
            "goal_idx": goal_idx,
            "goal_sequence": list(self.goals),
        }
        state['done'] = False
        state['info'] = {"goal_idx": goal_idx}
        return state

    def reset(self):
        self.sc = 0
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), self._state()

    def step(self, action: int, supplementary_logs: Dict[str, Any] = None):
        self.sc += 1
        state = self._state()
        on_last_goal = state['obs']['goal_idx'] == len(self.goals) - 1
        state['reward'] = 1.0 if action == 0 and on_last_goal else -0.01
        state['done'] = (action == 0 and on_last_goal) or self.sc >= self.steps_per_goal * len(self.goals)
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), state
