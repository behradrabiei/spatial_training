import ray
from collections import deque
from typing import List, Dict, Any, Iterator, Optional, Tuple
from string import Template
from PIL import Image
from longnav.utils.vlm_worker import VLMWorker
import numpy as np
import time 

def select_action(
    action_probs,
    *,
    deterministic: bool = False,
    stop_prob_threshold: Optional[float] = None,
) -> Tuple[int, bool]:
    """Select an action id from action probabilities.

    Returns:
        (action_id, spguard_triggered)
    """
    pick = (lambda p: int(np.argmax(p))) if deterministic \
        else (lambda p: int(np.random.choice(len(p), p=p / p.sum())))
    action_id = pick(action_probs)
    if action_id == 0 and stop_prob_threshold is not None and action_probs[0] < stop_prob_threshold:
        return pick(action_probs[1:]) + 1, True
    return action_id, False

def substitute_convo_template(conversation_template: List[Dict], substitutions: Dict[str, Any]) -> List[Dict]:
    """
    Traverses the conversation template and substitutes any string.Template 
    objects found in 'text' fields using values from the 'obs' dictionary.
    
    Args:
        conversation_template: List of message dicts (role, content).
        substitutions: Dictionary containing substitution keys (e.g., 'instr_or_goal').
        
    Returns:
        A new conversation list with strings substituted.
    """
    new_conversation = []
    
    for message in conversation_template:
        # Shallow copy the message container
        new_message = message.copy()
        new_content = []
        
        # Iterate over the content list (e.g., [{"type": "image"}, {"type": "text", ...}])
        for item in message.get("content", []):
            new_item = item.copy()
            
            # Check if this item is a text component
            if "text" in new_item:
                text_obj = new_item["text"]
                
                # CASE A: It's a Template object (from the config)
                if "$" in text_obj:
                    try:
                        text_template = Template(text_obj)
                        # Perform the substitution
                        new_item["text"] = text_template.substitute(substitutions)
                    except KeyError as e:
                        raise
                        # Fallback to safe_substitute to prevent crashing on missing keys,
                        # but log it so we know something is wrong.
                        # print(f"Warning: Missing substitution key {e} in template.")
                        # new_item["text"] = text_template.safe_substitute(substitutions)
                        
                # CASE B: It's already a str (static text)
                elif isinstance(text_obj, str):
                    pass # Keep as is
                    
            new_content.append(new_item)
        new_message["content"] = new_content
        new_conversation.append(new_message)
        
    return new_conversation

class EpisodeRolloutMixin:
    def _pack_trajectory(self, buffer: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Converts list of dicts to a dict of numpy arrays (Columnar format).
        This format allows Ray to zero-copy transfer individual columns.
        """
        if not buffer:
            return {}
        
        # Fast dictionary of lists to list of dictionaries inversion
        keys = buffer[0].keys()
        stacked = {k: np.array([d[k] for d in buffer]) for k in keys}
        
        # Optimization: Cast probabilities to float32 to save 50% bandwidth
        if "probs" in stacked:
            stacked["action_probs"] = stacked["action_probs"].astype(np.float32)
        if "rewards" in stacked:
             stacked["rewards"] = stacked["rewards"].astype(np.float32)    
        return stacked
    
    def run_episode(self,env_handle, initial_state_ref,collect_trajectory=False):
        """
        Returns:
        - is_exhausted: Whether the sim ran out of episodes
        - final_info: The final info dict from the sim (contains success, spl, etc.)
        - trajectory: If collect_trajectory is True, returns the collected trajectory in columnar format (dict of numpy arrays).
        """
        import time
        self.reset() #we reset at the start to ensure clean state. not resetting at the end preserves state for downstream.
        try:
            # 1. Resolve the initial state (Blocking wait for reset to finish)
            # Ray automatically waits for initial_state_ref to be ready before starting this task,
            # but we call ray.get to access the data.
            # 'auto' keeps the legacy behaviour (attached patch coords => bev position ids);
            # 'standard' keeps stock mRoPE and only forwards the coords to the KV-prune
            # voxel selectors (see VLMWorker._turn_voxel_rows).
            pos_id_mode = self.rollout_config.get("pos_id_mode", "auto")
            pos_id_kwargs={
                "mode": "standard"
            }
            if len(initial_state_ref)==2:
                rgb,state_dict = initial_state_ref
                print("using normal mode")
            elif len(initial_state_ref)==3:
                rgb,patch_coords,state_dict = initial_state_ref
                pos_id_kwargs['patch_coords'] = patch_coords
                pos_id_kwargs['mode'] = "bev" if pos_id_mode == "auto" else pos_id_mode
                print(f"using {pos_id_kwargs['mode']} mode (patch coords attached)")
            step_count = 0
            done = False
            messages = substitute_convo_template(self.rollout_config['convo_start_template'],state_dict['obs'] | self.rollout_config)
            # 2. The Interaction Loop
            vlm_logs={}
            # Trajectory Buffer (List is fine here!)
            trajectory_buffer = []
            instr_or_goal = state_dict['obs']['instr_or_goal']
            if hasattr(self, "begin_episode"):
                self.begin_episode(state_dict.get('info', {}).get('episode_label', ''))
            while not done and step_count < self.rollout_config['max_steps']:
                # A. Prepare VLM Input
                rgb_numpy = rgb
                rgb_pil = Image.fromarray(rgb_numpy)
                # B. Call VLM (Blocking)
                # We must wait for the answer to decide the next step
                # print("inferring VLM with messages:")
                # print(messages)
                t0 = time.time()
                action_probs,action_logprobs,outputs = self.infer_probs(images=[rgb_pil],messages=messages,temperature = self.rollout_config['temperature'],pos_id_kwargs=pos_id_kwargs)
                
                vlm_logs |= {'mean/vlm_latency':time.time()-t0,'min/vlm_latency':time.time()-t0,'max/vlm_latency':time.time()-t0,'sum/vlm_latency':time.time()-t0,'sum/spguard_trigger_count':0}
                kv_stats = self.kv_stats()
                if kv_stats is not None:
                    # Resident cache length after any eviction/pruning -- the memory metric
                    # the context-compression experiments are scored on. kv_len is the
                    # layer-mean (memory-equivalent) length; layers differ only under
                    # layer-selective pruning, where kv_len_layer_max / kv_len_master expose
                    # the unpruned and master lengths.
                    kv_mean, kv_layer_max, kv_master = kv_stats
                    vlm_logs |= {'mean/kv_len': kv_mean, 'max/kv_len': kv_mean,
                                 'max/kv_len_layer_max': kv_layer_max,
                                 'mean/kv_len_master': kv_master}
                if getattr(self, "kv_prune_log_influence", False) or getattr(self, "influence_recompute", False):
                    # Leave-one-frame-out influence (paper/frame_influence), from the prune-mode
                    # replay probe or the leak-free recompute probe -- same schema either way.
                    # Ragged per-step rows (len t+1) land raw in sequence.json as
                    # sup/kl_influence; kv_visual_new marks frames whose KL is vacuous.
                    # Companion rows: hidden-frame action log-probs per frame, the base
                    # log-probs, and the joint probes ([kl, logp]; [[K, kl, logp], ...] for the
                    # age bands). No '/' in these names: a reduction prefix would make the
                    # logger average ragged lists.
                    vlm_logs["kl_influence"] = list(getattr(self, "_kl_influence_row", []))
                    vlm_logs["mean/kv_visual_new"] = getattr(self, "_prune_visual_new", 0)
                    vlm_logs["kl_influence_logp"] = list(getattr(self, "_kl_influence_logp", []))
                    vlm_logs["kl_influence_base_logp"] = list(getattr(self, "_kl_influence_base_logp", []))
                    vlm_logs["kl_influence_history_all"] = list(getattr(self, "_kl_influence_history_all", []))
                    vlm_logs["kl_influence_visual_all"] = list(getattr(self, "_kl_influence_visual_all", []))
                    vlm_logs["kl_influence_horizon"] = list(getattr(self, "_kl_influence_horizon", []))
                    # Leave-one-in rows (vlm.kv_prune_influence_measures contains keep_one): only
                    # frame f's visual slots visible, reference = visual_all; [] otherwise.
                    vlm_logs["kl_influence_keep_one"] = list(getattr(self, "_kl_influence_keep_one", []))
                    vlm_logs["kl_influence_keep_one_logp"] = list(getattr(self, "_kl_influence_keep_one_logp", []))
                if getattr(self, "influence_recompute", False):
                    # Cumulative counters: their episode total is the final value, so reduce with max.
                    vlm_logs |= {"max/influence_probe_count": self._influence_probe_count,
                                 "max/influence_probe_latency": self._influence_probe_latency}
                if getattr(self, "horizon_probe_layers", ()):
                    # paper/info_horizon: per-layer keep-one-in / leave-one-out KL of the current
                    # frame (raw rows in sequence.json as sup/horizon_*; no '/' in the names).
                    row = getattr(self, "_horizon_row", {}) or {}
                    vlm_logs["horizon_layers"] = list(self.horizon_probe_layers)
                    vlm_logs["horizon_keep_one"] = list(row.get("keep_one", []))
                    vlm_logs["horizon_leave_one"] = list(row.get("leave_one", []))
                    vlm_logs["horizon_base_logp"] = list(row.get("base_logp", []))
                    vlm_logs["mean/kv_visual_new"] = row.get("n_cur_tokens", 0)
                    vlm_logs |= {"max/horizon_probe_count": self._horizon_probe_count,
                                 "max/horizon_probe_latency": self._horizon_probe_latency}
                if getattr(self, "selector_probe", False):
                    # paper/selector_kl: per visual token of the current frame, the flat selectors'
                    # layer-0 criteria and the leaky per-token KLs (raw rows in sequence.json as
                    # sup/selector_*; no '/' in the names).
                    from longnav.utils.selector_probe import log_rows
                    vlm_logs |= log_rows(self)
                if getattr(self, "context_window_mode", None) == "prune":
                    vlm_logs |= {
                        "mean/prune_score_latency": self._prune_score_latency,
                        "sum/prune_score_latency": self._prune_score_latency,
                        "sum/prune_replay_count": self._prune_replay_count,
                        "mean/kv_visual_slots": self._prune_visual_slots,
                        "mean/kv_text_slots": self._prune_text_slots,
                    }
                    # Staged arms: visual slots held by each stage's layers (shallow to deep).
                    for s_idx, n_vis in enumerate(getattr(self, "_prune_stage_visual", [])):
                        vlm_logs[f"mean/kv_stage_visual_{s_idx}"] = n_vis
                    if getattr(self, "_prune_visual_model", None) is not None:
                        # Visual slots averaged over all decoder layers, raw and in frames' worth.
                        vlm_logs["mean/kv_visual_model"] = self._prune_visual_model
                        if self._prune_visual_model_frames is not None:
                            vlm_logs["mean/kv_visual_model_frames"] = self._prune_visual_model_frames
                            vlm_logs["max/kv_visual_model_frames"] = self._prune_visual_model_frames
                    if getattr(self, "kv_prune_log_layer_influence", False):
                        # One KL per probe layer (vlm.kv_prune_layer_influence_starts), raw in
                        # sequence.json as sup/layer_influence_{hist,all}; [] on skipped steps.
                        vlm_logs["layer_influence_hist"] = list(getattr(self, "_layer_influence_hist", []))
                        vlm_logs["layer_influence_all"] = list(getattr(self, "_layer_influence_all", []))
                    if getattr(self, "kv_prune_log_keep_one", False):
                        # Per probe layer, per cached frame: keep-one-in and leave-one-out KL
                        # (sup/keep_one_influence, sup/leave_one_influence); [] on skipped steps.
                        vlm_logs["keep_one_influence"] = [list(r) for r in getattr(self, "_keep_one_rows", [])]
                        vlm_logs["leave_one_influence"] = [list(r) for r in getattr(self, "_leave_one_rows", [])]
                        vlm_logs["mean/kv_visual_new"] = getattr(self, "_prune_visual_new", 0)
                    if getattr(self, "_vtp_log", None):
                        # paper/token_prune: the window's per-frame keep masks after this step's
                        # prune (raw in sequence.json as sup/vtp_*; see VLMWorker._log_vtp). With
                        # kv_prune_log_masks off only the scalar mean/vtp_n_new is carried.
                        vlm_logs |= dict(self._vtp_log)
                try:
                    import torch
                    vlm_mem = torch.cuda.memory_allocated()/(1024**3)
                    vlm_logs |= {"vlm_mem_GB": vlm_mem, "max/vlm_mem_GB": vlm_mem}
                except:
                    print("warning: could not get vlm mem")
                if self.rollout_config.get("visualize_token_filtering"):
                    viz = self.get_filter_visualization()
                    if viz is not None:
                        vlm_logs["vis_keep_mask"] = viz[0]
                        vlm_logs["image_grid_thw"] = viz[1]
                if self.rollout_config.get("visualize_attention"):
                    aviz = self.get_attention_visualization()
                    if aviz is not None:
                        vlm_logs["attn_map"] = aviz[0]
                        vlm_logs["attn_grid_thw"] = aviz[1]
                if self.rollout_config.get("visualize_attention_heads"):
                    hviz = self.get_attention_heads_visualization()
                    if hviz is not None:
                        vlm_logs["attn_heads"] = hviz[0]
                        vlm_logs["attn_grid_thw"] = hviz[1]
                if self.rollout_config.get("visualize_attention_3d"):
                    viz3d = self.get_attention_3d_visualization()
                    if viz3d is not None:
                        vlm_logs["attn_hist"] = viz3d[0]
                        vlm_logs["attn_hist_grids"] = viz3d[1]
                        # The renderer labels its colorbar from this; the maps
                        # themselves carry no hint of what they measure.
                        vlm_logs["attn_signal"] = self.attn_weighting
                # print(f"vlm step{step_count}")
                # print("done")
                #except for the first turn, all messages follow the exact same template.
                action_id, spguard_triggered = select_action(
                    action_probs,
                    deterministic=self.rollout_config.get("deterministic", False),
                    stop_prob_threshold=self.rollout_config.get("stop_prob_threshold"),
                )
                if spguard_triggered:
                    vlm_logs['sum/spguard_trigger_count']=1
                    
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))
                vlm_logs |= {'mean/entropy':entropy,'mean/action_prob':float(action_probs[action_id]),"action_probs":action_probs.tolist()} 
                # D. Store Transition
               

                # D. Step Simulator (Blocking) ---------------------------RAY----------------------------- 
                t0 = time.time()
                # del rgb,state_dict
                state_ref = ray.get(env_handle.step.remote(action_id,supplementary_logs=vlm_logs))
                if len(state_ref)==2:
                    rgb,state_dict = state_ref
                    pos_id_kwargs.pop('patch_coords', None)
                elif len(state_ref)==3:
                    rgb,patch_coords,state_dict = state_ref
                    pos_id_kwargs['patch_coords'] = patch_coords
                    pos_id_kwargs['mode'] = "bev" if pos_id_mode == "auto" else pos_id_mode
                vlm_logs = {'mean/sim_latency':time.time()-t0,'min/sim_latency':time.time()-t0,'max/sim_latency':time.time()-t0}
                if collect_trajectory:
                    # Append dict to list - fast and simple
                    trajectory_dict = {
                        "actions": action_id,
                        "rollout_logprobs": action_logprobs,
                        "rollout_probs": action_probs,
                        "rewards": state_dict.get("reward", 0.0),
                        "dones": state_dict['done'],
                        # "spl": state_dict['info']['spl'],
                        # "success": state_dict['info']['success'],
                        # "distance_to_goal": state_dict['info']['distance_to_goal'],
                        **state_dict['info'],
                    }
                    trajectory_buffer.append(trajectory_dict)
                messages = substitute_convo_template(self.rollout_config['convo_turn_template'],{"action":self.rollout_config['action_space'][action_id]})
                # print(f"sim step{step_count}")
                done = state_dict['done']
                step_count += 1
                # Convert list of dicts -> Dict of Numpy Arrays (Zero-Copy Friendly)
            final_trajectory = self._pack_trajectory(trajectory_buffer) if collect_trajectory else None
            final_info = state_dict['info'] | {"steps":step_count, "instr_or_goal":instr_or_goal}
            # Return Clean Tuple (No Actor Handles here)
            return state_dict['is_exhausted'], final_info, final_trajectory
        
        except Exception as e:
            print(f"Episode failed: {e}")
            import traceback
            traceback.print_exc()
            # Return handles anyway so we don't leak resources (or handle crash logic)
            return False, None,None

class RolloutWorker(VLMWorker, EpisodeRolloutMixin):
    def __init__(self, rollout_config: Dict[str, Any], **vlm_kwargs):
        """
        Explicitly handles argument separation to avoid MRO issues.
        
        Args:
            rollout_config: Arguments intended for the EpisodeRolloutMixin.
            **vlm_kwargs: All other arguments (model_id, dtype, etc.) passed to VLMWorker.
        """
        # 1. Initialize the VLM Worker (The Heavy Lifter)
        # We pass only the relevant VLM args to avoid 'unexpected keyword argument' errors.
        vlm_kwargs["visualize_attention"] = rollout_config.get("visualize_attention", False)
        vlm_kwargs["visualize_attention_heads"] = rollout_config.get("visualize_attention_heads", False)
        vlm_kwargs["visualize_attention_3d"] = rollout_config.get("visualize_attention_3d", False)
        vlm_kwargs["attn3d_layers"] = rollout_config.get("attn3d_layers", [-1])
        VLMWorker.__init__(self, **vlm_kwargs)
        import os
        # 2. Initialize the Mixin State
        # Since the Mixin's __init__ was just setting this variable, we can do it here directly
        # effectively bypassing the need for cooperative inheritance in the parents.
        self.rollout_config = rollout_config

    
        
class InferenceWorker(RolloutWorker):
    """Stateful model inference with the actor interface used by collect_rollouts."""

    def load_adapter(self, path, peft_config=None):
        from longnav.utils.checkpoints import load_adapter
        load_adapter(self, path, peft_config)

    def run_episode(self, env_handle, initial_state_ref):
        self.save_pixels = False
        exhausted, result, self.trajectory = super().run_episode(
            env_handle, initial_state_ref, collect_trajectory=True)
        return exhausted, result

    def postprocess_episode(self, return_inputs=False, eval=True):
        if return_inputs or not eval:
            raise ValueError("Inference workers do not return training inputs or compute training log-probabilities")
        return self.trajectory, None, None


def collect_rollouts(
    env_handles: list,
    vlm_handles: list,
    shard_iterator: Iterator[list[str]],
    target_episodes: int = float('inf'),
    postprocess_kwargs = None,
    wandb_logger = None
) -> tuple[list,list,list]:
    """
    Collect inference episodes and flush logs in dispatch order.

    returns: trajectory buffer, result list, log list, indexed by dispatch_id
    """

    postprocess_kwargs = postprocess_kwargs or {}

    # --- 1. Initialize Pools ---
    idle_vlms = deque(vlm_handles)
    ready_sims = deque()

    # --- 2. Tracking Futures ---
    pending_resets = {}   # reset_ref -> sim_handle
    active_episodes = {}  # ep_ref -> dispatch_id

    # VLM post-processing
    pending_postproc = {} # pp_ref -> vlm_handle, dispatch_id 
    # Sim logging
    pending_logs = {} # log_ref -> sim_handle, dispatch_id

    trajectory_buffer = []
    trajectory_ids = []

    result_dict = {}
    log_dict = {}
    iterator_exhausted = False

    last_dispatch_time = time.time()
    # --- 3. Bootstrap: Initial Sharding & Resets ---
    for env_handle in env_handles:
        try:
            if ray.get(env_handle.is_exhausted.remote()):
                initial_shard = next(shard_iterator)
                env_handle.assign_shard.remote(initial_shard)
            reset_ref = env_handle.reset.remote()
            pending_resets[reset_ref] = env_handle
        except StopIteration:
            iterator_exhausted = True
            print("Warning: Not enough shards for all workers during bootstrap.")
            pass
    print(f"Bootstrapping: Initializing {len(env_handles)} environments...")
    initial_live_sims = len(pending_resets)
    # Helper to check if we should keep the loop alive
    def has_work():

        # 1. Are tasks currently running?
        is_active = len(active_episodes) > 0 or len(pending_postproc) > 0 or len(pending_logs) > 0
        # 2. Do we still want to launch new tasks (now or in the future)? (Resources available AND Target not met)
        potential = len(trajectory_buffer) + len(active_episodes) + len(pending_postproc)
        want_launch = (potential < target_episodes) and initial_live_sims > 0
        return is_active or want_launch

    dispatch_counter = 0
    # --- Event Loop ---
    while has_work():
        # A. Dispatch (IDENTICAL)
        total_potential = len(trajectory_buffer) + len(active_episodes) + len(pending_postproc)

        while (idle_vlms and ready_sims and total_potential < target_episodes):
            vlm = idle_vlms.popleft()
            sim, init_state_ref = ready_sims.popleft()
            ep_ref = vlm.run_episode.remote(sim, init_state_ref)
            active_episodes[ep_ref] = dispatch_counter,vlm,sim
            try:  # name the episode in flight (the sim logs it only when it ends)
                _label = ray.get(init_state_ref)[-1].get("info", {}).get("episode_label", "?")
                print(f"[dispatch] episode {dispatch_counter}: {_label}", flush=True)
            except Exception:
                pass
            dispatch_counter +=1
            total_potential +=1


        # B. Wait for Events
        all_watch_refs = list(pending_resets.keys()) + \
                         list(active_episodes.keys()) + \
                         list(pending_postproc.keys()) + \
                         list(pending_logs.keys())

        if not all_watch_refs:
            break

        ready_refs, _ = ray.wait(all_watch_refs, num_returns=1,timeout=15.0)
        if not ready_refs:
            # If we get here, the orchestrator is "stuck" waiting.
            # We can use this moment to diagnose.

            # Simple deadlock detector:
            current_time = time.time()
            if current_time - last_dispatch_time > 360: # 6 minutes
                # A slow artifact render (e.g. one attn3d video per decoder layer)
                # can legitimately occupy a sim actor for minutes, so warn and
                # re-arm the timer instead of aborting the run.
                print(f"DEBUG: System frozen for >6m. Active: {len(active_episodes)}, "
                      f"PostProc: {len(pending_postproc)}, Flushing: {len(pending_logs)}, "
                      f"Resetting: {len(pending_resets)}")
                if wandb_logger is not None:
                    ray.get(wandb_logger.alert.remote(title="Rollout Collection Frozen",text=f"Active Episodes: {len(active_episodes)}, Pending PostProc: {len(pending_postproc)}",level="ERROR"))
                last_dispatch_time = current_time

            continue # Jump back to start of loop (and potentially dispatch more if resources freed up)

        # CHANGE 3: Update timestamp when we actually get a result
        last_dispatch_time = time.time()
        for ref in ready_refs:

            # --- CASE 1: Reset Finished ---
            if ref in pending_resets:
                env_handle = pending_resets.pop(ref)
                ready_sims.append((env_handle, ref))

            # --- CASE 2: Episode Finished ---
            elif ref in active_episodes:
                # print("handling finished episode")
                dispatch_id, vlm, sim =  active_episodes.pop(ref)
                # Unpack results
                is_exhausted, result = ray.get(ref)
                result_dict[dispatch_id] = result

                # send vlm and sim to post episode processing
                pp_ref = vlm.postprocess_episode.remote(**postprocess_kwargs)
                pending_postproc[pp_ref] = vlm,dispatch_id

                log_ref = sim.flush_logs_to_disk.remote()
                pending_logs[log_ref] = sim,dispatch_id,is_exhausted

            # --- CASE 3: VLM Post-Processing Finished ---
            elif ref in pending_postproc:
                vlm,dispatch_id = pending_postproc.pop(ref)
                trajectory_buffer.append(ref)
                trajectory_ids.append(dispatch_id)
                idle_vlms.append(vlm)
                print(f"Collected episode {len(trajectory_buffer)}")

            # --- CASE 4: Sim Log Flush Finished
            elif ref in pending_logs:
                sim,dispatch_id,is_exhausted = pending_logs.pop(ref)
                log_dict[dispatch_id] = ref # save the path to the log
                # send the sim to reset/reshard so it can start working again asap
                try:
                    # print("logging done",end="")
                    if is_exhausted:
                        # print("assigning shard")
                        new_shard = next(shard_iterator)
                        sim.assign_shard.remote(new_shard)
                    # print("resetting sim")
                    new_reset_ref = sim.reset.remote()
                    pending_resets[new_reset_ref] = sim
                except StopIteration:
                    # No more work. Retire the Habitat worker.
                    # iterator_exhausted = True
                    pass
    rollouts = [t for _, t in sorted(zip(trajectory_ids, trajectory_buffer))]
    log_dict |={v[1]:k for k,v in pending_logs.items()}
    num_rollouts = len(rollouts)
    result_list = [result_dict[i] for i in range(num_rollouts)]
    log_list = [log_dict[i] for i in range(num_rollouts)]
    return ray.get(rollouts), result_list, log_list


