import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from Run_util import Config, run_rag_evaluation, create_duckdb_config, create_chroma_config, create_faiss_config
import threading
import copy

LOCAL_DIR = "CDS4RAG_HEBO"
DATASET = [0]

class TwoStageOpenBoxOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.fixed_fidelity = {
            "Question_ratio": 1,
            "Question_difficulty": 1,
            "Corpus_scale": 0,
            "Dataset_category": 0
        }
        
        self.stage1_space_config = [
            {'name': 'Database_type', 'type': 'cat', 'categories': [1, 2, 3]},
            {'name': 'CHUNK_SIZE', 'type': 'int', 'lb': 256, 'ub': 1024},
            {'name': 'CHUNK_OVERLAP', 'type': 'int', 'lb': 32, 'ub': 128},
            {'name': 'embedding_temperature', 'type': 'num', 'lb': 0.0, 'ub': 1.0},
            {'name': 'embedding_num_ctx', 'type': 'int', 'lb': 512, 'ub': 2048},
            {'name': 'embedding_repeat_penalty', 'type': 'num', 'lb': 0.9, 'ub': 1.5},
            {'name': 'embedding_top_k', 'type': 'int', 'lb': 10, 'ub': 100}
        ]
        
        self.stage2_space_config = [
            {'name': 'RETRIEVER_K', 'type': 'int', 'lb': 1, 'ub': 10},
            {'name': 'chat_temperature', 'type': 'num', 'lb': 0.0, 'ub': 1.0},
            {'name': 'chat_num_ctx', 'type': 'int', 'lb': 512, 'ub': 8192},
            {'name': 'chat_repeat_penalty', 'type': 'num', 'lb': 0.9, 'ub': 1.5},
            {'name': 'chat_top_k', 'type': 'int', 'lb': 10, 'ub': 100}
        ]
        
        self.stage1_design_space = DesignSpace().parse(self.stage1_space_config)
        self.stage1_opt = HEBO(self.stage1_design_space, rand_sample=1000, scramble_seed=self.random_state)
        
        self.stage2_design_space = None
        self.stage2_opt = None
        
        self.stage1_results = []
        self.stage2_results = []
        self.global_history = []
        
        self.stage2_history_configs = []
        self.stage2_history_scores = []
        self.history_warm_start_log = []
        self.dedup_float_decimals = 3
        
        self.stage1_best_score = -np.inf
        self.this_stage_best_score = -np.inf
        self.stage1_best_params = None
        self.stage1_best_result = None
        
        self.stage2_best_score = -np.inf
        self.stage2_best_params = None
        self.stage2_best_result = None
        
        self.best_params = None
        self.best_score = -np.inf
        
        self.best_vectorstore = None
        self.output_dir = "HEBO_2_results"
        self.optimization_start_time = None
        self.current_stage1_T_base = None
        self.current_stage1_T_switch = None
    
    def _normalize_config_dict(self, config_dict):
        normalized = {}
        int_params = {
            "Database_type", "CHUNK_SIZE", "CHUNK_OVERLAP",
            "embedding_num_ctx", "embedding_top_k",
            "RETRIEVER_K", "chat_num_ctx", "chat_top_k"
        }
        for k, v in config_dict.items():
            # Unified handling of numpy types

            if isinstance(v, (np.integer,)):
                normalized[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                normalized[k] = round(float(v), self.dedup_float_decimals)
            else:
                if k in int_params:
                    try:
                        normalized[k] = int(v)
                    except Exception:
                        normalized[k] = v
                else:
                    normalized[k] = v
        return normalized

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def _config_key(self, config_dict):
        norm = self._normalize_config_dict(config_dict)
        return tuple(sorted(norm.items()))

    def _merge_duplicate_configs(self):
        """Merge duplicates in historical configs, averaging performance (with floating point tolerance)
        
        Returns:
            Tuple[List[dict], List[float]]: Deduplicated config list (dicts) and corresponding average score list
        """
        if len(self.stage2_history_configs) == 0:
            return [], []
        
        config_to_scores = {}
        config_to_obj = {}
        
        for config, score in zip(self.stage2_history_configs, self.stage2_history_scores):
            if hasattr(config, 'get_dictionary'):
                cfg_dict = config.get_dictionary()
            elif isinstance(config, dict):
                cfg_dict = config
            elif isinstance(config, pd.Series):
                cfg_dict = config.to_dict()
            else:
                try:
                    cfg_dict = dict(config)
                except Exception:
                    continue
            cfg_key = self._config_key(cfg_dict)
            if cfg_key not in config_to_scores:
                config_to_scores[cfg_key] = []
                config_to_obj[cfg_key] = cfg_dict
            config_to_scores[cfg_key].append(score)
        
        unique_configs = []
        avg_scores = []
        for cfg_key, scores in config_to_scores.items():
            unique_configs.append(config_to_obj[cfg_key])
            avg_scores.append(float(np.mean(scores)))
        
        print(f"Historical config deduplication: Original {len(self.stage2_history_configs)}, Unique {len(unique_configs)}")
        if len(unique_configs) < len(self.stage2_history_configs):
            print(f"Merged {len(self.stage2_history_configs) - len(unique_configs)} duplicate configs (using floating point tolerance)")
        
        return unique_configs, avg_scores

    def _select_initial_configs_from_history(self, num_configs=3):
        """Select excellent configurations from history as initial points based on probability (after deduplication)
        
        Args:
            num_configs: Number of configurations to select
            
        Returns:
            List[dict]: Selected configuration list (dicts)
        """
        unique_configs, avg_scores = self._merge_duplicate_configs()
        
        if len(unique_configs) == 0:
            return []
        
        sorted_indices = np.argsort(avg_scores)[::-1]
        
        top_50_percent = max(1, len(sorted_indices) // 2)
        top_indices = sorted_indices[:top_50_percent]
        
        ranks = np.arange(1, len(top_indices) + 1)
        probabilities = 1.0 / ranks
        probabilities = probabilities / probabilities.sum()
        
        num_to_select = min(num_configs, len(top_indices))
        selected_indices = np.random.choice(
            top_indices, 
            size=num_to_select, 
            replace=False, 
            p=probabilities
        )
        
        selected_configs = [unique_configs[i] for i in selected_indices]
        
        print(f"Selected {len(selected_configs)} excellent configurations from deduplicated history as initial points")
        print(f"Historical average scores of selected configs: {[avg_scores[i] for i in selected_indices]}")
        
        for idx, config_idx in enumerate(selected_indices):
            config_dict = unique_configs[config_idx]
            config_str = str(sorted(config_dict.items()))
            
            def cfg_to_dict(c):
                if hasattr(c, 'get_dictionary'):
                    return c.get_dictionary()
                elif isinstance(c, dict):
                    return c
                elif isinstance(c, pd.Series):
                    return c.to_dict()
                else:
                    try:
                        return dict(c)
                    except Exception:
                        return {}
            count = sum(1 for c in self.stage2_history_configs 
                        if str(sorted(cfg_to_dict(c).items())) == config_str)
            if count > 1:
                print(f"  Config {idx+1} appeared {count} times in history")
        
        return selected_configs

    def _initialize_stage2_optimizer_with_history(self, num_initial_from_history=2, num_random=3, cycle_count=0):
        """Initialize Stage 2 optimizer using historical configs + random configs (HEBO)"""
        self.stage2_design_space = DesignSpace().parse(self.stage2_space_config)
        
        if len(self.stage2_history_configs) == 0:
            print("First cycle, using standard initialization (5 random points)")
            self.stage2_opt = HEBO(
                self.stage2_design_space,
                rand_sample=5,
                scramble_seed=self.random_state,
            )
            return
        
        historical_configs = self._select_initial_configs_from_history(num_initial_from_history)
        
        self.stage2_opt = HEBO(
            self.stage2_design_space,
            rand_sample=0,
            scramble_seed=self.random_state * (cycle_count + 1),
        )
        
        print(f"\nRe-evaluating {len(historical_configs)} historical configs (using current best vectorstore)")
        
        unique_configs, avg_scores = self._merge_duplicate_configs()
        
        def cfg_key_dict(d):
            return self._config_key(d)
        
        for idx, cfg_dict in enumerate(historical_configs):
            unique_config_idx = None
            hist_key = cfg_key_dict(cfg_dict)
            for i, unique_cfg in enumerate(unique_configs):
                if cfg_key_dict(unique_cfg) == hist_key:
                    unique_config_idx = i
                    break
            
            old_score = avg_scores[unique_config_idx] if unique_config_idx is not None else 0.0
            
            obj_value = self.objective_function_stage2(cfg_dict, vectorstore=self.best_vectorstore)
            new_score = -obj_value
            
            rec_df = pd.DataFrame([cfg_dict])
            self.stage2_opt.observe(rec_df, np.array([obj_value]))
            
            print(f"  Config {idx+1}: Hist Avg Score={old_score:.4f}, New Score={new_score:.4f}, Delta={new_score-old_score:+.4f}")
            
            self.history_warm_start_log.append({
                "cycle": int(cycle_count),
                "config": {k: (int(v) if isinstance(v, (np.int32, np.int64)) else float(v) if isinstance(v, (np.float32, np.float64)) else v)
                           for k, v in cfg_dict.items()},
                "old_avg_score": float(old_score),
                "new_score": float(new_score),
                "delta": float(new_score - old_score),
                "timestamp": float(time.time() - self.optimization_start_time)
            })
        
        print(f"\nAdding {num_random} random sampling configs")
        rec_rand = self.stage2_opt.suggest(n_suggestions=num_random)
        for i in range(len(rec_rand)):
            cfg_dict = rec_rand.iloc[i].to_dict()
            obj_value = self.objective_function_stage2(cfg_dict, vectorstore=self.best_vectorstore)
            self.stage2_opt.observe(rec_rand.iloc[[i]], np.array([obj_value]))
            print(f"  Random Config {i+1}: Score={-obj_value:.4f}")

    def create_config_from_params(self, params, stage=1):
        if stage == 1:
            db_type = int(params["Database_type"])
            
            if db_type == 1:
                config = create_duckdb_config()
            elif db_type == 2:
                config = create_chroma_config()
            elif db_type == 3:
                config = create_faiss_config()
            else:
                config = Config()
                
            for param_name, param_value in params.items():
                if hasattr(config, param_name):
                    if param_name in ['CHUNK_SIZE', 'CHUNK_OVERLAP', 'embedding_num_ctx', 'embedding_top_k']:
                        param_value = int(param_value)
                    setattr(config, param_name, param_value)
            
        else:
            db_type = int(self.stage1_best_params['Database_type'])
            
            if db_type == 1:
                config = create_duckdb_config()
            elif db_type == 2:
                config = create_chroma_config()
            elif db_type == 3:
                config = create_faiss_config()
            else:
                config = Config()
                
            for param_name, param_value in self.stage1_best_params.items():
                if hasattr(config, param_name):
                    if param_name in ['CHUNK_SIZE', 'CHUNK_OVERLAP', 'embedding_num_ctx', 'embedding_top_k']:
                        param_value = int(param_value)
                    setattr(config, param_name, param_value)
            
            for param_name, param_value in params.items():
                if hasattr(config, param_name):
                    if param_name in ['RETRIEVER_K', 'chat_num_ctx', 'chat_top_k']:
                        param_value = int(param_value)
                    setattr(config, param_name, param_value)
        
        for param_name, param_value in self.fixed_fidelity.items():
            if hasattr(config, param_name):
                setattr(config, param_name, param_value)
                
        return config
    
    def objective_function_stage1(self, params):
        try:
            config = self.create_config_from_params(params, stage=1)
            start_time = time.time()
            timed_out = False
            
            def timeout_handler():
                print("Warning: Config runtime exceeded 1000s, forcing evaluation interruption")
                raise TimeoutError("Config evaluation timeout")
            
            try:
                timer = threading.Timer(1000, timeout_handler)
                timer.start()
                vectorstore, results = run_rag_evaluation(config, only_retrieval=True)
                timer.cancel()
                end_time = time.time()
                
                if end_time - start_time > 1000:
                    timed_out = True
                    
            except TimeoutError:
                end_time = time.time()
                timed_out = True
                
            except Exception as eval_error:
                try:
                    timer.cancel()
                except:
                    pass
                    
                end_time = time.time()
                
                if end_time - start_time > 1000:
                    timed_out = True
                return 0.0
            
            if timed_out:
                result_entry = {
                    'params': params,
                    'mrr': 0.0,
                    'ndcg': 0.0,
                    'context_similarity': 0.0,
                    'total_time': end_time - start_time,
                    'total_tokens': 0,
                    'timed_out': True
                }
                
                self.stage1_results.append(result_entry)
                global_time = time.time() - self.optimization_start_time
                self.global_history.append({
                    'stage': 1,
                    'iteration': len(self.stage1_results),
                    'timestamp': global_time,
                    'context_similarity': 0.0,
                    'lexical_ac': None,
                    'timed_out': True
                })
                return 0.0
            
            context_similarity = results['retrieval_metrics']['context_similarity']
            result_entry = {
                'params': params,
                'mrr': results['retrieval_metrics']['mean_reciprocal_rank'],
                'ndcg': results['retrieval_metrics']['ndcg'],
                'context_similarity': context_similarity,
                'total_time': results['total_time_seconds'],
                'total_tokens': results['retrieval_token_usage']['total_tokens'],
                'timed_out': False,
                'vectorstore': vectorstore
            }
            
            iteration = len(self.stage1_results) + 1
            self.stage1_results.append(result_entry)
            
            global_time = time.time() - self.optimization_start_time
            self.global_history.append({
                'stage': 1,
                'iteration': iteration,
                'timestamp': global_time,
                'context_similarity': context_similarity,
                'lexical_ac': None,
                'timed_out': False
            })
            
            if context_similarity > self.stage1_best_score:
                self.stage1_best_score = context_similarity
                self.stage1_best_params = params.to_dict()
                self.stage1_best_result = result_entry
                self.best_vectorstore = vectorstore

            if context_similarity > self.this_stage_best_score:
                self.best_vectorstore = vectorstore
                self.this_stage_best_score = context_similarity
            
            return -context_similarity
            
        except Exception as e:
            return 0.0
    
    def objective_function_stage2(self, params, vectorstore=None):
        try:
            config = self.create_config_from_params(params, stage=2)
            start_time = time.time()
            timed_out = False
            
            def timeout_handler():
                print("Warning: Config runtime exceeded 1000s, forcing evaluation interruption")
                raise TimeoutError("Config evaluation timeout")
            
            try:
                timer = threading.Timer(1000, timeout_handler)
                timer.start()
                used_vectorstore = vectorstore if vectorstore is not None else self.best_vectorstore
                results = run_rag_evaluation(config, only_retrieval=False, saved_vectorstore=used_vectorstore)
                timer.cancel()
                end_time = time.time()
                
                if end_time - start_time > 1000:
                    timed_out = True
                    
            except TimeoutError:
                end_time = time.time()
                timed_out = True
                
            except Exception as eval_error:
                try:
                    timer.cancel()
                except:
                    pass
                    
                end_time = time.time()
                
                if end_time - start_time > 1000:
                    timed_out = True
                return 0.0
            
            if timed_out:
                result_entry = {
                    'params': params,
                    'lexical_ac': 0.0,
                    'context_similarity': 0.0,
                    'total_time': end_time - start_time,
                    'total_tokens': 0,
                    'timed_out': True
                }
                
                iteration = len(self.stage2_results) + 1
                self.stage2_results.append(result_entry)
                
                global_time = time.time() - self.optimization_start_time
                self.global_history.append({
                    'stage': 2,
                    'iteration': iteration,
                    'timestamp': global_time,
                    'context_similarity': 0.0,
                    'lexical_ac': 0.0,
                    'timed_out': True
                })
                return 0.0
            
            lexical_ac = results['generation_metrics']['answer_f1_score']
            context_similarity = results['retrieval_metrics']['context_similarity']
            
            result_entry = {
                'params': params,
                'lexical_ac': lexical_ac,
                'context_similarity': context_similarity,
                'answer_precision': results['generation_metrics']['answer_precision'],
                'answer_f1': results['generation_metrics']['answer_f1_score'],
                'avg_similarity': results['average_similarity'],
                'total_time': results['total_time_seconds'],
                'total_tokens': results['generation_token_usage']['total_tokens'] + results['retrieval_token_usage']['total_tokens'],
                'timed_out': False
            }
            
            iteration = len(self.stage2_results) + 1
            self.stage2_results.append(result_entry)
            
            global_time = time.time() - self.optimization_start_time
            self.global_history.append({
                'stage': 2,
                'iteration': iteration,
                'timestamp': global_time,
                'context_similarity': context_similarity,
                'lexical_ac': lexical_ac,
                'timed_out': False
            })
            
            if lexical_ac > self.stage2_best_score:
                self.stage2_best_score = lexical_ac
                self.stage2_best_params = params
                self.stage2_best_result = result_entry
                
                if lexical_ac > self.best_score:
                    self.best_score = lexical_ac
                    self.best_params = {**self.stage1_best_params, **self.stage2_best_params}
            
            return -lexical_ac
            
        except Exception as e:
            return 0.0

    def run_time_based_optimization(self, total_time=3600, stage_ratio=0.5,
                                    output_dir=None,
                                    num_initial_from_history=3, num_random_initial=2,
                                    beta=2.0, alpha=0.5,
                                    min_stage1_trials=1,
                                    min_stage1_trials_first_cycle=2):
        """
        Time-based Cyclic Optimization (Adaptive Threshold + Historical Warm-start)
        - Stage 1: Adaptive threshold based on historical best and current quantile.
        - Stage 2: Warm-started with good configurations from previous cycles (HEBO).
        """
        if output_dir:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        cycle_count = 0
        stage1_total_iter = 0
        stage2_total_iter = 0

        total_start_time = time.time()
        self.optimization_start_time = total_start_time
        total_end_time = total_start_time + total_time

        while time.time() < total_end_time:
            stage2_single_iter = 0
            cycle_count += 1
            print(f"\n--- Starting Optimization Cycle {cycle_count} ---")

            remaining_time = total_end_time - time.time()
            if remaining_time <= 0:
                break

            stage1_time = min(remaining_time * stage_ratio, remaining_time)
            stage1_end_time = time.time() + stage1_time

            warmup_trials = (min_stage1_trials_first_cycle
                             if cycle_count == 1 and min_stage1_trials_first_cycle is not None
                             else min_stage1_trials)
            stage1_iter_in_cycle = 0

            self.current_cycle_stage1_scores = []

            T_round = cycle_count
            ramp = T_round / (T_round + beta)
            S_ref = self.stage1_best_score if self.stage1_best_score != -np.inf else 0.0
            S_best = 0.0 if self.this_stage_best_score == -np.inf else self.this_stage_best_score

            def _estimate_S_target():
                all_scores = [r['context_similarity'] for r in self.stage1_results if not r.get('timed_out', False)]
                all_scores += list(self.current_cycle_stage1_scores)
                if not all_scores:
                    return 0.3
                q_hat = 0.95
                S_q = float(np.quantile(all_scores, q_hat))
                return S_q

            S_target = _estimate_S_target()
            T_base = S_ref + (S_target - S_ref) * ramp
            self.current_stage1_T_base = T_base

            def q50(values):
                if len(values) >= 2:
                    return float(np.quantile(values, 0.5))
                return max(values) if values else 0.0

            q50_val = q50(self.current_cycle_stage1_scores)
            T_switch = alpha * q50_val + (1 - alpha) * T_base
            self.current_stage1_T_switch = T_switch

            print(f"Stage 1: Allocated Time {stage1_time:.1f}s")
            print(f"Stage 1 Threshold Init: S_ref={S_ref:.4f}, ramp={ramp:.4f}, S_target={S_target:.4f}, T_base={T_base:.4f}, q50={q50_val:.4f}, Init T_switch={T_switch:.4f}, S_best(init)={S_best:.4f}")
            print(f"Stage 1 Warmup Threshold: Allow early switch only after at least {warmup_trials} trials")

            while time.time() < stage1_end_time:
                stage1_total_iter += 1
                stage1_iter_in_cycle += 1
                rec = self.stage1_opt.suggest(n_suggestions=1)
                score = self.objective_function_stage1(rec.iloc[0])
                self.stage1_opt.observe(rec, np.array([score]))
                current_context = -score

                self.current_cycle_stage1_scores.append(current_context)
                S_best = max(S_best, current_context)

                old_T_switch = self.current_stage1_T_switch
                q50_val = q50(self.current_cycle_stage1_scores)
                T_switch = alpha * q50_val + (1 - alpha) * T_base
                self.current_stage1_T_switch = T_switch

                print("rec:", rec.iloc[0].to_dict(), "score(Context Similarity):", current_context)
                print("iter %d: Context Similarity=%.4f" % (stage1_total_iter, current_context))
                if T_switch > old_T_switch + 1e-8:
                    print(f"Threshold Update: q50={q50_val:.4f}, T_switch {old_T_switch:.4f} -> {T_switch:.4f}")

                if self.global_history and self.global_history[-1]['stage'] == 1:
                    self.global_history[-1]['T_base'] = T_base
                    self.global_history[-1]['T_switch'] = T_switch
                    self.global_history[-1]['ramp'] = ramp
                    self.global_history[-1]['S_ref'] = S_ref
                    self.global_history[-1]['q80'] = q50_val

                if stage1_iter_in_cycle < warmup_trials:
                    print(f"Stage 1 Warmup: {stage1_iter_in_cycle}/{warmup_trials}, skip early stop check")
                    continue

                if S_best >= T_switch:
                    print(f"Stage 1 Early Stop: S_best={S_best:.4f} >= T_switch={T_switch:.4f}")
                    break

            remaining_time = total_end_time - time.time()
            if remaining_time <= 0:
                break
            stage2_time = remaining_time
            stage2_end_time = time.time() + stage2_time
            print(f"Stage 2: Allocated Time {stage2_time:.1f}s")

            self._initialize_stage2_optimizer_with_history(
                num_initial_from_history=num_initial_from_history,
                num_random=num_random_initial, cycle_count=cycle_count
            )

            while time.time() < stage2_end_time:
                stage2_single_iter += 1
                stage2_total_iter += 1

                rec = self.stage2_opt.suggest(n_suggestions=1)
                params = rec.iloc[0].to_dict()
                obj_value = self.objective_function_stage2(params, vectorstore=self.best_vectorstore)

                self.stage2_history_configs.append(params)
                self.stage2_history_scores.append(-obj_value)

                self.stage2_opt.observe(rec, np.array([obj_value]))

                print("iter %d: F1 score=%.4f" % (stage2_total_iter, -obj_value))

                if stage2_total_iter >= 10:
                    end_iter = 5
                else:
                    end_iter = 10

                if stage2_single_iter >= end_iter:
                    print(f"Stage 2: Second round iteration exceeded {10}, early stopping")
                    break
            
            print(f"Cycle {cycle_count} Results:")
            print(f"  Stage 1 Iterations: {stage1_iter_in_cycle}")
            print(f"  Stage 2 Iterations: {stage2_single_iter}")
            print(f"- Stage 1 Best Context-Similarity: {self.stage1_best_score:.4f}")
            print(f"- Stage 2 Best F1 score: {self.stage2_best_score:.4f}")
            print(f"- Global Best F1 score: {self.best_score:.4f}")
            
            self.this_stage_best_score = -np.inf
            self.stage2_design_space = DesignSpace().parse(self.stage2_space_config)
            self.stage2_opt = HEBO(
                self.stage2_design_space,
                rand_sample=2, 
                scramble_seed=self.random_state*(cycle_count+1),
            )
    
            print(f"- Best F1 score: {self.best_score:.4f}")
        
        # Calculate total time and save results
        total_time_used = time.time() - total_start_time
        print(f"\nOptimization Completed. Time: {total_time_used:.1f}s, Total {cycle_count} cycles")
        print(f"Stage 1 Total Iter: {stage1_total_iter}, Stage 2 Total Iter: {stage2_total_iter}")
        
        self.save_results(total_time_used, stage1_total_iter, stage2_total_iter, cycle_count)
        
        return self.best_params, self.best_score
    
    def save_results(self, total_time, stage1_iter, stage2_iter, cycle_count=1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        all_results = []
        
        for i, record in enumerate(self.global_history):
            stage = record['stage']
            
            if stage == 1:
                iteration = record.get('iteration', 0)
                if 0 < iteration <= len(self.stage1_results):
                    result_copy = {}
                    for key, value in self.stage1_results[iteration-1].items():
                        if key != 'vectorstore':
                            result_copy[key] = value
                else:
                    result_copy = {
                        'context_similarity': record['context_similarity'],
                        'timed_out': record.get('timed_out', False)
                    }
                
                result_copy['stage'] = 1
                result_copy['global_iteration'] = i + 1
                result_copy['timestamp'] = record['timestamp']
                all_results.append(result_copy)
            else:
                iteration = record.get('iteration', 0)
                if 0 < iteration <= len(self.stage2_results):
                    result_copy = {}
                    for key, value in self.stage2_results[iteration-1].items():
                        result_copy[key] = value
                else:
                    result_copy = {
                        'context_similarity': record['context_similarity'],
                        'lexical_ac': record['lexical_ac'],
                        'timed_out': record.get('timed_out', False)
                    }
                    
                result_copy['stage'] = 2
                result_copy['global_iteration'] = i + 1
                result_copy['timestamp'] = record['timestamp']
                all_results.append(result_copy)
        
        if all_results:
            all_df = pd.DataFrame(all_results)
            all_csv = os.path.join(self.output_dir, f"time_based_results_{timestamp}.csv")
            all_df.to_csv(all_csv, index=False)
        
        def convert_numpy_types(obj):
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        stage1_params = {}
        if self.stage1_best_params:
            for k, v in self.stage1_best_params.items():
                stage1_params[k] = v
        
        stage2_params = {}
        if self.stage2_best_params:
            for k, v in self.stage2_best_params.items():
                stage2_params[k] = v
        
        best_params = {}
        if self.best_params:
            for k, v in self.best_params.items():
                best_params[k] = v
        
        stage1_params = convert_numpy_types(stage1_params)
        stage2_params = convert_numpy_types(stage2_params)
        best_params = convert_numpy_types(best_params)
        
        summary = {
            "optimization_info": {
                "method": "Time-based Cyclic HEBO with Historical Warm-start",
                "total_cycles": cycle_count,
                "stage1_iterations": int(stage1_iter),
                "stage2_iterations": int(stage2_iter),
                "stage2_historical_configs": len(self.stage2_history_configs),
                "random_state": int(self.random_state),
                "total_time_seconds": float(total_time)
            },
            "stage1_best": {
                "context_similarity": float(self.stage1_best_score),
                "params": stage1_params
            },
            "stage2_best": {
                "lexical_ac": float(self.stage2_best_score),
                "params": stage2_params
            },
            "overall_best": {
                "lexical_ac": float(self.best_score),
                "params": best_params
            }
        }
        
        summary_json = os.path.join(self.output_dir, f"time_based_summary_{timestamp}.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        
        self.plot_optimization_history(timestamp, cycle_count)
        
        self.save_history_details(timestamp)
            
    def plot_optimization_history(self, timestamp, cycle_count=1):
        if not self.global_history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        context_scores = []
        lexical_scores = []
        timestamps = []
        stage_markers = []
        iteration_numbers = []
        
        for i, record in enumerate(self.global_history):
            iteration_numbers.append(i + 1)
            timestamps.append(record['timestamp'])
            context_scores.append(record['context_similarity'])
            lexical_scores.append(record['lexical_ac'] if record['lexical_ac'] is not None else 0.0)
            stage_markers.append(record['stage'])
        
        context_best = np.maximum.accumulate(context_scores)
        lexical_best = []
        current_best_lexical = -np.inf
        for score in lexical_scores:
            if score > current_best_lexical and score > 0:
                current_best_lexical = score
            lexical_best.append(current_best_lexical if current_best_lexical > 0 else 0)
        
        ax1 = axes[0]
        stage1_indices = [i for i, stage in enumerate(stage_markers) if stage == 1]
        stage2_indices = [i for i, stage in enumerate(stage_markers) if stage == 2]
        
        if stage1_indices:
            stage1_iters = [iteration_numbers[i] for i in stage1_indices]
            stage1_context = [context_scores[i] for i in stage1_indices]
            ax1.plot(stage1_iters, stage1_context, 'bo-', alpha=0.7, label='Context Similarity (Stage 1)')
        
        if stage2_indices:
            stage2_iters = [iteration_numbers[i] for i in stage2_indices]
            stage2_context = [context_scores[i] for i in stage2_indices]
            stage2_lexical = [lexical_scores[i] for i in stage2_indices]
            ax1.plot(stage2_iters, stage2_context, 'b^--', alpha=0.7, label='Context Similarity (Stage 2)')
            ax1.plot(stage2_iters, stage2_lexical, 'go-', alpha=0.7, label='Lexical AC (Stage 2)')
        
        ax1.plot(iteration_numbers, context_best, 'b-', linewidth=2, label='Best Context Similarity')
        if stage2_indices:
            ax1.plot(iteration_numbers, lexical_best, 'g-', linewidth=2, label='Best Lexical AC')
        
        ax1.set_xlabel('Iteration Number')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Optimization History by Iteration (Cycles: {cycle_count})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        timestamps_min = [t / 60 for t in timestamps]
        
        if stage1_indices:
            stage1_times = [timestamps_min[i] for i in stage1_indices]
            stage1_context = [context_scores[i] for i in stage1_indices]
            ax2.plot(stage1_times, stage1_context, 'bo-', alpha=0.7, label='Context Similarity (Stage 1)')
        
        if stage2_indices:
            stage2_times = [timestamps_min[i] for i in stage2_indices]
            stage2_context = [context_scores[i] for i in stage2_indices]
            stage2_lexical = [lexical_scores[i] for i in stage2_indices]
            ax2.plot(stage2_times, stage2_context, 'b^--', alpha=0.7, label='Context Similarity (Stage 2)')
            ax2.plot(stage2_times, stage2_lexical, 'go-', alpha=0.7, label='Lexical AC (Stage 2)')
        
        ax2.plot(timestamps_min, context_best, 'b-', linewidth=2, label='Best Context Similarity')
        if stage2_indices:
            ax2.plot(timestamps_min, lexical_best, 'g-', linewidth=2, label='Best Lexical AC')
        
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Score')
        ax2.set_title(f'Optimization History by Time (Cycles: {cycle_count})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"optimization_history_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_history_details(self, timestamp):
        """Extra save of full history (Stage 2 evaluation trajectory + warm start evaluation effect + global history)"""
        stage2_history = []
        for cfg, score in zip(self.stage2_history_configs, self.stage2_history_scores):
            if hasattr(cfg, 'get_dictionary'):
                cfg_dict = cfg.get_dictionary()
            elif isinstance(cfg, dict):
                cfg_dict = cfg
            elif isinstance(cfg, pd.Series):
                cfg_dict = cfg.to_dict()
            else:
                try:
                    cfg_dict = dict(cfg)
                except Exception:
                    cfg_dict = {}
            safe_cfg = {k: (int(v) if isinstance(v, (np.int32, np.int64))
                            else float(v) if isinstance(v, (np.float32, np.float64))
                            else v)
                        for k, v in cfg_dict.items()}
            stage2_history.append({"config": safe_cfg, "score": float(score)})
        
        warm_start_evals = self.history_warm_start_log
        
        global_hist = []
        for rec in self.global_history:
            global_hist.append({
                "stage": int(rec.get("stage", 0)),
                "iteration": int(rec.get("iteration", 0)) if rec.get("iteration", 0) is not None else 0,
                "timestamp": float(rec.get("timestamp", 0.0)),
                "context_similarity": float(rec.get("context_similarity", 0.0)) if rec.get("context_similarity") is not None else 0.0,
                "lexical_ac": float(rec.get("lexical_ac", 0.0)) if rec.get("lexical_ac") is not None else 0.0,
                "timed_out": bool(rec.get("timed_out", False)),
            })
        
        data = {
            "history_file_version": 1,
            "output_dir": self.output_dir,
            "random_state": int(self.random_state),
            "stage2_history_total": len(stage2_history),
            "warm_start_total": len(warm_start_evals),
            "stage2_history": stage2_history,
            "warm_start_evaluations": warm_start_evals,
            "global_history": global_hist,
        }
        
        hist_json = os.path.join(self.output_dir, f"time_based_history_{timestamp}.json")
        with open(hist_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def run_multiple_experiments(num_experiments=10, total_time=3600, stage_ratio=0.5,
                             base_seed=42, context_threshold=0.85, dataset_categories=None):
    """Run multiple random experiments and summarize results
    
    Args:
        num_experiments: Number of experiments per dataset category
        total_time: Total time for each experiment (seconds)
        stage_ratio: Time ratio for first stage in each cycle
        base_seed: Base random seed
        context_threshold: Context-Similarity early stopping threshold
        dataset_categories: List of dataset categories to run, defaults to [0,1,2,3]
    """
    if dataset_categories is None:
        dataset_categories = [0, 1, 2, 3]
        
    all_results = {}
    all_best = {}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments_root_dir = os.path.join(LOCAL_DIR, f"multi_dataset_exp_{timestamp}")
    os.makedirs(experiments_root_dir, exist_ok=True)
    
    for dataset_category in dataset_categories:
        print(f"\n\n======== Dataset Category {dataset_category} ========")
        
        dataset_dir = os.path.join(experiments_root_dir, f"dataset_{dataset_category}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        results = []
        best_overall = {"score": -np.inf, "params": None, "exp_id": None}
        
        for exp_id in range(num_experiments):
            print(f"\n==== Experiment {exp_id+1}/{num_experiments} (Dataset Category: {dataset_category}) ====")
            seed = base_seed + exp_id
            
            exp_dir = os.path.join(dataset_dir, f"exp_{exp_id}")
            os.makedirs(exp_dir, exist_ok=True)
            
            optimizer = TwoStageOpenBoxOptimizer(random_state=seed)
            
            optimizer.fixed_fidelity = {
                "Question_ratio": 0.5,
                "Question_difficulty": 1,
                "Corpus_scale": 0,
                "Dataset_category": dataset_category
            }
            
            best_params, best_score = optimizer.run_time_based_optimization(
                total_time=total_time,
                stage_ratio=stage_ratio,
                output_dir=exp_dir
            )
            
            exp_result = {
                "exp_id": exp_id,
                "seed": seed,
                "dataset_category": dataset_category,
                "best_score": best_score,
                "best_params": best_params,
                "stage1_best_score": optimizer.stage1_best_score,
                "stage1_iterations": len(optimizer.stage1_results),
                "stage2_iterations": len(optimizer.stage2_results)
            }
            results.append(exp_result)

            if best_score > best_overall["score"]:
                best_overall["score"] = best_score
                best_overall["params"] = copy.deepcopy(best_params)
                best_overall["exp_id"] = exp_id
        
        summarize_multiple_experiments(results, dataset_dir, best_overall)
        
        all_results[dataset_category] = results
        all_best[dataset_category] = best_overall
    
    summarize_all_datasets(all_results, all_best, experiments_root_dir, dataset_categories)
    
    return all_results, all_best

def summarize_multiple_experiments(results, experiments_dir, best_overall):
    """Summarize multiple experiment results"""
    scores = [r["best_score"] for r in results]
    context_scores = [r["stage1_best_score"] for r in results]
    stage1_iters = [r["stage1_iterations"] for r in results]
    stage2_iters = [r["stage2_iterations"] for r in results]
    
    summary = {
        "num_experiments": len(results),
        "lexical_ac": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        },
        "context_similarity": {
            "mean": float(np.mean(context_scores)),
            "std": float(np.std(context_scores)),
            "min": float(np.min(context_scores)),
            "max": float(np.max(context_scores)),
            "median": float(np.median(context_scores))
        },
        "iterations": {
            "stage1_mean": float(np.mean(stage1_iters)),
            "stage1_total": int(np.sum(stage1_iters)),
            "stage2_mean": float(np.mean(stage2_iters)),
            "stage2_total": int(np.sum(stage2_iters))
        },
        "best_overall": {
            "exp_id": int(best_overall["exp_id"]),
            "score": float(best_overall["score"]),
            "params": best_overall["params"]
        }
    }
    
    if best_overall["params"]:
        for key, value in best_overall["params"].items():
            if isinstance(value, (np.int32, np.int64)):
                best_overall["params"][key] = int(value)
            elif isinstance(value, (np.float32, np.float64)):
                best_overall["params"][key] = float(value)
    
    summary_path = os.path.join(experiments_dir, "experiments_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    
    print("\n=== Multi-Experiment Summary ===")
    print(f"Number of experiments: {len(results)}")
    print(f"Lexical AC: Avg {float(np.mean(scores)):.4f} ± {float(np.std(scores)):.4f}, Max {float(np.max(scores)):.4f}")
    print(f"Context Similarity: Avg {float(np.mean(context_scores)):.4f} ± {float(np.std(context_scores)):.4f}")
    print(f"Best Experiment: Exp {best_overall['exp_id']}, Lexical AC: {best_overall['score']:.4f}")

def summarize_all_datasets(all_results, all_best, root_dir, dataset_categories):
    """Summarize experiment results for all datasets"""
    summary = {
        "datasets": {},
        "best_per_dataset": {},
        "overall_best": {"dataset": None, "exp_id": None, "score": -np.inf}
    }
    
    for category in dataset_categories:
        results = all_results[category]
        best = all_best[category]
        
        scores = [r["best_score"] for r in results]
        context_scores = [r["stage1_best_score"] for r in results]
        
        summary["datasets"][str(category)] = {
            "num_experiments": len(results),
            "lexical_ac": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores))
            },
            "context_similarity": {
                "mean": float(np.mean(context_scores)),
                "std": float(np.std(context_scores)),
                "min": float(np.min(context_scores)),
                "max": float(np.max(context_scores)),
                "median": float(np.median(context_scores))
            }
        }
        
        summary["best_per_dataset"][str(category)] = {
            "exp_id": int(best["exp_id"]),
            "score": float(best["score"]),
            "params": best["params"]
        }
        
        if best["params"]:
            for key, value in best["params"].items():
                if isinstance(value, (np.int32, np.int64)):
                    best["params"][key] = int(value)
                elif isinstance(value, (np.float32, np.float64)):
                    best["params"][key] = float(value)
        
        if best["score"] > summary["overall_best"]["score"]:
            summary["overall_best"]["score"] = float(best["score"])
            summary["overall_best"]["dataset"] = int(category)
            summary["overall_best"]["exp_id"] = int(best["exp_id"])
    
    summary_path = os.path.join(root_dir, "all_datasets_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    print("\n=== Multi-Dataset Experiment Summary ===")
    print(f"Dataset Categories: {dataset_categories}")
    for category in dataset_categories:
        mean_score = summary["datasets"][str(category)]["lexical_ac"]["mean"]
        max_score = summary["datasets"][str(category)]["lexical_ac"]["max"]
        print(f"Dataset {category}: Avg Lexical AC {mean_score:.4f}, Max {max_score:.4f}")
    
    best_dataset = summary["overall_best"]["dataset"]
    best_score = summary["overall_best"]["score"]
    print(f"Global Best: Dataset {best_dataset}, Lexical AC: {best_score:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Time-based BO Cyclic Optimization (using HEBO + Historical Initialization)')
    parser.add_argument('--total_time', type=int, default=3600, help='Total optimization time (seconds)')
    parser.add_argument('--stage_ratio', type=float, default=0.5, help='Ratio of time for first stage in each cycle')
    parser.add_argument('--num_experiments', type=int, default=10, help='Number of experiments per dataset')
    parser.add_argument('--base_seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--context_threshold', type=float, default=0.85, help='Context-Similarity early stopping threshold')
    parser.add_argument('--dataset_categories', type=int, nargs='+', default=DATASET, 
                        help='Dataset categories to test, e.g., 0 1 2 3 4 5')
    args = parser.parse_args()

    all_results, all_best = run_multiple_experiments(
        num_experiments=args.num_experiments,
        total_time=args.total_time,
        stage_ratio=args.stage_ratio,
        base_seed=args.base_seed,
        context_threshold=args.context_threshold,
        dataset_categories=args.dataset_categories
    )

if __name__ == "__main__":
    main()