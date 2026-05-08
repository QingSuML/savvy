import csv
import json
import os
from collections import defaultdict

import numpy as np

from .datasets import HM3DVOSDataset, ScannetVOSDataset
from .metrics import compute_sequence_stq_vpq, evaluate_vos_consistency
from .stress import (
    apply_dynamic_spatial_dilation,
    apply_macro_sever,
    apply_sparse_id_flickering,
    apply_temporal_dropout,
)


class VOSEvaluator:
    def __init__(self, 
                 dataset, 
                 ignore_label=0, 
                 iou_thr=0.5, 
                 ios_thr=0.5, 
                 cooccur_sim_thr=0.5, 
                 max_pattern_size=2):
        self.dataset = dataset
        self.ignore_label = ignore_label
        self.iou_thr = iou_thr
        self.ios_thr = ios_thr
        self.cooccur_sim_thr= cooccur_sim_thr
        self.max_pattern_size = max_pattern_size
        self.results_per_scene = {}
                
    def evaluate(self, scene_names=None):
        
        if isinstance(scene_names, str):
            s_ids = [scene_names]
        else:
            s_ids = scene_names or self.dataset.get_scene_ids()
            
        for sid in s_ids:
            pl, gl = self.dataset.load_scene_frames(sid)
            if not pl: 
                continue

            # Compute OGA metrics on the evaluated prediction masks.
            res = evaluate_vos_consistency(
                pl,
                gl, 
                self.ignore_label, 
                self.iou_thr, 
                self.ios_thr, 
                self.cooccur_sim_thr, 
                self.max_pattern_size
            )
            
            # Compute standard video segmentation baselines on the same masks.
            res['baselines'] = compute_sequence_stq_vpq(
                pl,
                gl,
                ignore_label=self.ignore_label,
                vpq_iou_threshold=0.5,
                sever_ratio_threshold=0.5,
                k_values=(0, 5, 15, 25, 35, float('inf'))
            )
            
            self.results_per_scene[sid] = res

    def evaluate_macro_sever_test(self, scene_names=None):
        num_severs = [1, 2, 3, 4]

        if isinstance(scene_names, str):
            s_ids = [scene_names]
        else:
            s_ids = scene_names or self.dataset.get_scene_ids()
            
        for sid in s_ids:
            pl, gl = self.dataset.load_scene_frames(sid)
            if not pl: 
                continue

            for num in num_severs:
                print(f"\n[*] Running Evaluation with macro sever ({num})")                
                if num > 0.0:
                    pl_stressed = apply_macro_sever(pl, num_severs=num)
                else:
                    pl_stressed = pl

                res = evaluate_vos_consistency(
                    pl_stressed,
                    gl, 
                    self.ignore_label, 
                    self.iou_thr, 
                    self.ios_thr, 
                    self.cooccur_sim_thr, 
                    self.max_pattern_size
                )
                
                # Compute standard video segmentation baselines on the stressed masks.
                res['baselines'] = compute_sequence_stq_vpq(
                    pl_stressed,
                    gl,
                    ignore_label=self.ignore_label,
                    vpq_iou_threshold=0.5,
                    k_values=(0, 5, 15, 25, 35, float('inf'))
                )
                
                self.results_per_scene[sid + f'_macro_sever_{num}'] = res
    
    def evaluate_drop_test(self, scene_names=None):
        drop_prob = [0.05, 0.1, 0.15, 0.2]

        if isinstance(scene_names, str):
            s_ids = [scene_names]
        else:
            s_ids = scene_names or self.dataset.get_scene_ids()
            
        for sid in s_ids:
            pl, gl = self.dataset.load_scene_frames(sid)
            if not pl: 
                continue

            for prob in drop_prob:
                print(f"\n[*] Running Evaluation with Dropout ({prob})")                
                if prob > 0.0:
                    pl_stressed = apply_temporal_dropout(pl, drop_prob=prob)
                else:
                    pl_stressed = pl

                res = evaluate_vos_consistency(
                    pl_stressed,
                    gl, 
                    self.ignore_label, 
                    self.iou_thr, 
                    self.ios_thr, 
                    self.cooccur_sim_thr, 
                    self.max_pattern_size
                )

                res['baselines'] = compute_sequence_stq_vpq(
                    pl_stressed,
                    gl,
                    ignore_label=self.ignore_label,
                    vpq_iou_threshold=0.5,
                    k_values=(0, 5, 15, 25, 35, float('inf'))
                )
                
                self.results_per_scene[sid + f'_dropout_{prob}'] = res

    def evaluate_clutter_test(self, scene_names=None, mode="overwrite"):
        severity_levels = [1, 2, 4, 8]

        if isinstance(scene_names, str):
            s_ids = [scene_names]
        else:
            s_ids = scene_names or self.dataset.get_scene_ids()

        for sid in s_ids:
            pl, gl = self.dataset.load_scene_frames(sid)
            if not pl:
                continue

            for severity in severity_levels:
                print(f"\n[*] Running Evaluation with Aggressive Dilation / Clutter ({mode}, Base Iters: {severity})")

                pl_stressed = apply_dynamic_spatial_dilation(
                    pl,
                    base_iters=severity,
                    ignore_label=self.ignore_label,
                    mode=mode,
                    random_order=True,
                    seed=42,
                )

                res = evaluate_vos_consistency(
                    pl_stressed,
                    gl,
                    self.ignore_label,
                    self.iou_thr,
                    self.ios_thr,
                    self.cooccur_sim_thr,
                    self.max_pattern_size
                )

                res['baselines'] = compute_sequence_stq_vpq(
                    pl_stressed,
                    gl,
                    ignore_label=self.ignore_label,
                    vpq_iou_threshold=0.5,
                    k_values=(0, 5, 15, 25, 35, float('inf'))
                )

                self.results_per_scene[sid + f'_clutter_{mode}_{severity}'] = res

    def evaluate_dilation_test(self, scene_names=None):
        severity_levels = [1, 2, 4, 8]

        if isinstance(scene_names, str):
            s_ids = [scene_names]
        else:
            s_ids = scene_names or self.dataset.get_scene_ids()

        for sid in s_ids:
            pl, gl = self.dataset.load_scene_frames(sid)
            if not pl:
                continue

            for severity in severity_levels:
                print(f"\n[*] Running Evaluation with Void-Only Dilation (Base Iters: {severity})")

                pl_stressed = apply_dynamic_spatial_dilation(
                    pl,
                    base_iters=severity,
                    ignore_label=self.ignore_label,
                    mode="void_only",
                )

                res = evaluate_vos_consistency(
                    pl_stressed,
                    gl,
                    self.ignore_label,
                    self.iou_thr,
                    self.ios_thr,
                    self.cooccur_sim_thr,
                    self.max_pattern_size
                )

                res['baselines'] = compute_sequence_stq_vpq(
                    pl_stressed,
                    gl,
                    ignore_label=self.ignore_label,
                    vpq_iou_threshold=0.5,
                    k_values=(0, 5, 15, 25, 35, float('inf'))
                )

                self.results_per_scene[sid + f'_dilation_void_{severity}'] = res

    def evaluate_flickering_test(self, scene_names=None):
        fraction_level = [0.025, 0.05, 0.075, 0.1]

        if isinstance(scene_names, str):
            s_ids = [scene_names]
        else:
            s_ids = scene_names or self.dataset.get_scene_ids()
            
        for sid in s_ids:
            pl, gl = self.dataset.load_scene_frames(sid)
            if not pl: 
                continue

            for frac in fraction_level:
                print(f"\n[*] Running Evaluation with flickering with fraction: ({frac})")                
                if frac > 0.0:
                    pl_stressed = apply_sparse_id_flickering(pl, noise_fraction=frac)
                else:
                    pl_stressed = pl

                res = evaluate_vos_consistency(
                    pl_stressed,
                    gl, 
                    self.ignore_label, 
                    self.iou_thr, 
                    self.ios_thr, 
                    self.cooccur_sim_thr, 
                    self.max_pattern_size
                )
                
                # Compute standard video segmentation baselines on the stressed masks.
                res['baselines'] = compute_sequence_stq_vpq(
                    pl_stressed,
                    gl,
                    ignore_label=self.ignore_label,
                    vpq_iou_threshold=0.5,
                    k_values=(0, 5, 15, 25, 35, float('inf'))
                )
                
                self.results_per_scene[sid + f'_flickering_frac_{frac}'] = res

    def save_results(self, output_dir="./eval_results", tag=None):
        """Exports the full nested results to JSON and a flattened macro summary to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"oga_full_results_{tag}" if tag is not None else "oga_full_results"

        # Save full nested JSON.
        json_path = os.path.join(output_dir, filename + ".json")
        with open(json_path, 'w') as f:
            json.dump(self.results_per_scene, f, indent=4)
            
        # Save flattened macro CSV.
        csv_path = os.path.join(output_dir, filename + ".csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            header = [
                "Scene ID", "STQ", "AQ", "GQ", 
                "VPQ_0", "VPQ_5", "VPQ_15", "VPQ_25", "VPQ_35", "VPQ_inf",
                "IP (C)", "IP (P)", "IP (G)", "IC (P)", "IC (G)", "Spat.IP", "T.Bleed",
                "TS (C)", "Cluster", "Patt k=1", "Patt k=2", "Patt k=3"
            ]
            writer.writerow(header)
            
            for sid, res in self.results_per_scene.items():
                base = res.get('baselines', {})
                ip = res.get('identity_persistence', {})
                ts = res.get('temporal_stability', {})

                curve = ts.get('pattern_curve', {})
                patt_1 = curve.get(1, curve.get('1', 0))
                patt_2 = curve.get(2, curve.get('2', 0))
                patt_3 = curve.get(3, curve.get('3', 0))
                
                row = [
                    sid,
                    f"{base.get('STQ', 0):.4f}", f"{base.get('AQ', 0):.4f}", f"{base.get('GQ', 0):.4f}",
                    f"{base.get('VPQ_0', 0):.4f}", f"{base.get('VPQ_5', 0):.4f}", f"{base.get('VPQ_15', 0):.4f}",
                    f"{base.get('VPQ_25', 0):.4f}", f"{base.get('VPQ_35', 0):.4f}", f"{base.get('VPQ_inf', 0):.4f}",
                    f"{ip.get('combined', 0):.4f}", f"{ip.get('prediction_axis', 0):.4f}", f"{ip.get('gt_axis', 0):.4f}",
                    f"{ip.get('ic_p', 0):.4f}", f"{ip.get('ic_g', 0):.4f}",
                    f"{ip.get('pred_spatial', 0):.4f}", f"{ip.get('temporal_bleed', 0):.4f}",
                    f"{ts.get('combined', 0):.4f}", f"{ts.get('cluster', 0):.4f}",
                    f"{patt_1:.4f}", 
                    f"{patt_2:.4f}", 
                    f"{patt_3:.4f}"
                ]
                writer.writerow(row)
                
        print(f"\n[*] Results saved to: {json_path} and {csv_path}")

    def print_summary(self):
        if not self.results_per_scene: return print("No results.")
        
        # Keep the table wide enough for all OGA and baseline columns.
        h = "-" * 235 
        print(
            h + 
            f"\n{'Scene ID':<16} | "
            f"{'STQ':<7} | "
            f"{'AQ':<7} | "
            f"{'GQ':<7} | "
            f"{'VPQ_0':<7} | "
            f"{'VPQ_5':<7} | "
            f"{'VPQ_15':<7} | "
            f"{'VPQ_25':<7} | "
            f"{'VPQ_35':<7} | "
            f"{'VPQ_inf':<7} | "
            f"{'IP (C)':<7} | "
            f"{'IP (P)':<7} | "
            f"{'IP (G)':<7} | "
            f"{'IC (P)':<7} | "
            f"{'IC (G)':<7} | "
            f"{'Spat.IP':<7} | "
            f"{'T.Bleed':<7} | "
            f"{'TS (C)':<7} | "
            f"{'Cluster':<8} | "
            f"{'Patt k=1':<8} | "
            f"{'Patt k=2':<8} | "
            f"{'Patt k=3':<8}"
            "\n" 
            + h
        )
        
        sz_agg = {s: defaultdict(list) for s in ['S', 'M', 'L', 'ALL']}
        for sid, res in self.results_per_scene.items():
            base = res['baselines']
            ip = res['identity_persistence']
            ts = res['temporal_stability']
            
            print(f"{sid:<16} | "
                  f"{base.get('STQ', 0):<7.4f} | "
                  f"{base.get('AQ', 0):<7.4f} | "
                  f"{base.get('GQ', 0):<7.4f} | "
                  f"{base.get('VPQ_0', 0):<7.4f} | "
                  f"{base.get('VPQ_5', 0):<7.4f} | "
                  f"{base.get('VPQ_15', 0):<7.4f} | "
                  f"{base.get('VPQ_25', 0):<7.4f} | "
                  f"{base.get('VPQ_35', 0):<7.4f} | "
                  f"{base.get('VPQ_inf', 0):<7.4f} | "
                  f"{ip['combined']:<7.4f} | "
                  f"{ip['prediction_axis']:<7.4f} | "
                  f"{ip['gt_axis']:<7.4f} | "
                  f"{ip.get('ic_p', 0):<7.4f} | "
                  f"{ip.get('ic_g', 0):<7.4f} | "
                  f"{ip['pred_spatial']:<7.4f} | "
                  f"{ip['temporal_bleed']:<7.4f} | "
                  f"{ts['combined']:<7.4f} | "
                  f"{ts['cluster']:<8.4f} | "
                  f"{ts['pattern_curve'].get(1,0):<8.4f} | "
                  f"{ts['pattern_curve'].get(2,0):<8.4f} | "
                  f"{ts['pattern_curve'].get(3,0):<8.4f}"
                 )
            
            for k, v in res['size_breakdown'].items():
                if v['count'] > 0:
                    for m, val in v.items():
                        if m == 'curve': 
                            for ck, cv in val.items(): 
                                sz_agg[k][f'c{ck}'].append(cv)
                        else: 
                            sz_agg[k][m].append(val)

        print(h + "\n" + " " * 90 + "=== NOVEL METRICS BY OBJECT SIZE ===\n" + h)
        for s in ['S', 'M', 'L']:
            d = {k: np.mean(v) for k, v in sz_agg[s].items()}
            count = int(np.sum(sz_agg[s].get('count', [0])))
            
            print(
                f"{s + ' (' + str(count) + ')':<16} | "
                f"{'N/A':<7} | {'N/A':<7} | {'N/A':<7} | " 
                f"{'N/A':<7} | {'N/A':<7} | {'N/A':<7} | " 
                f"{'N/A':<7} | {'N/A':<7} | {'N/A':<7} | " 
                f"{d.get('ip_comb',0):<7.4f} | "
                f"{d.get('ip_p',0):<7.4f} | "
                f"{d.get('ip_g',0):<7.4f} | "
                f"{d.get('ic_p',0):<7.4f} | "
                f"{d.get('ic_g',0):<7.4f} | "
                f"{d.get('ip_spat',0):<7.4f} | "
                f"{d.get('t_bleed',0):<7.4f} | "
                f"{d.get('ts_c',0):<7.4f} | "
                f"{d.get('cl_v',0):<8.4f} | "
                f"{d.get('c1',0):<8.4f} | "
                f"{d.get('c2',0):<8.4f} | "
                f"{d.get('c3',0):<8.4f}"
            )
        print(h)

if __name__ == "__main__":
    dataset = ScannetVOSDataset(gt_dir="/path/to/scannet/gt", pred_dir="./eval/")
    evaluator = VOSEvaluator(dataset, max_pattern_size=3)
    evaluator.evaluate(scene_names="scene0019_00")
    evaluator.print_summary()
