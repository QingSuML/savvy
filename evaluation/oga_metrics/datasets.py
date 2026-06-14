import glob
import os

import numpy as np


class ScannetVOSDataset:
    def __init__(self, gt_dir, pred_dir, pred_scene_map=None, gt_subdir="instance"):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.pred_scene_map = pred_scene_map or {}
        self.gt_subdir = gt_subdir
        
    def get_scene_ids(self):
        return sorted([
            d for d in os.listdir(self.gt_dir)
            if os.path.isdir(os.path.join(self.gt_dir, d))
            and os.path.isdir(os.path.join(self.gt_dir, d, self.gt_subdir))
        ])
        
    def load_scene_frames(self, scene_id):
        from PIL import Image

        gt_path = os.path.join(self.gt_dir, scene_id, self.gt_subdir)
        pred_scene_dir = self.pred_scene_map.get(scene_id, scene_id)
        pr_path = os.path.join(self.pred_dir, pred_scene_dir)
        
        if not all(os.path.exists(p) for p in [gt_path, pr_path]): 
            return [], []
            
        g_files = sorted(glob.glob(os.path.join(gt_path, "*.png")), \
                         key=lambda x: int(os.path.basename(x).split('.')[0]))
        p_map = {int(os.path.basename(p).split('.')[0]): \
                 p for p in glob.glob(os.path.join(pr_path, "*.png"))}
        
        total_frames = len(g_files)
        
        if total_frames <= 300:
            adaptive_interval = 1
            cutoff = total_frames
        elif total_frames <= 1000:
            adaptive_interval = 2
            cutoff = total_frames
        elif total_frames <= 1500:
            adaptive_interval = 3
            cutoff = total_frames
        else:
            adaptive_interval = 3
            cutoff = 1500
            
        g_files = g_files[:cutoff]
        print(f"[{scene_id}] Frames: {total_frames} -> "
              f"Capped at {cutoff}, Interval: {adaptive_interval}")

        gl, pl = [], []
        for gp in g_files:
            idx = int(os.path.basename(gp).split('.')[0])
            if idx % adaptive_interval == 0 and idx in p_map:
                gi = Image.open(gp)
                pi = Image.open(p_map[idx])
                
                if pi.size != gi.size: 
                    resample_filter = getattr(Image, 'Resampling', Image).NEAREST
                    pi = pi.resize(gi.size, resample=resample_filter)
                    
                gl.append(np.array(gi))
                pl.append(np.array(pi))
                
        return pl, gl

class HM3DVOSDataset:
    def __init__(self, gt_dir, pred_dir, pred_scene_map=None):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.pred_scene_map = pred_scene_map or {}

    def get_scene_ids(self):
        return sorted([
            d for d in os.listdir(self.gt_dir)
            if os.path.isdir(os.path.join(self.gt_dir, d))
            and (
                os.path.isdir(os.path.join(self.gt_dir, d, "jpg"))
                or os.path.isdir(os.path.join(self.gt_dir, d, "rgb"))
            )
            and os.path.isdir(os.path.join(self.gt_dir, d, "semantic"))
        ])

    def load_scene_frames(self, scene_id):
        from PIL import Image

        gt_path = os.path.join(self.gt_dir, scene_id, "semantic")
        pred_scene_dir = self.pred_scene_map.get(scene_id, scene_id)
        pr_path = os.path.join(self.pred_dir, pred_scene_dir)

        if not all(os.path.exists(p) for p in [gt_path, pr_path]):
            return [], []

        g_files = sorted(
            glob.glob(os.path.join(gt_path, "*.npy")),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        p_map = {
            int(os.path.basename(p).split('.')[0]): p
            for p in glob.glob(os.path.join(pr_path, "*.png"))
        }

        total_frames = len(g_files)

        if total_frames <= 300:
            adaptive_interval = 1
            cutoff = total_frames
        elif total_frames <= 1000:
            adaptive_interval = 2
            cutoff = total_frames
        elif total_frames <= 1500:
            adaptive_interval = 3
            cutoff = total_frames
        else:
            adaptive_interval = 3
            cutoff = 1500

        g_files = g_files[:cutoff]
        print(f"[{scene_id}] Frames: {total_frames} -> "
              f"Capped at {cutoff}, Interval: {adaptive_interval}")

        gl, pl = [], []
        for gp in g_files:
            idx = int(os.path.basename(gp).split('.')[0])
            if idx % adaptive_interval == 0 and idx in p_map:
                gi = np.load(gp)
                pi = np.array(Image.open(p_map[idx]))

                if pi.shape[:2] != gi.shape[:2]:
                    pil_pi = Image.fromarray(pi)
                    resample = getattr(Image, "Resampling", Image).NEAREST
                    pil_pi = pil_pi.resize((gi.shape[1], gi.shape[0]), resample=resample)
                    pi = np.array(pil_pi)

                gl.append(gi.astype(np.int64))
                pl.append(pi.astype(np.int64))

        return pl, gl
