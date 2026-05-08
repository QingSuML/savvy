from collections import deque

import cv2
import numpy as np


class AdaptiveGridManager:
    def __init__(self, base_points=32, min_points=16, history_size=3):
        self.base_points = base_points
        self.min_points = min_points
        # Smooth sudden edge-density spikes across nearby frames.
        self.density_history = deque(maxlen=history_size)
        self.current_grid = base_points

    def get_grid_size(self, frame_img):
        """
        Calculates edge density and returns a temporally smoothed grid size.
        Expects a numpy array image (RGB or Grayscale).
        """
        # Fast edge detection.
        if len(frame_img.shape) == 3:
            gray = cv2.cvtColor(frame_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_img
            
        edges = cv2.Canny(gray, 100, 200)
        raw_density = np.sum(edges > 0) / edges.size
        
        # Smooth density over a short history.
        self.density_history.append(raw_density)
        smoothed_density = sum(self.density_history) / len(self.density_history)
        
        # Determine target grid size from the smoothed density.
        if smoothed_density > 0.12:
            target_grid = self.min_points
        elif smoothed_density > 0.08:
            target_grid = int(self.base_points * 0.75)
        else:
            target_grid = self.base_points
            
        if target_grid != self.current_grid:
            print(f"  [~] Clutter shifted (Density: {smoothed_density:.3f}).",
                  f" Adapting SAM grid to {target_grid}x{target_grid}.")
            self.current_grid = target_grid
            
        return target_grid

class MaskFeedbackGridManager:
    def __init__(self, base_points=32, min_points=8, max_points=32):
        self.current_grid = base_points
        self.min_points = min_points
        self.max_points = max_points
        
        # Mask-count thresholds for adapting the next SAM1 prompt grid.
        self.too_many_masks = 50
        self.too_few_masks = 20

    def update_and_get_grid(self, num_masks_generated):
        """
        Adjusts the grid size for the NEXT run based on the current run's mask count.
        """
        old_grid = self.current_grid
        
        if num_masks_generated > self.too_many_masks:
            self.current_grid = max(self.min_points, self.current_grid - 8)
            if self.current_grid != old_grid:
                print(f"  [~] Mask explosion ({num_masks_generated} masks). Reducing SAM grid to {self.current_grid}x{self.current_grid}.")
                
        elif num_masks_generated < self.too_few_masks:
            self.current_grid = min(self.max_points, self.current_grid + 8)
            
        return self.current_grid

class SmoothedMaskFeedbackGridManager:
    def __init__(self, base_points=32, min_points=8, max_points=32, history_size=3):
        self.current_grid = base_points
        self.min_points = min_points
        self.max_points = max_points
        
        # Smooth mask-count feedback across nearby segmenter calls.
        self.mask_history = deque(maxlen=history_size)
        
        # Mask-count thresholds for adapting the next SAM1 prompt grid.
        self.too_many_masks = 50  
        self.too_few_masks = 20   

    def update_and_get_grid(self, num_masks_generated):
        self.mask_history.append(num_masks_generated)
        
        # Wait for a full history buffer before adapting.
        if len(self.mask_history) < self.mask_history.maxlen:
            return self.current_grid
            
        avg_masks = sum(self.mask_history) / len(self.mask_history)
        old_grid = self.current_grid
        
        if avg_masks > self.too_many_masks:
            self.current_grid = max(self.min_points, self.current_grid - 8)
        elif avg_masks < self.too_few_masks:
            self.current_grid = min(self.max_points, self.current_grid + 8)
            
        if self.current_grid != old_grid:
            print(f"  [~] Smoothed mask count ({avg_masks:.1f}). Adjusted grid to {self.current_grid}x{self.current_grid}.")
            # Clear history to add a short cooldown after changing grid size.
            self.mask_history.clear()
            
        return self.current_grid
