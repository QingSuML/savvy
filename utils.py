
import numpy as np

def show_masks_fast(masks_dict, ax, show_id=True, img=None, seed=0):
    """
    Composite all object masks into a single RGBA image and render in one imshow call.
    Removes all axis margins and ensures the image fills the frame.
    """
    if not masks_dict:
        return

    # Use the first mask to infer the render size.
    first_mask = next(iter(masks_dict.values()))
    h, w = first_mask.shape[-2], first_mask.shape[-1]

    # RGBA compositing canvas.
    composite = np.zeros((h, w, 4), dtype=np.float32)

    for obj_id, mask in masks_dict.items():
        binary = mask.reshape(h, w).astype(bool)
        if not binary.any():
            continue

        # Deterministic stable color per object ID.
        rng = np.random.default_rng(seed=int(obj_id)+ seed)
        color = np.concatenate([rng.random(3), [0.7]]).astype(np.float32)

        # Alpha-composite this mask over the canvas.
        alpha_mask = binary[:, :, None]  # [H, W, 1]
        src_alpha = color[3]
        dst_alpha = composite[:, :, 3:4]

        out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
        safe_out_alpha = np.where(out_alpha > 0, out_alpha, 1.0)
        blended_rgb = (
            color[:3] * src_alpha
            + composite[:, :, :3] * dst_alpha * (1 - src_alpha)
        ) / safe_out_alpha

        composite[:, :, :3] = np.where(alpha_mask, blended_rgb, composite[:, :, :3])
        composite[:, :, 3:4] = np.where(alpha_mask, out_alpha, dst_alpha)

        # Annotate the object centroid.
        if show_id:
            ys, xs = np.where(binary)
            cx, cy = xs.mean(), ys.mean()
            ax.text(cx, cy, str(obj_id), color="white", fontsize=5,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc=color[:3].tolist(), alpha=0.6),
                    zorder=3)

    if img is not None:
        ax.imshow(img)
    ax.imshow(composite)

    ax.axis('off')
    ax.set_axis_off()
    
    # Match plot limits to the image size.
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
