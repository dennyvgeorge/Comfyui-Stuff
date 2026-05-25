"""SeamBlender v2.2 - Overlap alpha-blending. Uses batch_index math only."""
import os, glob, numpy as np, cv2, shutil

class SeamBlender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "trigger": ("BOOLEAN", {"forceInput": True}),
            "batch_index": ("INT", {"forceInput": True}),
            "output_folder": ("STRING", {"default": "J:/pinokio/api/comfy.git/app/output/cinema_raw"}),
            "backup_folder": ("STRING", {"default": "J:/pinokio/api/comfy.git/app/output/cinema_raw_overlap"}),
            "batch_size": ("INT", {"default": 25, "min": 2, "max": 200}),
            "overlap": ("INT", {"default": 8, "min": 0, "max": 100}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "process"
    CATEGORY = "Batch Video Processor"
    OUTPUT_NODE = True

    def process(self, trigger, batch_index, output_folder, backup_folder, batch_size, overlap):
        os.makedirs(backup_folder, exist_ok=True)
        frame_pattern = "frame.%06d.exr"
        step_size = batch_size - overlap
        if batch_index > 0:
            overlap_start = batch_index * step_size + 1
            saved = 0
            for i in range(overlap):
                frame_num = overlap_start + i
                src = os.path.join(output_folder, frame_pattern % frame_num)
                dst = os.path.join(backup_folder, f"batch{batch_index-1:02d}_" + frame_pattern % frame_num)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    saved += 1
                    print(f"[SeamBlender] Backed up frame {frame_num}")
            print(f"[SeamBlender] Batch {batch_index}: backed up {saved} frames ({overlap_start}-{overlap_start+overlap-1})")
        else:
            print(f"[SeamBlender] Batch 0: baseline")
        if not trigger:
            return (f"batch {batch_index} done",)
        print(f"[SeamBlender] === FINAL BLEND PASS ===")
        backup_files = sorted(glob.glob(os.path.join(backup_folder, "batch*_frame.*.exr")))
        if not backup_files:
            return ("no backups found",)
        blended_count = 0
        for bf in backup_files:
            basename = os.path.basename(bf)
            frame_filename = basename.split("_", 1)[1]
            frame_num = int(frame_filename.replace("frame.", "").replace(".exr", ""))
            current_path = os.path.join(output_folder, frame_filename)
            if not os.path.exists(current_path):
                continue
            old_img = cv2.imread(bf, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            new_img = cv2.imread(current_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if old_img is None or new_img is None or old_img.shape != new_img.shape:
                continue
            batch_num = int(basename.split("_")[0].replace("batch", ""))
            overlap_zone_start = (batch_num + 1) * step_size + 1
            local_pos = frame_num - overlap_zone_start
            alpha_old = 1.0 - (local_pos / max(overlap - 1, 1))
            alpha_old = float(np.clip(alpha_old, 0.0, 1.0))
            blended = cv2.addWeighted(old_img, alpha_old, new_img, 1.0 - alpha_old, 0)
            cv2.imwrite(current_path, blended)
            blended_count += 1
            print(f"  Blended {frame_filename} (pos {local_pos}/{overlap}): old={alpha_old:.0%} new={1-alpha_old:.0%}")
        print(f"[SeamBlender] === DONE: {blended_count} frames blended ===")
        return (f"Done: {blended_count} blended",)

NODE_CLASS_MAPPINGS = {"SeamBlender": SeamBlender}
NODE_DISPLAY_NAME_MAPPINGS = {"SeamBlender": "Seam Blender"}
