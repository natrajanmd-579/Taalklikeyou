import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "doubi-killer/TalkLikeYou"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    # Motion models
    "checkpoints/audio_encoder.pth": os.path.join(BASE_DIR, "checkpoints", "motion", "audio_encoder.pth"),
    "checkpoints/motion/lip_motion.pt": os.path.join(BASE_DIR, "checkpoints", "motion", "lip_motion.pt"),
    "checkpoints/motion/pose_motion.pt": os.path.join(BASE_DIR, "checkpoints", "motion", "pose_motion.pt"),

    # Runtime aux models
    "checkpoints/ditto/ditto_pytorch/aux_models/2d106det.onnx": os.path.join(BASE_DIR, "checkpoints", "runtime", "aux_models", "2d106det.onnx"),
    "checkpoints/ditto/ditto_pytorch/aux_models/det_10g.onnx": os.path.join(BASE_DIR, "checkpoints", "runtime", "aux_models", "det_10g.onnx"),
    "checkpoints/ditto/ditto_pytorch/aux_models/face_landmarker.task": os.path.join(BASE_DIR, "checkpoints", "runtime", "aux_models", "face_landmarker.task"),
    "checkpoints/ditto/ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx": os.path.join(BASE_DIR, "checkpoints", "runtime", "aux_models", "hubert_streaming_fix_kv.onnx"),
    "checkpoints/ditto/ditto_pytorch/aux_models/landmark203.onnx": os.path.join(BASE_DIR, "checkpoints", "runtime", "aux_models", "landmark203.onnx"),

    # Runtime models
    "checkpoints/ditto/ditto_pytorch/models/appearance_extractor.pth": os.path.join(BASE_DIR, "checkpoints", "runtime", "models", "appearance_extractor.pth"),
    "checkpoints/ditto/ditto_pytorch/models/decoder.pth": os.path.join(BASE_DIR, "checkpoints", "runtime", "models", "decoder.pth"),
    "checkpoints/ditto/ditto_pytorch/models/lmdm_v0.4_hubert.pth": os.path.join(BASE_DIR, "checkpoints", "runtime", "models", "lmdm_v0.4_hubert.pth"),
    "checkpoints/ditto/ditto_pytorch/models/motion_extractor.pth": os.path.join(BASE_DIR, "checkpoints", "runtime", "models", "motion_extractor.pth"),
    "checkpoints/ditto/ditto_pytorch/models/stitch_network.pth": os.path.join(BASE_DIR, "checkpoints", "runtime", "models", "stitch_network.pth"),
    "checkpoints/ditto/ditto_pytorch/models/warp_network.pth": os.path.join(BASE_DIR, "checkpoints", "runtime", "models", "warp_network.pth"),
}

def main():
    print(f"Downloading TalkLikeYou checkpoints ({len(FILES)} files, ~2.3 GB total)...")
    for repo_file, target_path in FILES.items():
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"[SKIP] Already exists: {os.path.relpath(target_path, BASE_DIR)}")
            continue

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        print(f"[DOWNLOADING] {repo_file} -> {os.path.relpath(target_path, BASE_DIR)} ...")
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=repo_file)
        shutil.copy2(downloaded, target_path)
        print(f"[OK] Saved {os.path.relpath(target_path, BASE_DIR)}")

    print("\nAll checkpoints are ready!")

if __name__ == "__main__":
    main()
