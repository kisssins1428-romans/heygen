import sys, traceback, os, subprocess, base64, tempfile

print("🚀 handler.py starting...", flush=True)

try:
    import runpod
    print("✅ All imports successful", flush=True)
except Exception as e:
    print(f"❌ Import error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

LATENTSYNC_DIR = "/workspace/LatentSync"
CONFIG_PATH = os.path.join(LATENTSYNC_DIR, "configs/unet/stage2_512.yaml")
CKPT_PATH = os.path.join(LATENTSYNC_DIR, "checkpoints/latentsync_unet.pt")
WHISPER_DIR = os.path.join(LATENTSYNC_DIR, "checkpoints/whisper")

def fix_whisper_paths():
    """HuggingFace가 whisper/whisper/tiny.pt 로 저장하는 경우 심볼릭 링크로 수정"""
    for model in ["tiny.pt", "small.pt"]:
        target = os.path.join(WHISPER_DIR, model)
        nested = os.path.join(WHISPER_DIR, "whisper", model)
        if not os.path.exists(target) and os.path.exists(nested):
            os.symlink(nested, target)
            print(f"🔗 Symlink created: {target} -> {nested}", flush=True)
        elif os.path.exists(target):
            print(f"✅ Whisper {model} found at {target}", flush=True)
        else:
            print(f"⚠️ Whisper {model} not found anywhere", flush=True)

fix_whisper_paths()

print(f"🔍 CWD: {os.getcwd()}", flush=True)
print(f"🔍 CONFIG exists: {os.path.exists(CONFIG_PATH)}", flush=True)
print(f"🔍 CKPT exists: {os.path.exists(CKPT_PATH)}", flush=True)

def run_latentsync(job):
    input_data = job.get("input", {})
    video_input = input_data.get("video_url") or input_data.get("video_base64")
    audio_input = input_data.get("audio_url") or input_data.get("audio_base64")
    inference_steps = input_data.get("inference_steps", 20)
    guidance_scale = input_data.get("guidance_scale", 1.5)

    if not video_input or not audio_input:
        return {"error": "video_url and audio_url are required"}

    if not os.path.exists(CONFIG_PATH):
        result = subprocess.run(["find", LATENTSYNC_DIR, "-name", "*.yaml"], capture_output=True, text=True)
        return {"error": f"Config not found: {CONFIG_PATH}. Available: {result.stdout}"}

    if not os.path.exists(CKPT_PATH):
        result = subprocess.run(["find", os.path.join(LATENTSYNC_DIR, "checkpoints"), "-type", "f"], capture_output=True, text=True)
        return {"error": f"Checkpoint not found: {CKPT_PATH}. Available: {result.stdout}"}

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input_video.mp4")
        audio_path = os.path.join(tmpdir, "input_audio.wav")
        audio_wav_path = os.path.join(tmpdir, "audio_16k.wav")
        output_path = os.path.join(tmpdir, "output.mp4")

        # 비디오 다운로드
        if str(video_input).startswith("http"):
            r = subprocess.run(["wget", "-q", "--timeout=60", "-O", video_path, video_input], capture_output=True, text=True)
            if r.returncode != 0:
                return {"error": f"Video download failed: {r.stderr}"}
        else:
            with open(video_path, "wb") as f:
                f.write(base64.b64decode(video_input + "=="))

        # 오디오 다운로드
        if str(audio_input).startswith("http"):
            r = subprocess.run(["wget", "-q", "--timeout=60", "-O", audio_path, audio_input], capture_output=True, text=True)
            if r.returncode != 0:
                return {"error": f"Audio download failed: {r.stderr}"}
        else:
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_input + "=="))

        # 오디오 16kHz mono WAV 변환
        r = subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ar", "16000", "-ac", "1",
            audio_wav_path, "-y", "-loglevel", "error"
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return {"error": f"Audio conversion failed: {r.stderr}"}

        # LatentSync inference 실행
        r = subprocess.run([
            "python", "-m", "scripts.inference",
            "--unet_config_path", CONFIG_PATH,
            "--inference_ckpt_path", CKPT_PATH,
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--enable_deepcache",
            "--video_path", video_path,
            "--audio_path", audio_wav_path,
            "--video_out_path", output_path,
            "--temp_dir", tmpdir,
        ], capture_output=True, text=True, cwd=LATENTSYNC_DIR)

        if r.returncode != 0:
            return {
                "error": "Inference failed",
                "stdout": r.stdout[-3000:],
                "stderr": r.stderr[-3000:]
            }

        if not os.path.exists(output_path):
            return {"error": f"Output not generated. stdout: {r.stdout[-1000:]}"}

        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "output_video_base64": output_b64,
            "message": "✅ Lip sync completed!"
        }

print("🎬 RunPod serverless worker ready!", flush=True)
runpod.serverless.start({"handler": run_latentsync})
