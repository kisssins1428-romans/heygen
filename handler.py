import sys, traceback, os, subprocess, base64, tempfile, shutil, binascii, urllib.request, urllib.parse, http.client, json

print("🚀 handler.py starting...", flush=True)

try:
    import runpod
    print("✅ runpod imported", flush=True)
except Exception as e:
    print(f"❌ Import error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

LATENTSYNC_DIR = os.environ.get("LATENTSYNC_DIR", "/workspace/LatentSync")
LATENTSYNC_PYTHON = os.environ.get("LATENTSYNC_PYTHON", sys.executable)
CONFIG_PATH = os.environ.get(
    "LATENTSYNC_CONFIG_PATH",
    os.path.join(LATENTSYNC_DIR, "configs/unet/stage2_512.yaml"),
)
CKPT_PATH = os.environ.get(
    "LATENTSYNC_CKPT_PATH",
    os.path.join(LATENTSYNC_DIR, "checkpoints/latentsync_unet.pt"),
)
WHISPER_TINY_PATH = os.environ.get(
    "LATENTSYNC_WHISPER_TINY_PATH",
    os.path.join(LATENTSYNC_DIR, "checkpoints/whisper/tiny.pt"),
)
WHISPER_SMALL_PATH = os.environ.get(
    "LATENTSYNC_WHISPER_SMALL_PATH",
    os.path.join(LATENTSYNC_DIR, "checkpoints/whisper/small.pt"),
)
INFERENCE_SCRIPT_PATH = os.path.join(LATENTSYNC_DIR, "scripts/inference.py")
SCHEDULER_CONFIG_PATH = os.path.join(LATENTSYNC_DIR, "configs/scheduler_config.json")


def get_int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


DOWNLOAD_TIMEOUT_SECONDS = get_int_env("DOWNLOAD_TIMEOUT_SECONDS", 60)
FFMPEG_TIMEOUT_SECONDS = get_int_env("FFMPEG_TIMEOUT_SECONDS", 300)
LATENTSYNC_TIMEOUT_SECONDS = get_int_env("LATENTSYNC_TIMEOUT_SECONDS", 1800)
MAX_BASE64_OUTPUT_MB = get_int_env("MAX_BASE64_OUTPUT_MB", 7)
PYTHON_PROBE_TIMEOUT_SECONDS = get_int_env("PYTHON_PROBE_TIMEOUT_SECONDS", 20)

print(f"🔍 LATENTSYNC_DIR exists : {os.path.exists(LATENTSYNC_DIR)}", flush=True)
print(f"🔍 CONFIG exists         : {os.path.exists(CONFIG_PATH)}", flush=True)
print(f"🔍 CKPT exists           : {os.path.exists(CKPT_PATH)}", flush=True)
print(f"🔍 WHISPER tiny exists   : {os.path.exists(WHISPER_TINY_PATH)}", flush=True)
print(f"🔍 WHISPER small exists  : {os.path.exists(WHISPER_SMALL_PATH)}", flush=True)
print(f"🔍 INFERENCE exists      : {os.path.exists(INFERENCE_SCRIPT_PATH)}", flush=True)
print(f"🔍 SCHEDULER exists      : {os.path.exists(SCHEDULER_CONFIG_PATH)}", flush=True)
print(f"🐍 Python executable     : {sys.executable}", flush=True)
print(f"🐍 LatentSync Python     : {LATENTSYNC_PYTHON}", flush=True)
print(f"🐍 Python version        : {sys.version.split()[0]}", flush=True)
print(f"🎞️ ffmpeg path           : {shutil.which('ffmpeg')}", flush=True)


def executable_exists(executable):
    return os.path.exists(executable) or shutil.which(executable) is not None


def python_runtime_probe(executable):
    if not executable_exists(executable):
        return {"error": f"Executable not found: {executable}"}

    script = """
import json, sys
info = {"executable": sys.executable, "version": sys.version}
try:
    import torch
    info.update({
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    })
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_capability"] = torch.cuda.get_device_capability(0)
except Exception as e:
    info["torch_error"] = repr(e)
print(json.dumps(info))
"""
    try:
        result = subprocess.run(
            [executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=PYTHON_PROBE_TIMEOUT_SECONDS,
        )
    except Exception as e:
        return {"error": repr(e)}

    if result.returncode != 0:
        return {
            "error": "Python probe failed",
            "returncode": result.returncode,
            "stdout": tail_text(result.stdout, 1000),
            "stderr": tail_text(result.stderr, 1000),
        }

    try:
        return json.loads(result.stdout)
    except Exception as e:
        return {
            "error": f"Could not parse Python probe output: {e}",
            "stdout": tail_text(result.stdout, 1000),
        }


def runtime_diagnostics():
    diagnostics = {
        "python_executable": sys.executable,
        "latentsync_python": LATENTSYNC_PYTHON,
        "python_version": sys.version,
        "latentsync_dir": LATENTSYNC_DIR,
        "config_path": CONFIG_PATH,
        "ckpt_path": CKPT_PATH,
        "whisper_tiny_path": WHISPER_TINY_PATH,
        "whisper_small_path": WHISPER_SMALL_PATH,
        "inference_script_path": INFERENCE_SCRIPT_PATH,
        "scheduler_config_path": SCHEDULER_CONFIG_PATH,
        "ffmpeg_path": shutil.which("ffmpeg"),
        "download_timeout_seconds": DOWNLOAD_TIMEOUT_SECONDS,
        "ffmpeg_timeout_seconds": FFMPEG_TIMEOUT_SECONDS,
        "latentsync_timeout_seconds": LATENTSYNC_TIMEOUT_SECONDS,
        "max_base64_output_mb": MAX_BASE64_OUTPUT_MB,
        "python_probe_timeout_seconds": PYTHON_PROBE_TIMEOUT_SECONDS,
    }

    try:
        import torch

        diagnostics.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            diagnostics["cuda_device_name"] = torch.cuda.get_device_name(0)
            diagnostics["cuda_device_capability"] = torch.cuda.get_device_capability(0)
    except Exception as e:
        diagnostics["torch_error"] = repr(e)

    if LATENTSYNC_PYTHON != sys.executable:
        diagnostics["latentsync_python_probe"] = python_runtime_probe(LATENTSYNC_PYTHON)

    return diagnostics


def tail_text(value, limit=3000):
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)[-limit:]


def required_whisper_checkpoint():
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.load(CONFIG_PATH)
        cross_attention_dim = int(config.model.cross_attention_dim)
    except Exception as e:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if "cross_attention_dim:" in line:
                        cross_attention_dim = int(line.split(":", 1)[1].split("#", 1)[0].strip())
                        break
                else:
                    raise ValueError("cross_attention_dim not found")
        except Exception:
            return WHISPER_TINY_PATH, {
                "warning": f"Could not read cross_attention_dim from config; defaulting to tiny.pt: {e}",
            }

    if cross_attention_dim == 384:
        return WHISPER_TINY_PATH, {"cross_attention_dim": cross_attention_dim}
    if cross_attention_dim == 768:
        return WHISPER_SMALL_PATH, {"cross_attention_dim": cross_attention_dim}

    return None, {
        "error": f"Unsupported cross_attention_dim: {cross_attention_dim}",
        "cross_attention_dim": cross_attention_dim,
    }


def find_files(path, args):
    if not os.path.exists(path):
        return ""

    try:
        return subprocess.run(
            ["find", path, *args],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout[-4000:]
    except Exception as e:
        return f"find failed: {e}"


def preflight_error():
    if not os.path.isdir(LATENTSYNC_DIR):
        return {
            "error": f"LatentSync directory not found: {LATENTSYNC_DIR}",
            "workspace_files": find_files("/workspace", ["-maxdepth", "3", "-type", "d"]),
            "diagnostics": runtime_diagnostics(),
        }

    if not executable_exists(LATENTSYNC_PYTHON):
        return {
            "error": f"LatentSync Python executable not found: {LATENTSYNC_PYTHON}",
            "diagnostics": runtime_diagnostics(),
        }

    whisper_path, whisper_info = required_whisper_checkpoint()
    if whisper_path is None:
        return {
            "error": whisper_info["error"],
            "diagnostics": runtime_diagnostics(),
        }

    required_files = [
        ("Config", CONFIG_PATH, [LATENTSYNC_DIR, ["-name", "*.yaml"]]),
        ("Checkpoint", CKPT_PATH, [os.path.join(LATENTSYNC_DIR, "checkpoints"), ["-type", "f"]]),
        ("Whisper checkpoint", whisper_path, [os.path.join(LATENTSYNC_DIR, "checkpoints"), ["-type", "f"]]),
        ("Inference script", INFERENCE_SCRIPT_PATH, [LATENTSYNC_DIR, ["-path", "*/scripts/*", "-type", "f"]]),
        ("Scheduler config", SCHEDULER_CONFIG_PATH, [os.path.join(LATENTSYNC_DIR, "configs"), ["-type", "f"]]),
    ]

    for label, path, find_args in required_files:
        if not os.path.exists(path):
            return {
                "error": f"{label} not found: {path}",
                "available_files": find_files(find_args[0], find_args[1]),
                "whisper_info": whisper_info,
                "diagnostics": runtime_diagnostics(),
            }

    if not shutil.which("ffmpeg"):
        return {
            "error": "ffmpeg not found in PATH",
            "diagnostics": runtime_diagnostics(),
        }

    return None


def decode_base64_payload(payload):
    if not isinstance(payload, str):
        raise ValueError("base64 input must be a string")

    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    payload = "".join(payload.split())
    payload += "=" * ((4 - len(payload) % 4) % 4)

    decoder = base64.urlsafe_b64decode if any(c in payload for c in "-_") else base64.b64decode

    try:
        return decoder(payload)
    except binascii.Error:
        return base64.urlsafe_b64decode(payload)


def write_input_file(input_value, output_path, label):
    if str(input_value).startswith(("http://", "https://")):
        try:
            request = urllib.request.Request(
                input_value,
                headers={"User-Agent": "runpod-latentsync/1.0"},
            )
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                with open(output_path, "wb") as f:
                    shutil.copyfileobj(response, f)
        except Exception as e:
            return {"error": f"{label} download failed: {e}"}
    else:
        try:
            with open(output_path, "wb") as f:
                f.write(decode_base64_payload(input_value))
        except Exception as e:
            return {"error": f"{label} base64 decode failed: {e}"}

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return {"error": f"{label} file was not written or is empty"}

    return None


def upload_file_to_presigned_url(file_path, upload_url):
    parsed = urllib.parse.urlsplit(upload_url)
    if parsed.scheme not in ("http", "https"):
        return {"error": f"Unsupported upload URL scheme: {parsed.scheme}"}

    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"

    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_cls(parsed.netloc, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    size = os.path.getsize(file_path)

    try:
        with open(file_path, "rb") as f:
            connection.request(
                "PUT",
                path,
                body=f,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(size),
                },
            )
            response = connection.getresponse()
            response_body = response.read(1000).decode("utf-8", errors="replace")

        if response.status not in (200, 201, 204):
            return {
                "error": f"Output upload failed with HTTP {response.status}",
                "response": response_body,
            }
    except Exception as e:
        return {"error": f"Output upload failed: {e}"}
    finally:
        connection.close()

    return None


def run_latentsync(job):
    input_data    = job.get("input", {})
    video_input   = input_data.get("video_url")  or input_data.get("video_base64")
    audio_input   = input_data.get("audio_url")  or input_data.get("audio_base64")
    inference_steps = input_data.get("inference_steps", 20)
    guidance_scale  = input_data.get("guidance_scale", 1.5)
    output_upload_url = input_data.get("output_upload_url")
    output_url = input_data.get("output_url") or input_data.get("output_download_url")

    if not video_input or not audio_input:
        return {"error": "video_url and audio_url are required"}

    error = preflight_error()
    if error:
        return error

    # tmpdir를 LatentSync 외부에 생성 (내부 충돌 방지)
    with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
        video_path    = os.path.join(tmpdir, "input_video.mp4")
        audio_path    = os.path.join(tmpdir, "input_audio.wav")
        audio_wav_path= os.path.join(tmpdir, "audio_16k.wav")
        output_path   = os.path.join(tmpdir, "output.mp4")
        temp_subdir   = os.path.join(tmpdir, "latentsync_tmp")
        os.makedirs(temp_subdir, exist_ok=True)

        # 비디오 다운로드
        error = write_input_file(video_input, video_path, "Video")
        if error:
            return error

        # 오디오 다운로드
        error = write_input_file(audio_input, audio_path, "Audio")
        if error:
            return error

        # 오디오 → 16kHz mono WAV 변환
        try:
            r = subprocess.run([
                "ffmpeg", "-i", audio_path,
                "-ar", "16000", "-ac", "1",
                audio_wav_path, "-y", "-loglevel", "error"
            ], capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as e:
            return {
                "error": f"Audio conversion timed out after {FFMPEG_TIMEOUT_SECONDS} seconds",
                "stdout": tail_text(e.stdout, 1000),
                "stderr": tail_text(e.stderr, 1000),
            }
        if r.returncode != 0:
            return {"error": f"Audio conversion failed: {r.stderr}"}

        # inference 실행 (cwd=LATENTSYNC_DIR 필수)
        cmd = [
            LATENTSYNC_PYTHON, "-m", "scripts.inference",
            "--unet_config_path",    CONFIG_PATH,
            "--inference_ckpt_path", CKPT_PATH,
            "--inference_steps",     str(inference_steps),
            "--guidance_scale",      str(guidance_scale),
            "--enable_deepcache",
            "--video_path",          video_path,
            "--audio_path",          audio_wav_path,
            "--video_out_path",      output_path,
            "--temp_dir",            temp_subdir,
        ]
        print(f"🎬 Running: {' '.join(cmd)}", flush=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = LATENTSYNC_DIR + os.pathsep + env.get("PYTHONPATH", "")
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=LATENTSYNC_DIR,
                env=env,
                timeout=LATENTSYNC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "error": f"Inference timed out after {LATENTSYNC_TIMEOUT_SECONDS} seconds",
                "stdout": tail_text(e.stdout),
                "stderr": tail_text(e.stderr),
                "diagnostics": runtime_diagnostics(),
            }

        if r.returncode != 0:
            return {
                "error": "Inference failed",
                "returncode": r.returncode,
                "stdout": tail_text(r.stdout),
                "stderr": tail_text(r.stderr),
                "diagnostics": runtime_diagnostics(),
            }

        if not os.path.exists(output_path):
            return {
                "error": "Output file not generated",
                "stdout": tail_text(r.stdout, 1000),
                "stderr": tail_text(r.stderr, 1000),
                "diagnostics": runtime_diagnostics(),
            }

        output_size = os.path.getsize(output_path)
        if output_upload_url:
            error = upload_file_to_presigned_url(output_path, output_upload_url)
            if error:
                error["diagnostics"] = runtime_diagnostics()
                return error

            return {
                "output_video_url": output_url or output_upload_url.split("?", 1)[0],
                "output_size_bytes": output_size,
                "message": "✅ Lip sync completed!",
            }

        max_base64_bytes = MAX_BASE64_OUTPUT_MB * 1024 * 1024
        if output_size > max_base64_bytes:
            return {
                "error": "Output file is too large to return as base64",
                "output_size_bytes": output_size,
                "max_base64_output_bytes": max_base64_bytes,
                "hint": "Provide input.output_upload_url as a presigned PUT URL and optional input.output_url for the returned download URL.",
                "diagnostics": runtime_diagnostics(),
            }

        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "output_video_base64": output_b64,
            "output_size_bytes": output_size,
            "message": "✅ Lip sync completed!",
        }

print("🎬 RunPod serverless worker ready!", flush=True)
runpod.serverless.start({"handler": run_latentsync})
