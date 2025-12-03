import os
import io
import base64
import torch
import torchaudio
import runpod

# Import your VibeVoice inference utilities (adjust path if needed)
from vibevoice.inference import load_model, tts_generate  # must exist

# --- Env ---
MODEL_NAME = os.getenv("MODEL_NAME", "vibevoice/VibeVoice-7B")
DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "de")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load once at startup ---
print(f"[VibeVoice] Loading model '{MODEL_NAME}' on {device}...")
model, processor = load_model(model_name=MODEL_NAME, hf_token=HF_TOKEN, device=device)
if hasattr(model, "eval"):
    model.eval()

def synthesize_speech(text: str, language: str) -> str:
    """
    Generate speech audio from text and return base64-encoded WAV.
    Expects tts_generate to return FloatTensor shaped [channels, samples] at 16 kHz.
    """
    audio = tts_generate(model=model, processor=processor, text=text, language=language, device=device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)                     # [samples] -> [1, samples]
    audio = audio.detach().cpu().to(torch.float32)    # safety

    buf = io.BytesIO()
    torchaudio.save(buf, audio, 16000, format="wav")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def handler(job):
    """
    Expects: { "input": { "text": str, "language": "en"|"zh"|"de" (optional) } }
    Returns: { "language": str, "audio_base64": str }
    """
    job_input = job.get("input", {})
    text = job_input.get("text")
    language = job_input.get("language", DEFAULT_LANGUAGE)

    if not isinstance(text, str) or not text.strip():
        return {"error": "Missing required 'text' (non-empty string)."}

    try:
        audio_b64 = synthesize_speech(text.strip(), language)
        return {"language": language, "audio_base64": audio_b64}
    except Exception as e:
        # Let RunPod mark the job as FAILED and surface the exception details
        raise RuntimeError(f"Inference failed: {e}")

runpod.serverless.start({"handler": handler})
