import os
import io
import base64
import torch
import torchaudio
import runpod

# Import VibeVoice modules directly (PYTHONPATH must include /app/VibeVoice)
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# --- Env ---
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/VibeVoice-1.5b")
DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "en")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# --- Device ---
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

# --- Load once at startup ---
print(f"[VibeVoice] Loading model '{MODEL_NAME}' on {device}...")

def load_model(model_name, hf_token=None, device="cuda"):
    print(f"Loading processor from {model_name}")
    processor = VibeVoiceProcessor.from_pretrained(model_name, token=hf_token)
    
    # Determine dtype and attention implementation
    if device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    else:
        load_dtype = torch.float32
        attn_impl = "sdpa"

    print(f"Loading model with dtype={load_dtype}, attn={attn_impl}")
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name,
            torch_dtype=load_dtype,
            attn_implementation=attn_impl,
            device_map=device,
            token=hf_token
        )
    except Exception as e:
        print(f"Warning: Failed to load with {attn_impl}, falling back to sdpa. Error: {e}")
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name,
            torch_dtype=load_dtype,
            attn_implementation="sdpa",
            device_map=device,
            token=hf_token
        )
    
    model.eval()
    return model, processor

model, processor = load_model(model_name=MODEL_NAME, hf_token=HF_TOKEN, device=device)

# Default voice path (local copy)
DEFAULT_VOICE_PATH = "demo/voices/en-Carter_man.wav"

def synthesize_speech(text: str, language: str) -> str:
    """
    Generate speech audio from text and return base64-encoded WAV.
    """
    # VibeVoice requires a voice sample. Use default if not provided.
    voice_sample_path = DEFAULT_VOICE_PATH
    
    # Verify voice file exists
    if not os.path.exists(voice_sample_path):
        # Fallback: try to find any wav in voices dir
        voices_dir = "demo/voices"
        if os.path.exists(voices_dir):
            wavs = [f for f in os.listdir(voices_dir) if f.endswith('.wav')]
            if wavs:
                voice_sample_path = os.path.join(voices_dir, wavs[0])
                print(f"Using fallback voice: {voice_sample_path}")
            else:
                raise RuntimeError("No voice samples found in /app/VibeVoice/demo/voices")
        else:
             raise RuntimeError(f"Voice directory not found: {voices_dir}")

    # Prepare inputs
    inputs = processor(
        text=[text],
        voice_samples=[[voice_sample_path]], 
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move tensors to device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None, # Auto
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
            is_prefill=True 
        )

    # Extract audio
    if not hasattr(outputs, 'speech_outputs') or outputs.speech_outputs is None:
        raise RuntimeError("Model generated no speech output.")

    audio = outputs.speech_outputs[0]
    
    # Ensure it's CPU float32
    audio = audio.detach().cpu().to(torch.float32)
    
    # Ensure [channels, samples] format for torchaudio.save
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, samples]

    buf = io.BytesIO()
    # Sample rate is 24000 according to demo
    torchaudio.save(buf, audio, 24000, format="wav")
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

