import json
import base64
import os
import subprocess
import sys
import tempfile

def play_audio(file_path):
    """Plays audio using available system tools."""
    players = ["ffplay", "aplay", "paplay", "mpg123"] # Common linux players
    
    # Try to find a player
    for player in players:
        try:
            # Check if player exists
            subprocess.run(["which", player], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"Playing with {player}...")
            if player == "ffplay":
                # ffplay needs -nodisp and -autoexit for audio
                subprocess.run([player, "-nodisp", "-autoexit", file_path], check=True)
            else:
                subprocess.run([player, file_path], check=True)
            return True
        except subprocess.CalledProcessError:
            continue
        except Exception as e:
            print(f"Error running {player}: {e}")
            continue
            
    print("No suitable audio player found. Please install ffplay, aplay, or paplay.")
    return False

def main():
    json_file = "runpod_response.json"
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse {json_file}")
        return

    # Try to find audio_base64
    audio_b64 = None
    
    # Check root level
    if "audio_base64" in data:
        audio_b64 = data["audio_base64"]
    # Check inside "output" (RunPod format)
    elif "output" in data:
        if isinstance(data["output"], dict) and "audio_base64" in data["output"]:
            audio_b64 = data["output"]["audio_base64"]
    
    if not audio_b64:
        print("Error: No 'audio_base64' found in response.json")
        # Debug: print keys
        print(f"Available keys: {list(data.keys())}")
        if "output" in data and isinstance(data["output"], dict):
             print(f"Keys in output: {list(data['output'].keys())}")
        return

    if len(audio_b64) < 100:
        print("Warning: Audio data seems very short or empty.")

    try:
        audio_bytes = base64.b64decode(audio_b64)
        print(f"Decoded {len(audio_bytes)} bytes of audio.")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(audio_bytes)
            temp_wav_path = temp_wav.name
            
        print(f"Saved to temporary file: {temp_wav_path}")
        
        # Play
        play_audio(temp_wav_path)
        
        # Cleanup (optional, maybe user wants to keep it?)
        # os.remove(temp_wav_path) 
        print(f"Audio file kept at: {temp_wav_path}")

    except Exception as e:
        print(f"Error processing audio: {e}")

if __name__ == "__main__":
    main()
