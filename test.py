import os
os.environ["MODEL_NAME"] = "microsoft/VibeVoice-1.5b"
from handler import synthesize_speech

if __name__ == "__main__":
    test_text = "Speaker 1: Hello, this is a test of the VibeVoice text to speech synthesis."
    test_language = "en"
    audio_base64 = synthesize_speech(test_text, test_language)
    print(f"Generated audio (base64): {audio_base64[:100]}...")  # Print first 100 chars

    # Save to file for verification
    import base64
    audio_data = base64.b64decode(audio_base64)
    with open("output.wav", "wb") as f:
        f.write(audio_data)
    print("Audio saved to output.wav")