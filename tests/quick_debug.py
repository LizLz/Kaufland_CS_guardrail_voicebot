import sounddevice as sd
import numpy as np

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Calculate the volume of the audio chunk
    volume = np.linalg.norm(indata) * 10
    
    # Create a visual volume bar
    bar = '|' * int(volume)
    print(f"Volume: {volume:05.1f} {bar}")

print("\n--- Verfügbare Mikrofone (Available Devices) ---")
print(sd.query_devices())

print("\n🎤 Sprechen Sie jetzt! (Speak now - Press Ctrl+C to stop)...")
try:
    # We use the exact same settings as your bot
    with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
        sd.sleep(10000) # Listen for 10 seconds
except KeyboardInterrupt:
    print("\nBeendet.")