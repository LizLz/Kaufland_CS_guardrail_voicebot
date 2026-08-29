import asyncio
import sounddevice as sd


class Microphone:
    """
    Captures audio from the default microphone and sends
    raw PCM audio chunks to a callback.

    Audio format:
        - 16 kHz
        - mono
        - 16-bit PCM
    """

    def __init__(
        self,
        send_callback,
        samplerate: int = 16000,
        channels: int = 1,
    ):
        self.send_callback = send_callback
        self.samplerate = samplerate
        self.channels = channels

        self._stream = None
        self._loop = asyncio.get_running_loop()

        self._chunk_count = 0

    def _callback(self, indata, frames, time_info, status):
        """Called by sounddevice whenever new microphone audio arrives."""

        if status:
            print(f"[Microphone Warning] {status}")

        # RawInputStream with int16 gives us raw PCM bytes.
        audio_bytes = bytes(indata)

        self._chunk_count += 1

        # sounddevice's callback runs in a separate audio thread.
        # Therefore we must safely pass the data back to asyncio.
        self._loop.call_soon_threadsafe(
            self.send_callback,
            audio_bytes,
        )

    def start(self):
        """Start capturing microphone audio."""

        if self._stream is not None:
            return

        self._stream = sd.RawInputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
        )

        self._stream.start()

        print("[Microphone] Recording started.")

    def stop(self):
        """Stop and close the microphone."""

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        print("[Microphone] Recording stopped.")