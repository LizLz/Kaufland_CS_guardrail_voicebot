import asyncio
import os
import shutil
import subprocess
import sys
import time
from typing import Optional

import aiohttp
from dotenv import load_dotenv

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from utility.audio import Microphone
from utility.logger import setup_logger

load_dotenv()
logger = setup_logger()


class Config:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

        self.stt_model = "nova-3"
        self.stt_language = "de"
        self.tts_model = "aura-2-lara-de"
        self.keyterms = ["Kaufland Pay", "Kaufland Card XTRA", "Bluecode", "Kaufland"]

        self._validate()

    def _validate(self):
        if not self.groq_api_key:
            raise ValueError("ERROR: GROQ_API_KEY is not set in the environment.")
        if not self.deepgram_api_key:
            raise ValueError("ERROR: DEEPGRAM_API_KEY is not set in the environment.")
        if not self._is_installed("ffplay"):
            raise RuntimeError("ERROR: ffplay is not installed. Please install ffmpeg to play audio.")

    @staticmethod
    def _is_installed(lib_name: str) -> bool:
        return shutil.which(lib_name) is not None
    

class LiveTranscriber:
    """Handles real-time speech-to-text transcription using Deepgram per turn."""
    def __init__(self, config: Config):
        self.client = AsyncDeepgramClient(api_key=config.deepgram_api_key)
        self.stt_model = config.stt_model
        self.stt_language = config.stt_language
        self.keyterms = config.keyterms

        self.transcript_future: Optional[asyncio.Future] = None
        self.current_connection = None
        self.utterance_buffer = []

        self.microphone = Microphone(send_callback=self._mic_callback)

    def _mic_callback(self, audio_bytes):
        if not self.current_connection:
            return

        async def send_audio():
            try:
                await self.current_connection.send_media(audio_bytes)
            except Exception as e:
                logger.debug(f"send_media failed for one audio chunk: {e}")

        asyncio.create_task(send_audio())

    async def listen(self, on_speech_detected=None) -> str:
        self.transcript_future = asyncio.Future()
        self.utterance_buffer = []

        self.microphone.start()

        async with self.client.listen.v1.connect(
            model=self.stt_model,
            language=self.stt_language,
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=1000,
            interim_results=True,
            keyterm=self.keyterms,
        ) as connection:
            self.current_connection = connection
            connection.on(EventType.MESSAGE, self._on_message)
            connection.on(EventType.ERROR, self._on_error)

            listen_task = asyncio.create_task(connection.start_listening())

            try:
                final_transcript = await asyncio.wait_for(self.transcript_future, timeout=20.0)
                return final_transcript
            except asyncio.TimeoutError:
                logger.warning("No speech detected within 20s timeout.")
                return ""
            finally:
                self.current_connection = None
                self.microphone.stop()
                listen_task.cancel()
                try:
                    await listen_task
                except (asyncio.CancelledError, Exception):
                    pass

    def _on_message(self, message, **kwargs) -> None:
        try:
            msg_type = getattr(message, "type", None)

            if msg_type == "UtteranceEnd":
                full_utterance = " ".join(self.utterance_buffer).strip()
                if full_utterance and self.transcript_future and not self.transcript_future.done():
                    self.transcript_future.set_result(full_utterance)
                return

            if hasattr(message, "channel") and hasattr(message.channel, "alternatives"):
                transcript = message.channel.alternatives[0].transcript.strip()
                is_final = getattr(message, "is_final", False)
                speech_final = getattr(message, "speech_final", False)

                if is_final and transcript:
                    self.utterance_buffer.append(transcript)

                if speech_final:
                    full_utterance = " ".join(self.utterance_buffer).strip()
                    if full_utterance and self.transcript_future and not self.transcript_future.done():
                        self.transcript_future.set_result(full_utterance)

        except Exception as e:
            logger.error(f"STT Parse Error: {e}")

    def _on_error(self, error, **kwargs) -> None:
        logger.error(f"STT Error: {error}")
        if self.transcript_future and not self.transcript_future.done():
            self.transcript_future.set_exception(Exception(f"STT Error: {error}"))

    async def cleanup(self):
        try:
            self.microphone.stop()
        except Exception:
            pass
        if self.current_connection:
            try:
                await self.current_connection.finish()
            except Exception:
                pass
        self.current_connection = None


class GraphProcessor:
    """Manages interaction with the Kaufland LangGraph workflow."""
    def __init__(self, config: Config):
        from src.graph import build_kaufland_graph  # matches the rest of the project's core.* import convention
        import uuid
        from langchain_core.messages import HumanMessage

        self.app = build_kaufland_graph()
        self.thread_id = str(uuid.uuid4())
        self.graph_config = {"configurable": {"thread_id": self.thread_id}}
        self._HumanMessage = HumanMessage

    def _full_state(self, user_text: str) -> dict:
        """Always initialize every SupportState field explicitly. A
        previous version only set 'messages', leaving every other field
        absent (not even a default) on the very first turn of a session —
        this matches a known failure pattern seen earlier in this project
        where nodes assumed fields like 'pending_escalation' were always
        present."""
        return {
            "messages": [self._HumanMessage(content=user_text)],
            "action": "",
            "retrieved_context": "",
            "confidence_score": 0.0,
            "confidence_tier": "",
            "escalation_ticket": {},
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }

    async def generate_response(self, user_text: str) -> dict:
        start_time = time.time()
        
        # Pass only the new message dict. This allows MemorySaver to retain 
        # persistent state like failed_attempt_count and pending_escalation across turns.
        state_in = {"messages": [self._HumanMessage(content=user_text)]}
        result = await self.app.ainvoke(state_in, config=self.graph_config)
        
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "text": result["messages"][-1].content,
            "action": result.get("action", ""),
            "confidence": result.get("confidence_score", 0.0),
            "pending_escalation": result.get("pending_escalation", False),
            "elapsed_ms": elapsed_ms,
        }


class SpeechSynthesizer:
    """Handles text-to-speech conversion and audio playback using Deepgram."""
    def __init__(self, config: Config):
        self.api_key = config.deepgram_api_key
        self.model_name = config.tts_model
        self.api_url = f"https://api.deepgram.com/v1/speak?model={self.model_name}&encoding=linear16&sample_rate=24000"
        self.player_process = None
        self.last_ttfb = 0

    async def speak(self, text: str):
        """One TTS request, one player process, per full answer — avoids
        the inter-sentence pauses caused by spawning a new process and
        making a new HTTP request per sentence."""
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        payload = {"text": text}

        self.player_process = await asyncio.create_subprocess_exec(
            "ffplay", "-autoexit", "-", "-nodisp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        request_start_time = time.time()
        self.last_ttfb = 0

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    first_byte_received = False

                    async for chunk in response.content.iter_chunked(1024):
                        if chunk and self.player_process and self.player_process.stdin:
                            if not first_byte_received:
                                self.last_ttfb = int((time.time() - request_start_time) * 1000)
                                first_byte_received = True
                            try:
                                self.player_process.stdin.write(chunk)
                                await self.player_process.stdin.drain()
                            except (ConnectionResetError, AttributeError):
                                break

            if self.player_process and self.player_process.stdin:
                self.player_process.stdin.close()
            if self.player_process:
                await self.player_process.wait()

        except Exception as e:
            logger.error(f"TTS Request Error: {e}")
        finally:
            self.stop()

    def stop(self):
        if self.player_process:
            try:
                self.player_process.terminate()
            except Exception:
                pass
            self.player_process = None

    async def cleanup(self):
        if self.player_process:
            try:
                self.player_process.terminate()
                await self.player_process.wait()
            except ProcessLookupError:
                pass
            self.player_process = None


# --- Main Application ---

class VoiceAssistant:
    TERMINATION_PHRASES = ["tschüss", "auf wiedersehen", "bye", "beenden"]

    def __init__(self, config: Config):
        self.transcriber = LiveTranscriber(config)
        self.llm_processor = GraphProcessor(config)
        self.synthesizer = SpeechSynthesizer(config)

    async def run(self):
        logger.info("Sprachassistent aktiviert", extra={"telemetry": {"session_id": self.llm_processor.thread_id}})
        turn_id = 0

        print("\n--- Sprachassistent aktiviert ---")
        print(f"Sagen Sie '{self.TERMINATION_PHRASES[0]}', um das Gespräch zu beenden.\n")

        while True:
            turn_id += 1
            telemetry = {
                "session_id": self.llm_processor.thread_id,
                "turn_id": turn_id,
                "stt_latency_ms": 0,
                "graph_latency_ms": 0,
                "tts_ttfb_ms": 0,
                "confidence": 0.0,
                "action": "",
                "escalated": False,
                "error_type": None,
            }

            try:
                print("\nHöre zu (Listening)...")
                stt_start = time.time()
                user_text = await self.transcriber.listen()
                telemetry["stt_latency_ms"] = int((time.time() - stt_start) * 1000)

                if not user_text:
                    continue

                logger.info("User spoke", extra={"telemetry": {"user_text": user_text}})
                print(f"Mensch: {user_text}")

                if any(phrase in user_text.lower().strip() for phrase in self.TERMINATION_PHRASES):
                    goodbye_message = "Auf Wiedersehen! Einen schönen Tag noch."
                    print(f"🤖 KI: {goodbye_message}")
                    await self.synthesizer.speak(goodbye_message)
                    break

                response = await self.llm_processor.generate_response(user_text)
                telemetry["graph_latency_ms"] = response["elapsed_ms"]
                telemetry["action"] = response["action"]
                telemetry["confidence"] = response["confidence"]
                telemetry["escalated"] = response["action"] == "escalate"

                print(f"🤖 KI: {response['text']}")
                await self.synthesizer.speak(response["text"])
                telemetry["tts_ttfb_ms"] = self.synthesizer.last_ttfb

                logger.info("Turn complete", extra={"telemetry": telemetry})

                if telemetry["escalated"]:
                    logger.warning("System Escalation Triggered", extra={"telemetry": telemetry})
                    break

            except Exception as e:
                telemetry["error_type"] = type(e).__name__
                logger.error(f"Error in main loop: {e}", extra={"telemetry": telemetry})
                print("Starte Hörschleife neu...")
                await asyncio.sleep(1)

    async def cleanup(self):
        logger.info("Räume Ressourcen auf (Cleaning up resources)...")
        await self.synthesizer.cleanup()
        await self.transcriber.cleanup()


async def main():
    assistant = None
    try:
        config = Config()
        assistant = VoiceAssistant(config)
        await assistant.run()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        logger.error(f"Konfigurationsfehler: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n--- Assistent durch Benutzer beendet ---")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
        sys.exit(1)
    finally:
        if assistant:
            await assistant.cleanup()
        sys.stdout.flush()
        logger.info("Herunterfahren abgeschlossen. (Shutdown complete)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass