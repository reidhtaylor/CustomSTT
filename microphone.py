# Import necessary libraries
import sounddevice as sd
import threading
import wave

class Microphone:
    """
    Records audio from device and sends it to a callback function as byte chunks
    """

    # Instance variables
    stop_trigger : threading.Event
    worker_thread : threading.Thread
    sample_rate : int
    
    final_data: bytearray

    def __init__(self, sample_rate=8000):
        """
        Initialize the mic's instance variables
        """

        # Set Defaults
        self.stop_trigger = threading.Event()
        self.worker_thread = threading.Thread(target=self.mic_thread_started, daemon=True)
        self.sample_rate = sample_rate
        
        self.final_data = bytearray()

    def open_mic(self):
        """
        Start the mic on a thread (non-blocking)
        """

        # Initialize worker thread to capture audio
        self.worker_thread.start()
    
    def get_recorded_audio(self) -> bytes:
        """
        Close the mic and cleanup
        """

        self.stop_trigger.set()
        self.worker_thread.join()
        
        return self.final_data

    def mic_thread_started(self):
        """
        Called upon start of thread - Open input stream

        NOTE : blocksize (bytes) = (samplerate / blocksize) (seconds)
        """

        # 2400 == 100ms == 0.1s
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=2400,
            callback=self.mic_stream_callback
        ):
            while not self.stop_trigger.is_set():
                self.stop_trigger.wait(timeout=0.1)

    def mic_stream_callback(self, indata, frames, time, status):
        """
        Start the mic on a thread (non-blocking)

        Args:
            indata: data from sound device input stream
            frames: chunk frame length
            time: chunk time length
            status: status change flag
        """

        # Mic status has changed
        if status:
            print("\033[31m> Mic Status Update\033[0m", status)

        # Convert to int16 PCM
        pcm = (indata * 32767).astype("int16").tobytes()

        self.final_data.extend(pcm)