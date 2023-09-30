import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import hamming, medfilt

class MicrophoneFrequencyAnalyzer:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK_SIZE = int(self.RATE * 0.2)
        self.MOVING_AVERAGE_WINDOW = 5  # Number of previous chunks to average
        self.THRESHOLD = 270000  # Adjust this threshold as needed
        self.audio = pyaudio.PyAudio()
        self.audio_buffer = []

    def open_microphone_stream(self):
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )

    def apply_moving_average(self, audio_data):
        self.audio_buffer.append(audio_data)
        if len(self.audio_buffer) > self.MOVING_AVERAGE_WINDOW:
            self.audio_buffer.pop(0)
        return np.mean(self.audio_buffer, axis=0, dtype=np.int16)

    def analyze_frequency(self):
        try:
            while True:
                audio_data = self.stream.read(self.CHUNK_SIZE)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Apply Hamming window to the audio data
                windowed_audio = audio_array * hamming(len(audio_array))
                
                # Apply moving average filter to reduce noise
                filtered_audio = self.apply_moving_average(windowed_audio)
                
                fft_result = fft(filtered_audio)
                
                # Apply median filter to the FFT result
                filtered_fft_result = medfilt(np.abs(fft_result), kernel_size=3)
                
                # Find the lowest frequency above the threshold
                lowest_freq_above_threshold = self.find_lowest_freq_above_threshold(filtered_fft_result)
                
                if lowest_freq_above_threshold is not None:
                    print(f"Lowest Frequency Above Threshold: {lowest_freq_above_threshold} Hz")
                
                # Plot the FFT result
                self.plot_fft_result(filtered_fft_result)
        except KeyboardInterrupt:
            pass

    def find_lowest_freq_above_threshold(self, fft_result):
        freqs = np.fft.fftfreq(len(fft_result), 1.0 / self.RATE)
        magnitudes = np.abs(fft_result)
        
        # Find frequencies above the threshold
        above_threshold = np.where(magnitudes > self.THRESHOLD)
        
        if len(above_threshold[0]) > 0:
            print(magnitudes[above_threshold[0]])
            # Get the frequency with the lowest index (lowest frequency) above the threshold
            return freqs[above_threshold[0][0]]
        else:
            return None

    def plot_fft_result(self, fft_result):
        plt.clf()
        freqs = np.fft.fftfreq(len(fft_result), 1.0 / self.RATE)
        plt.plot(freqs[:900], np.abs(fft_result)[:900])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT Result (Frequency < 900 Hz)')
        plt.grid(True)
        plt.xlim(0, 900)
        plt.pause(0.01)

    def close_microphone_stream(self):
        self.stream.stop_stream()
        self.stream.close()

    def cleanup(self):
        self.audio.terminate()

if __name__ == "__main__":
    analyzer = MicrophoneFrequencyAnalyzer()
    analyzer.open_microphone_stream()
    plt.ion()  # Enable interactive mode for plotting
    analyzer.analyze_frequency()
    analyzer.close_microphone_stream()
    analyzer.cleanup()