import pyaudio
import cv2
import random
import time
import numpy as np
from scipy.fft import fft
from scipy.signal import hamming, medfilt

class Guitator:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK_SIZE = int(self.RATE * 0.2)
        self.audio = pyaudio.PyAudio()
        self.Filter_THRESHOLD = 1400000 
        self.Filter_freq_range = (77, 1000)
        self.valid_buffer = 6
        self.tune_list = {
            'e2':('E','2',82.41,[0,-1,-1,-1,-1,-1]),
            'f2':('F','2',87.31,[1,-1,-1,-1,-1,-1]),
            'g2':('G','2',98.00,[3,-1,-1,-1,-1,-1]),
            'a2':('A','2',110,[-1,0,-1,-1,-1,-1]),
            'b2':('B','2',123.47,[-1,2,-1,-1,-1,-1]),
            'c3':('C','3',130.81,[-1,3,-1,-1,-1,-1]),
            'd3':('D','3',146.83,[-1,-1,0,-1,-1,-1]),
            'e3':('E','3',164.81,[-1,-1,2,-1,-1,-1]),
            'f3':('F','3',174.61,[-1,-1,3,-1,-1,-1]),
            'g3':('G','3',196,[-1,-1,-1,0,-1,-1]),
            'a3':('A','3',220.00,[-1,-1,-1,2,-1,-1]),
            'b3':('B','3',246.94,[-1,-1,-1,-1,0,-1]),
            'c4':('C','4',261.63,[-1,-1,-1,-1,1,-1]),
            'd4':('D','4',293.66,[-1,-1,-1,-1,3,-1]),
            'e4':('E','4',329.63,[-1,-1,-1,-1,-1,0]),
            'f4':('F','4',349.23,[-1,-1,-1,-1,-1,1]),
            'g4':('G','4',392.00,[-1,-1,-1,-1,-1,3])
        }
        for key,tune in self.tune_list.items():
            tune[3].reverse()

    def open_microphone_stream(self):
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )
    
    @staticmethod
    def draw_line(image,number=[0,0,0,0,0,0]):
        height,width = image.shape[0:2]

        left_top = [width*0.2,height*0.6]
        
        space = int(height*0.3/6)

        for i in range(0,6,1):
            start_point = np.array(left_top) + np.array([0,space])*i
            end_point = start_point+np.array([width*0.6,0])
            cv2.line(image, (int(start_point[0]),int(start_point[1])), (int(end_point[0]),int(end_point[1])), (255,255,255), thickness=2)
            
            mid_pt = start_point+(end_point-start_point)/2
            if number[i]!=-1:
                text = str(number[i])
                cv2.putText(image, text, (int(mid_pt[0]),int(end_point[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
   
    def drawIndicator(self,image,curr,target):
        height,width = image.shape[0:2]
        length = width*0.3
        indicator_mid =  np.array([width*0.5,height*0.2])
        
        total_buff=30
        indicator = (curr-target)/total_buff
        if indicator>1:
            indicator=1
        elif indicator<-1:
            indicator=-1
        print(indicator)
        pos = indicator*length
        start_point = indicator_mid+np.array([pos,0])+np.array([0,-10])
        end_point = indicator_mid+np.array([pos,0])+np.array([0,10])
        cv2.line(image, (int(start_point[0]),int(start_point[1])), (int(end_point[0]),int(end_point[1])), (255,255,255), thickness=2)

        indicator = self.valid_buffer/total_buff
        pos = indicator*length
        start_point = indicator_mid+np.array([pos,0])+np.array([0,-20])
        end_point = indicator_mid+np.array([pos,0])+np.array([0,20])
        cv2.line(image, (int(start_point[0]),int(start_point[1])), (int(end_point[0]),int(end_point[1])), (255,255,255), thickness=1)

        indicator = -self.valid_buffer/total_buff
        pos = indicator*length
        start_point = indicator_mid+np.array([pos,0])+np.array([0,-20])
        end_point = indicator_mid+np.array([pos,0])+np.array([0,20])
        cv2.line(image, (int(start_point[0]),int(start_point[1])), (int(end_point[0]),int(end_point[1])), (255,255,255), thickness=1)

    def getHz(self, fft_result):
        freqs = np.fft.fftfreq(len(fft_result), 1.0 / self.RATE)
        magnitudes = np.abs(fft_result)
       

        # Find frequencies within the specified frequency range and above the threshold
        valid_indices = np.where((freqs >= self.Filter_freq_range[0]) & (freqs <= self.Filter_freq_range[1]) & (magnitudes > self.Filter_THRESHOLD))

        if len(valid_indices[0]) > 0:
            # Get the frequencies within the range and above the threshold
            valid_frequencies = freqs[valid_indices]
            print("valid mag: ",magnitudes[valid_indices])
            
            # Get the corresponding magnitudes for valid frequencies
            valid_magnitudes = magnitudes[valid_indices]
            
            # Sort the valid frequencies based on their magnitudes
            sorted_indices = np.argsort(valid_frequencies)
            sorted_valid_frequencies = valid_frequencies[sorted_indices]
            print("valid freq: ",sorted_valid_frequencies)
            # median_frequency = sorted_valid_frequencies[int(len(sorted_valid_frequencies)*0.85)]
            
            sorted_valid_frequencies = np.average(sorted_valid_frequencies[0:2])
            
            return sorted_valid_frequencies
        else:
            return None

    def Main(self):
        timer = time.time()

        
        width, height = 500, 500
        random_value =self.tune_list['e3']
        rate = 1
        image = np.zeros((height, width, 3), dtype=np.uint8)

        valid = False
        color=(255,255,255)
        try:
            while True:
                audio_data = self.stream.read(self.CHUNK_SIZE)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                fft_result = fft(audio_array)
               # Apply median filter to the FFT result
                filtered_fft_result = medfilt(np.abs(fft_result), kernel_size=3)
                
                # Find the lowest frequency above the threshold
                dominant_frequency = self.getHz(filtered_fft_result)

                now = time.time()
                image = np.zeros((height, width, 3), dtype=np.uint8)

                # if dominant_frequency is not None:
                #     for hz in dominant_frequency:
                #         if abs(hz - random_value[2])<self.valid_buffer:
                #             valid=True
                if dominant_frequency:
                    print(f"Dominant Frequency: {dominant_frequency} Hz")
                    self.drawIndicator(image,dominant_frequency,random_value[2])
                    if abs(dominant_frequency - random_value[2])<self.valid_buffer:
                                valid=True
                # if now-timer>round(1/rate):

                if valid:
                    color=(0,255,0)
                    if now-timer>round(1/rate): 
                        valid = False
                        color=(255,255,255)
                        random_key, random_value = random.choice(list(self.tune_list.items()))
                        print("random_key: ",random_key)
                        print("random_value: ",random_value)
                        timer = time.time()
                else:
                    timer = time.time()
                cv2.putText(image, str(random_value[0]), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 2)
                cv2.putText(image, str(random_value[1]), (280, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(image, "Hz: "+str(random_value[2]), (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image,"rate: "+str(rate), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1 , (255, 255, 255), 1)

                self.draw_line(image,number=random_value[3])
                cv2.imshow('Practice', image)

                key = cv2.waitKey(1)
                if key==ord("q"):
                    break
                elif key==ord("w"):
                    rate+=0.2
                elif key ==ord("s"):
                    rate-=0.2
                elif key ==ord("a"):
                    self.valid_buffer+=1
                elif key ==ord("d"):
                    self.valid_buffer-=1
                rate = round(rate,1)
                if rate<=0:
                    rate = 0.2

        except KeyboardInterrupt:
            pass

    def close_microphone_stream(self):
        self.stream.stop_stream()
        self.stream.close()

    def cleanup(self):
        self.audio.terminate()

if __name__ == "__main__":
    analyzer = Guitator()
    analyzer.open_microphone_stream()
    analyzer.Main()
    analyzer.close_microphone_stream()
    analyzer.cleanup()
    cv2.destroyAllWindows()