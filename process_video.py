import os
import cv2
import numpy as np
import av
import tkinter as tk
from tkinter import filedialog, ttk
import threading

"""
Function to process MoSeq IR videos for DeepLabCut/SLEAP keypoint tracking
"""

class VideoProcessor:
    def __init__(self, master):
        """
        Initialize the VideoProcessor GUI.
        
        :param master: The root Tkinter window
        """
        self.master = master
        master.title("Video Processor")
        master.geometry("400x200")

        # Create and pack GUI elements
        self.select_button = tk.Button(master, text="Select Videos", command=self.select_videos)
        self.select_button.pack(pady=10)

        self.process_button = tk.Button(master, text="Process Videos", command=self.process_videos, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.progress_bar = ttk.Progressbar(master, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.pack(pady=10)

        self.status_label = tk.Label(master, text="Ready")
        self.status_label.pack(pady=10)

        self.input_files = []

    def select_videos(self):
        """
        Open a file dialog to select video files for processing.
        Enable the process button if videos are selected.
        """
        self.input_files = filedialog.askopenfilenames(filetypes=[("Video Files", "*.avi *.mp4")])
        if self.input_files:
            self.status_label.config(text=f'{len(self.input_files)} videos selected')
            self.process_button.config(state=tk.NORMAL)

    def process_videos(self):
        """
        Open a directory dialog to select the output directory.
        Start the video processing in a separate thread.
        """
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.process_button.config(state=tk.DISABLED)
            self.select_button.config(state=tk.DISABLED)
            self.status_label.config(text='Processing...')
            
            thread = threading.Thread(target=self.process_thread, args=(output_dir,))
            thread.start()

    def process_thread(self, output_dir):
        """
        Process all selected videos in a separate thread.
        Update the progress bar for each processed video.
        
        :param output_dir: Directory to save processed videos
        """
        for i, file in enumerate(self.input_files):
            self.process_video(file, output_dir)
            progress = int((i + 1) / len(self.input_files) * 100)
            self.master.after(0, self.update_progress, progress)
        
        self.master.after(0, self.processing_finished)

    def update_progress(self, value):
        """
        Update the progress bar value.
        
        :param value: Progress percentage (0-100)
        """
        self.progress_bar['value'] = value

    def processing_finished(self):
        """
        Update GUI elements when video processing is complete.
        """
        self.status_label.config(text='Processing complete')
        self.process_button.config(state=tk.NORMAL)
        self.select_button.config(state=tk.NORMAL)

    def adjust_brightness_contrast(self, frame, brightness=150, contrast=210):
        """
        Adjust the brightness and contrast of a frame.
        
        :param frame: Input frame
        :param brightness: Brightness adjustment value
        :param contrast: Contrast adjustment value
        :return: Adjusted frame
        """
        alpha = (contrast + 100) / 100.0
        beta = brightness
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted

    def adaptive_histogram_equalization(self, frame):
        """
        Apply adaptive histogram equalization to a frame.
        
        :param frame: Input frame
        :return: Equalized frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        return equalized_bgr

    def process_video(self, input_file, output_dir):
        """
        Process a single video file.
        Apply brightness/contrast adjustment and adaptive histogram equalization to each frame.
        Save the processed video using PyAV.
        
        :param input_file: Path to the input video file
        :param output_dir: Directory to save the processed video
        """
        output_name = os.path.splitext(os.path.basename(input_file))[0] + '_processed.mp4'
        output_path = os.path.join(output_dir, output_name)

        with av.open(input_file) as input_container:
            input_stream = input_container.streams.video[0]
            
            output_container = av.open(output_path, mode='w')
            output_stream = output_container.add_stream('h264', rate=input_stream.average_rate)
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = 'yuv420p'

            for frame in input_container.decode(input_stream):
                img = frame.to_ndarray(format='bgr24')
                
                adjusted_frame = self.adjust_brightness_contrast(img)
                equalized_frame = self.adaptive_histogram_equalization(adjusted_frame)
                
                out_frame = av.VideoFrame.from_ndarray(equalized_frame, format='bgr24')
                packet = output_stream.encode(out_frame)
                output_container.mux(packet)

            # Flush the encoder
            for packet in output_stream.encode():
                output_container.mux(packet)

            output_container.close()

if __name__ == '__main__':
    root = tk.Tk()
    app = VideoProcessor(root)
    root.mainloop()
