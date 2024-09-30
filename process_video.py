import os
import av
import numpy as np
import tkinter as tk
from tkinter import filedialog
from threading import Thread
from PIL import Image, ImageTk
from skimage import exposure, color, img_as_ubyte
import time

"""
Function to process Infrared Videos for DeepLabCut/SLEAP keypoint analysis
"""

def timing(f):
    """
    Decorator to measure the execution time of a function.

    Args:
    f (callable): function to be timed

    Returns:
    callable: A wrapped version of the function that prints its execution time
    """
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'Function {f.__name__} took: {te - ts:.4f} seconds')
        return result
    return wrap

class VideoProcessor:
    """Class to process videos using PyAV."""

    def __init__(self, brightness=150, contrast=210, fps=30, display=True):
        """
        Initialize processor arguments.

        Args:
        
        brightness (int): Brightness adjustment value. Higher values increase brightness. Default is 150.
        contrast (int): Contrast adjustment value. Values above 100 increase contrast. Default is 210.
        fps (int): Frames per second for the output video. Default is 30.
        display (bool): Whether to display the processed video during processing. Default is True.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.display = display
        self.fps = fps

    def adjust_brightness_contrast(self, frame):
        """
        Adjusts the brightness and contrast of a frame.

        Args:
        frame (numpy.array): the frame of the video, represented as a 3D numpy array of shape height x width x channels
        
        Returns:
        adjusted (numpy.array): Adjusted frame with modified brightness and contrast.
        """
        alpha = (self.contrast + 100) / 100.0
        beta = self.brightness
        adjusted = np.clip(frame * alpha + beta, 0, 255).astype(np.uint8)
        return adjusted

    def adaptive_histogram_equalization(self, frame):
        """
        Applies adaptive histogram equalization to a frame. This effectively spreads out the pixel values of the frame,
        "smoothening" the image.

        Args:
        frame (numpy.array): the frame of the video, represented as a 3D numpy array of shape height x width x channels

        Returns:
        equalized_rgb (numpy.array): Post-processed frame
        """
        gray = color.rgb2gray(frame)
        equalized = exposure.equalize_adapthist(gray, clip_limit=0.02)
        equalized = img_as_ubyte(equalized)
        equalized_rgb = np.stack([equalized]*3, axis=-1)
        return equalized_rgb

    @timing
    def process(self, file_path, output_dir, update_callback=None):
        """
        Process a single video file, applying brightness/contrast adjustments and adaptive histogram equalization.

        Args:
        file_path (str): Path to the input video file.
        output_dir (str): Directory where the processed video will be saved.
        update_callback (callable, optional): Function to call with each processed frame for display purposes.
                                                Should accept a numpy array representing the frame.
        Returns:
            None
        Raises:
            Exception: If there's an error opening the input file.

        Note:
            The processed video is saved in the output_dir with '_processed' appended to the original filename.
        """
        try:
            input_container = av.open(file_path)
        except Exception as e:
            print('Exception:', e)
            return

        output_name = os.path.splitext(os.path.basename(file_path))[0] + '_processed.mp4'
        output_path = os.path.join(output_dir, output_name)

        output_container = av.open(output_path, mode='w')
        stream = output_container.add_stream('libx264', rate=self.fps) 
        stream.pix_fmt = 'yuv420p'

        for frame in input_container.decode(video=0):
            frame_array = frame.to_ndarray(format='rgb24')
            adjusted_frame = self.adjust_brightness_contrast(frame_array)
            equalized_frame = self.adaptive_histogram_equalization(adjusted_frame)

            new_frame = av.VideoFrame.from_ndarray(equalized_frame, format='rgb24')

            # Set stream dimensions if not already set
            if stream.width is None or stream.height is None:
                stream.width = new_frame.width
                stream.height = new_frame.height

            packet = stream.encode(new_frame)
            if packet:
                output_container.mux(packet)

            if self.display and update_callback:
                update_callback(equalized_frame)

        # Flush encoder
        packet = stream.encode()
        if packet:
            output_container.mux(packet)

        # Close containers
        input_container.close()
        output_container.close()

class VideoProcessingGUI:
    """GUI application for video processing."""

    def __init__(self, master):
        """
        Initialize the VideoProcessingGUI.

        Args:
        master (tk.Tk): The root window of the application.
        """
        self.master = master
        master.title("Video Processor")

        self.processor = VideoProcessor()

        self.input_folder = ''
        self.output_folder = ''

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        """
        Create and arrange all GUI elements.

        This method sets up labels, buttons, sliders, and a canvas for displaying processed frames.
        """
        self.input_label = tk.Label(self.master, text="Input Folder:")
        self.input_label.pack()

        self.input_button = tk.Button(self.master, text="Select Input Folder", command=self.select_input_folder)
        self.input_button.pack()

        self.output_label = tk.Label(self.master, text="Output Folder:")
        self.output_label.pack()

        self.output_button = tk.Button(self.master, text="Select Output Folder", command=self.select_output_folder)
        self.output_button.pack()

        self.brightness_label = tk.Label(self.master, text="Brightness:")
        self.brightness_label.pack()

        self.brightness_scale = tk.Scale(self.master, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_brightness)
        self.brightness_scale.set(self.processor.brightness)
        self.brightness_scale.pack()

        self.contrast_label = tk.Label(self.master, text="Contrast:")
        self.contrast_label.pack()

        self.contrast_scale = tk.Scale(self.master, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.update_contrast)
        self.contrast_scale.set(self.processor.contrast)
        self.contrast_scale.pack()

        self.display_var = tk.BooleanVar()
        self.display_var.set(self.processor.display)
        self.display_checkbox = tk.Checkbutton(self.master, text="Display Video", variable=self.display_var, command=self.update_display)
        self.display_checkbox.pack()

        self.start_button = tk.Button(self.master, text="Start Processing", command=self.start_processing)
        self.start_button.pack()

        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack()

    def select_input_folder(self):
        """
        Open a dialog for selecting the input folder and update the GUI accordingly.
        """
        self.input_folder = filedialog.askdirectory()
        self.input_label.config(text=f"Input Folder: {self.input_folder}")

    def select_output_folder(self):
        """
        Open a dialog for selecting the output folder and update the GUI accordingly.
        """
        self.output_folder = filedialog.askdirectory()
        self.output_label.config(text=f"Output Folder: {self.output_folder}")

    def update_brightness(self, val):
        """
        Update the brightness setting of the video processor.

        Args:
        val (str): The new brightness value as a string.
        """
        self.processor.brightness = int(val)

    def update_contrast(self, val):
        """
        Update the contrast setting of the video processor.

        Args:
        val (str): The new contrast value as a string.
        """
        self.processor.contrast = int(val)

    def update_display(self):
        """
        Update the display setting of the video processor based on the checkbox state.
        """
        self.processor.display = self.display_var.get()

    def start_processing(self):
        """
        Initiate the video processing in a separate thread.

        This method checks if input and output folders are selected before starting the process.
        """
        if not self.input_folder or not self.output_folder:
            print("Please select input and output folders")
            return
        Thread(target=self.process_folder).start()

    def process_folder(self):
         """
        Process all video files in the selected input folder.

        This method iterates through all .avi and .mp4 files in the input folder,
        processes each file, and saves the result in the output folder.
        """
        files = os.listdir(self.input_folder)
        video_files = [f for f in files if f.lower().endswith(('.avi', '.mp4'))]
        for file in video_files:
            file_path = os.path.join(self.input_folder, file)
            print(f"Processing {file_path}")
            self.processor.process(file_path, self.output_folder, update_callback=self.display_frame)
        print("Processing complete")

    def display_frame(self, frame):
        """
        Schedule the display of a processed video frame.

        Args:
        frame (numpy.array): The processed video frame to display.
        """
        self.master.after(0, self.update_image, frame)

    def update_image(self, frame):
        """
        Update the canvas with a new processed video frame.

        Args:
        frame (numpy.array): The processed video frame to display.
        """
        image = Image.fromarray(frame)
        image = image.resize((640, 480), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.canvas.image_tk = image_tk  # Keep reference

def main():
    root = tk.Tk()
    gui = VideoProcessingGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
