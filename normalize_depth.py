import numpy as np
import cv2

"""
Simple OpenCV-based python script to normalize depth videos for visualization
"""

def normalize_depth_frame(depth_frame, min_depth=None, max_depth=None):
    """
    Normalize a depth frame for visualization.
    
    Args:
    depth_frame (numpy.ndarray): Input depth frame
    min_depth (float): Minimum depth value (optional)
    max_depth (float): Maximum depth value (optional)
    
    Returns:
    numpy.ndarray: Normalized depth frame (0-255 uint8)
    """
    depth_frame = depth_frame.astype(np.float32)
    if min_depth is None:
        min_depth = np.min(depth_frame)
    if max_depth is None:
        max_depth = np.max(depth_frame)
    
    depth_frame = np.clip(depth_frame, min_depth, max_depth)
    normalized_frame = (depth_frame - min_depth) / (max_depth - min_depth)
    normalized_frame = (normalized_frame * 255).astype(np.uint8)
    
    return normalized_frame

def process_depth_video(input_path, output_path, min_depth=None, max_depth=None, display=True):
    """
    Process a depth video, normalizing and visualizing each frame.
    
    Args:
    input_path (str): Path to the input depth video
    output_path (str): Path to save the output visualized video
    min_depth (float): Minimum depth value (optional)
    max_depth (float): Maximum depth value (optional)
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Normalize the depth frame
        normalized_frame = normalize_depth_frame(frame, min_depth, max_depth)
        # apply jet colormap for visualization - the classic "bluish" depth vids
        color_mapped_frame = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)
        
        # Write the frame
        out.write(color_mapped_frame)
        
        # display
        if display:
          cv2.imshow('Processed Depth Frame', color_mapped_frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
