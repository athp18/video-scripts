import os
import cv2
import ffmpeg
import numpy as np

# Function to adjust brightness and contrast
def adjust_brightness_contrast(frame, brightness=0, contrast=0):
    alpha = (contrast + 100) / 100.0
    beta = brightness
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return adjusted

# Function to perform adaptive histogram equalization on a frame
def adaptive_histogram_equalization(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return equalized_bgr

def process(file, output_dir, display=True):
    try:
        cap = cv2.VideoCapture(file)
    except Exception as e:
        print('Exception', e)
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_name = os.path.splitext(os.path.basename(file))[0] + '_processed.mp4'
    output_path = os.path.join(output_dir, output_name)
    
    ffmpeg_cmd = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
        .output(output_path, pix_fmt='yuv420p', vcodec='libx264', preset='ultrafast', tune='film')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        adjusted_frame = adjust_brightness_contrast(frame, brightness=150, contrast=210)
        equalized_frame = adaptive_histogram_equalization(adjusted_frame)
        frame_bytes = equalized_frame.astype(np.uint8).tobytes()
        
        ffmpeg_cmd.stdin.write(frame_bytes)
        
        if display:
            cv2.imshow('Equalized Video', equalized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    ffmpeg_cmd.stdin.close()
    ffmpeg_cmd.wait()
    cv2.destroyAllWindows()

def process_folder(folder, output_dir, display=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = os.listdir(folder)
    for file in files:
        if file.endswith('.avi') or file.endswith('.mp4'):
            process(os.path.join(folder, file), output_dir, display)

def main():
    path = input()
    output_dir = input()
    display = input()
    if display.lower() == 'true':
      display = True
    else:
      display = False
    process_folder(path=path, output_dir=output_dir, display=display)

if __name__ == '__main__':
    main()
