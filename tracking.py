import cv2
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        self.x = np.zeros((dim_x, 1))  # state
        self.P = np.eye(dim_x)  # uncertainty covariance
        self.Q = np.eye(dim_x)  # process uncertainty
        self.R = np.eye(dim_z)  # measurement uncertainty
        
        self.F = np.eye(dim_x)  # state transition matrix
        self.H = np.zeros((dim_z, dim_x))  # measurement function
        
        self.y = np.zeros((dim_z, 1))  # residual
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.K = np.zeros((dim_x, dim_z))  # Kalman gain
        
    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
    
    def update(self, z):
        self.y = z - np.dot(self.H, self.x)
        self.S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.K, self.y)
        I = np.eye(self.dim_x)
        self.P = np.dot((I - np.dot(self.K, self.H)), self.P)

class TrackedObject:
    def __init__(self, initial_position, initial_color):
        self.kf = KalmanFilter(6, 3)  # 6 state variables (x, y, vx, vy, ax, ay), 3 measurements (x, y, color)
        self.kf.x[:2] = initial_position.reshape(2, 1)
        self.kf.F = np.array([[1, 0, 1, 0, 0.5, 0],
                              [0, 1, 0, 1, 0, 0.5],
                              [0, 0, 1, 0, 1, 0],
                              [0, 0, 0, 1, 0, 1],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])
        self.kf.R *= 10
        self.kf.Q[4:, 4:] *= 0.01
        self.color = initial_color
        self.age = 0
        self.total_visible_count = 0
        self.consecutive_invisible_count = 0
    
    def predict(self):
        return self.kf.predict()[:2].flatten()
    
    def update(self, measurement):
        self.kf.update(measurement)
        self.age += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
    
    def update_color(self, color):
        self.color = 0.7 * self.color + 0.3 * color
    
    def mark_missing(self):
        self.consecutive_invisible_count += 1

class EMTracker:
    def __init__(self, num_objects, dim=2):
        self.num_objects = num_objects
        self.dim = dim
        self.tracked_objects = []
        self.min_weight = 0.01
        self.max_age = 10
        self.min_hits = 3

    def initialize(self, points, colors):
        for point, color in zip(points, colors):
            self.tracked_objects.append(TrackedObject(point, color))

    def predict(self):
        return np.array([obj.predict() for obj in self.tracked_objects])

    def update(self, detections, colors):
        predicted_positions = self.predict()
        
        if len(detections) == 0:
            for obj in self.tracked_objects:
                obj.mark_missing()
            return
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(self.tracked_objects), len(detections)))
        for i, obj in enumerate(self.tracked_objects):
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(obj.predict() - detection)
        
        # Perform Hungarian algorithm for assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_indices = np.column_stack((row_ind, col_ind))
        unmatched_detections = [i for i in range(len(detections)) if i not in col_ind]
        unmatched_trackers = [i for i in range(len(self.tracked_objects)) if i not in row_ind]
        
        # Update matched trackers
        for row, col in matched_indices:
            self.tracked_objects[row].update(np.append(detections[col], colors[col]))
            self.tracked_objects[row].update_color(colors[col])
        
        # Handle unmatched detections and trackers
        for i in unmatched_detections:
            self.tracked_objects.append(TrackedObject(detections[i], colors[i]))
        
        for i in unmatched_trackers:
            self.tracked_objects[i].mark_missing()
        
        # Remove dead tracklets
        self.tracked_objects = [obj for obj in self.tracked_objects 
                                if obj.age >= self.min_hits or obj.consecutive_invisible_count <= self.max_age]

def detect_objects(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color range for object detection (adjust as needed)
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([180, 255, 255])
    
    # Create a mask using the color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    min_area = 100
    max_area = 10000
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    # Get centroids and colors of valid contours
    centroids = []
    colors = []
    for contour in valid_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append([cx, cy])
            
            # Get average color of the contour region
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_color = cv2.mean(frame, mask=mask)[:3]
            colors.append(mean_color)
    
    return np.array(centroids), np.array(colors), valid_contours

def draw_tracking_results(frame, tracker, contours):
    # Draw contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    
    # Draw tracked objects
    for i, obj in enumerate(tracker.tracked_objects):
        color = tuple(map(int, obj.color))
        
        # Convert mean and covariance to integer pixel coordinates
        mean_px = tuple(map(int, obj.kf.x[:2, 0]))
        
        # Calculate covariance of position
        cov = obj.kf.P[:2, :2]
        
        # Calculate eigenvalues and eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Calculate angle and axes lengths for the ellipse
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        axes_lengths = tuple(map(int, 4 * np.sqrt(eigenvalues)))
        
        # Draw the ellipse
        cv2.ellipse(frame, mean_px, axes_lengths, angle, 0, 360, color, 2)
        
        # Draw the centroid
        cv2.circle(frame, mean_px, 5, color, -1)
        
        # Add label
        cv2.putText(frame, f"Object {i+1}", (mean_px[0] + 10, mean_px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add velocity vector
        velocity = obj.kf.x[2:4, 0]
        end_point = tuple(map(int, mean_px + velocity * 10))
        cv2.arrowedLine(frame, mean_px, end_point, color, 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path
    
    # Initialize EM tracker
    tracker = EMTracker(num_objects=5)  # Allow tracking of up to 5 objects
    
    # Initialize tracker flag
    is_tracker_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        centroids, colors, contours = detect_objects(frame)
        
        if not is_tracker_initialized and len(centroids) > 0:
            tracker.initialize(centroids, colors)
            is_tracker_initialized = True
        
        if is_tracker_initialized:
            # Update tracker
            tracker.update(centroids, colors)
            
            # Draw tracking results
            frame = draw_tracking_results(frame, tracker, contours)

        # Display the frame
        cv2.imshow('Advanced EM Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
