import pandas as pd

class BoundingBoxTracker:
    def __init__(self, detection_filename, annotation_filename, video_path):
        self.detection_data = pd.read_csv(detection_filename)
        self.annotation_data = pd.read_csv(annotation_filename)
        self.video_path = video_path