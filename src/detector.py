import numpy as np
from PIL import Image
from ultralytics import YOLO

class PPEDetector:
    def __init__(self, model_path: str):
        """
        Initializes the YOLOv8 model with the specified weights file.
        """
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image, conf_threshold: float, iou_threshold: float = 0.5):
        """
        Runs inference on an uploaded image and returns the annotated image array.
        
        Parameters:
        - image: PIL Image object
        - conf_threshold: Minimum confidence score to filter detections
        - iou_threshold: Intersection Over Union threshold for non-maximum suppression
        """
        # Run YOLOv8 prediction
        results = self.model.predict(
            source=image, 
            conf=conf_threshold, 
            iou=iou_threshold,
            save=False  # We handle the visualization in memory instead of saving files
        )
        
        # results[0].plot() returns a BGR numpy array containing the bounding boxes
        annotated_image_bgr = results[0].plot()
        
        # Convert BGR (OpenCV format) back to RGB (Streamlit/PIL format)
        annotated_image_rgb = annotated_image_bgr[..., ::-1]
        
        return annotated_image_rgb