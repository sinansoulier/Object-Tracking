import numpy as np
import pandas as pd

from src.utils.bounding_box import BoundingBox

class TrackerUtils:
    @staticmethod
    def IoU(bbox_1: BoundingBox, bbox_2: BoundingBox) -> float:
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.
        Args:
            bbox_1 (BoundingBox): Object representing the bounding box, with attributes left, top, width, and height.
            bbox_2 (BoundingBox): Object representing the bounding box, with attributes left, top, width, and height.
        Returns:
            The IoU of the two bounding boxes.
        """
        intersection_area = BoundingBox.intersection_area(bbox_1, bbox_2)
        
        iou = intersection_area / float(bbox_1.area() + bbox_2.area() - intersection_area)
        
        if iou < 0.0:
            iou = 0.0
        elif iou > 1.0:
            iou = 1.0

        return iou

    @staticmethod
    def similarity_matrix(bbox_list_left: list[BoundingBox], bbox_list_right: list[BoundingBox]) -> np.ndarray:
        """
        Calculates the similarity matrix two list of bounding boxes.
        Args:
            det_df: A pandas DataFrame containing the detections.
        Returns:
            A numpy array of shape (num_detections, num_detections) containing the IoU of each detection pair.
        """
        sim_matrix = np.zeros((len(bbox_list_left), len(bbox_list_right)))
        for (i, bbox_left) in enumerate(bbox_list_left):
            for (j, bbox_right) in enumerate(bbox_list_right):
                sim_matrix[i, j] = TrackerUtils.IoU(bbox_left, bbox_right)
        return sim_matrix
    
    @staticmethod
    def df_to_bbox_list(df: pd.DataFrame) -> list[BoundingBox]:
        """
        Converts a pandas DataFrame to a list of bounding boxes.
        Args:
            df (pd.DataFrame): A pandas DataFrame containing the detections.
        Returns:
            list[BoundingBox]: A list of bounding boxes.
        """
        return list(map(lambda x: BoundingBox(x[0], x[1], x[2], x[3]),
                        df[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values))