import numpy as np

class TrackerUtils:
    @staticmethod
    def IoU(bbox_1: np.ndarray, bbox_2: np.ndarray) -> float:
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.
        Args:
            bbox_1: A numpy array of shape (4,) representing the left, top, width, and height of the first bounding box.
            bbox_2: A numpy array of shape (4,) representing the left, top, width, and height of the second bounding box.
        Returns:
            The IoU of the two bounding boxes.
        """
        x1, y1, w1, h1 = bbox_1
        x2, y2, w2, h2 = bbox_2
        
        xA, yA = max(x1, x2), max(y1, y2)
        xB, yB = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xB <= xA or yB <= yA:
            return 0.0
        
        intersection_area = (xB - xA) * (yB - yA)
        
        bbox_1_area = w1 * h1
        bbox_2_area = w2 * h2
        
        iou = intersection_area / float(bbox_1_area + bbox_2_area - intersection_area)
        
        if iou < 0.0:
            iou = 0.0
        elif iou > 1.0:
            iou = 1.0

        return iou

    @staticmethod
    def similarity_matrix(bbox_list_left: np.ndarray, bbox_list_right: np.ndarray) -> np.ndarray:
        """
        Calculates the similarity matrix two list of bounding boxes.
        Args:
            det_df: A pandas DataFrame containing the detections.
        Returns:
            A numpy array of shape (num_detections, num_detections) containing the IoU of each detection pair.
        """
        sim_matrix = np.zeros((bbox_list_left.shape[0], bbox_list_right.shape[0]))
        for (i, bbox_left) in enumerate(bbox_list_left):
            for (j, bbox_right) in enumerate(bbox_list_right):
                sim_matrix[i, j] = TrackerUtils.IoU(bbox_left, bbox_right)
        return sim_matrix