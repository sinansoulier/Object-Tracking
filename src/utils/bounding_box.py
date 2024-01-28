import pandas as pd

from src.KalmanFilter import KalmanFilter
from src.utils.point import Point

class BoundingBox:
    """
    A class representing a bounding box.
    """
    def __init__(self, left: int, top: int, width: int, height: int):
        self.left: int = left
        self.top: int = top
        self.width: int = width
        self.height: int = height
        self.center = Point(left + width / 2, top + height / 2)

    def print(self):
        """
        Prints the bounding box coordinates and dimensions.
        """
        print(f"left: {self.left}, top: {self.top}, width: {self.width}, height: {self.height}")

    def area(self) -> int:
        """
        Calculates the area of the bounding box.
        Returns:
            (int): The area of the bounding box.
        """
        return self.width * self.height

    def apply_kalman_filter(self, kalman_filter: KalmanFilter) -> 'BoundingBox':
        """
        Applies the Kalman filter to the current BoundingBox.
        Args:
            kalman_filter (KalmanFilter): The Kalman filter to apply.
        Returns:
            (BoundingBox): A new BoundingBox, with the Kalman filter applied.
        """
        predicted_state = kalman_filter.predict()[0]
        new_center = Point(predicted_state[0][0], predicted_state[1][0])

        return self.translation_from_center(new_center)
    
    def translation_from_center(self, new_center: Point) -> 'BoundingBox':
        """
        Returns a new BoundingBox, translated from the current BoundingBox, with the new center.
        Args:
            new_center (Point): The new center.
        Returns:
            (BoundingBox): A new BoundingBox, translated from the current BoundingBox, with the new center.
        """
        dx = new_center.x - self.center.x
        dy = new_center.y - self.center.y
        return BoundingBox(self.left + dx, self.top + dy, self.width, self.height)

    @staticmethod
    def intersection_area(bbox_1: 'BoundingBox', bbox_2: 'BoundingBox') -> int:
        """
        Calculates the intersection area of two bounding boxes.
        Args:
            bbox_1 (BoundingBox): Object representing the bounding box, with attributes left, top, width, and height.
            bbox_2 (BoundingBox): Object representing the bounding box, with attributes left, top, width, and height.
        Returns:
            (int): The intersection area of the two bounding boxes.
        """
        xA, yA = max(bbox_1.left, bbox_2.left), max(bbox_1.top, bbox_2.top)
        xB, yB = min(bbox_1.left + bbox_1.width, bbox_2.left + bbox_2.width), min(bbox_1.top + bbox_1.height, bbox_2.top + bbox_2.height)

        if xB <= xA or yB <= yA:
            return 0
        
        return (xB - xA) * (yB - yA)

    @staticmethod
    def bbox_from_row(row: pd.Series) -> 'BoundingBox':
        """
        Creates a BoundingBox from a pandas Series.
        Args:
            row (pd.Series): The pandas Series containing the bounding box coordinates and dimensions.
        Returns:
            (BoundingBox): A new BoundingBox, created from the pandas Series.
        """
        return BoundingBox(row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height'])