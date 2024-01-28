import pandas as pd

from src.KalmanFilter import KalmanFilter
from src.utils.point import Point

class BoundingBox:
    def __init__(self, left: int, top: int, width: int, height: int):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.center = Point(left + width / 2, top + height / 2)
    
    def print(self):
        print(f"left: {self.left}, top: {self.top}, width: {self.width}, height: {self.height}")

    def area(self) -> int:
        return self.width * self.height
    
    @staticmethod
    def intersection_area(bbox_1: 'BoundingBox', bbox_2: 'BoundingBox') -> int:
        xA, yA = max(bbox_1.left, bbox_2.left), max(bbox_1.top, bbox_2.top)
        xB, yB = min(bbox_1.left + bbox_1.width, bbox_2.left + bbox_2.width), min(bbox_1.top + bbox_1.height, bbox_2.top + bbox_2.height)

        if xB <= xA or yB <= yA:
            return 0
        
        return (xB - xA) * (yB - yA)

    @staticmethod
    def bbox_from_row(row: pd.Series) -> 'BoundingBox':
        return BoundingBox(row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height'])


    def apply_kalman_filter(self, kalman_filter: KalmanFilter) -> 'BoundingBox':
        """
        Applies the Kalman filter to the current BoundingBox.

        Args:
            kalman_filer: The Kalman filter to apply.
        Returns:
            A new BoundingBox, with the Kalman filter applied.
        """
        predicted_state = kalman_filter.predict()[0]
        new_center = Point(predicted_state[0][0], predicted_state[1][0])

        return self.translation_from_center(new_center)
    
    def translation_from_center(self, new_center: Point) -> 'BoundingBox':
        """
        Returns a new BoundingBox, translated from the current BoundingBox, with the new center.

        Args:
            new_center: The new center.
        Returns:
            A new BoundingBox, translated from the current BoundingBox, with the new center.
        """
        dx = new_center.x - self.center.x
        dy = new_center.y - self.center.y
        return BoundingBox(self.left + dx, self.top + dy, self.width, self.height)