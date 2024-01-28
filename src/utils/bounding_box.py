class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

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