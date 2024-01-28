import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image as Img

from src.utils.bounding_box import BoundingBox

class VideoUtils:
    """
    Utility class for videos.
    """
    @staticmethod
    def patch_from_bbox(image: np.ndarray, bbox: BoundingBox) -> Img.Image:
        """
        Returns a patch from an image, given a bounding box.
        Args:
            image (np.ndarray): The image.
            bbox (BoundingBox): The bounding box.
        Returns:
            (Img.Image): The patch.
        """
        top = max(int(bbox.top), 0)
        left = max(int(bbox.left), 0)
        bottom = min(int(bbox.top + bbox.height), image.shape[0])
        right = min(int(bbox.left + bbox.width), image.shape[1])

        if right <= left or bottom <= top:
            return Img.fromarray(np.zeros((int(bbox.height), int(bbox.width), 3), dtype=np.uint8), mode='RGB')

        np_patch = image[top:bottom, left:right]
        return Img.fromarray(np_patch, mode='RGB')

    def patches_from_bbox_list(image: np.ndarray, bbox_list: list[BoundingBox]) -> list[Img.Image]:
        """
        Returns a list of patches from an image, given a list of bounding boxes.
        Args:
            image (np.ndarray): The image.
            bbox_list (list[BoundingBox]): The list of bounding boxes.
        Returns:
            (Img.Image): The list of patches.
        """
        return list(map(lambda bbox: VideoUtils.patch_from_bbox(image, bbox), bbox_list))
    
    @staticmethod
    def load_video_images(path: str) -> np.ndarray:
        """
        Loads a video.
        Args:
            path (str): The path pointing to all the images.
        Returns:
            (np.ndarray): The video.
        """
        list_files = sorted(os.listdir(path))
        images = []
        for filename in tqdm(list_files):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(path, filename)
                image = cv2.imread(image_path)
                images.append(image)
        return np.array(images)

    @staticmethod
    def export_video_with_tracking(df: pd.DataFrame, folder_path: str, output: str, fps: int, frame_size: tuple) -> None:
        """
        Exports a video with tracking.
        Args:
            df (pd.DataFrame): A pandas DataFrame containing the detections.
            folder_path (str): The path to the folder containing the images.
            output (str): The path to the output video.
            fps (int): The FPS of the output video.
            frame_size (tuple): The frame size of the output video.
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output, fourcc, fps, frame_size)

        color = (0, 0, 255)

        list_files = sorted(os.listdir(folder_path))
        for i in tqdm(range(len(list_files))):
            filename = list_files[i]
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)

                image = cv2.resize(image, frame_size)

                for _, row in df[df['frame'] == i + 1].iterrows():
                    bbox = row[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.astype(np.int32)
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
                    cv2.putText(image, f"{int(row['id'])}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                video_writer.write(image)

        video_writer.release()