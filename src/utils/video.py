import os
import cv2
import numpy as np
import pandas as pd

class Video:
    @staticmethod
    def export_video_with_tracking(df: pd.DataFrame, folder_path: str, output: str, fps: int, frame_size: tuple) -> None:
        """
        Exports a video with tracking.
        Args:
            df: A pandas DataFrame containing the detections.
            folder_path: The path to the folder containing the images.
            output: The path to the output video.
            fps: The FPS of the output video.
            frame_size: The frame size of the output video.
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output, fourcc, fps, frame_size)

        color = (0, 0, 255)

        for i, filename in enumerate(sorted(os.listdir(folder_path))):
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