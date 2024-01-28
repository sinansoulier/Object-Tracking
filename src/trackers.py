import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.utils.tracker import TrackerUtils
from src.utils.tracker_type import TrackerType
from src.utils.bounding_box import BoundingBox

class Trackers:
    def __init__(self):
        self.list_columns = ['bb_left', 'bb_top', 'bb_width', 'bb_height']

    def track(self, df: pd.DataFrame, tracker_type: TrackerType) -> pd.DataFrame:
        """
        Matches tracks to detections, using the specified tracker type.
        Args:
            det_df: A pandas DataFrame containing the detections.
            tracker_type: The type of tracker to use.
        Returns:
            A pandas DataFrame containing the matches.
        """
        match tracker_type:
            case TrackerType.GREEDY:
                return self.greedy(df)
            case TrackerType.HUNGARIAN:
                return self.hungarian_tracking(df)
            case _:
                raise ValueError(f"Invalid tracker type: {tracker_type}")

    def greedy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Matches tracks to detections.

        Args:
            det_df: A pandas DataFrame containing the detections.
        Returns:
            A pandas DataFrame containing the matches.
        """
        frames = df['frame'].unique()
        for frame in range(1, frames.shape[0]):
            # Assign IDs to the first frame
            if frame == 1:
                df.loc[df['frame'] == frame, 'id'] = np.arange(df[df['frame'] == frame].shape[0])

            # Get current frame n
            tracks_frame = self.get_tracks(df, frame)
            tracks_frame_bboxes: list[BoundingBox] = TrackerUtils.df_to_bbox_list(tracks_frame)
            # Get next frame n+1
            detections_frame = self.get_detections(df, frame)
            detections_frame_bboxes: list[BoundingBox] = TrackerUtils.df_to_bbox_list(detections_frame)
            # Get similarity matrix between n and n+1
            sim_matrix = TrackerUtils.similarity_matrix(tracks_frame_bboxes, detections_frame_bboxes)
            # Set similarity matrix values below 0.5 to 0.0
            sim_matrix[sim_matrix < 0.5] = 0.0
            # Get the indices of the maximum values in each row
            max_indices = np.argmax(sim_matrix, axis=1)

            # For each row in the similarity matrix, if the maximum value is greater than 0.0,
            # assign the ID of the current frame to the ID of the next frame
            for i, max_index in enumerate(max_indices):
                if sim_matrix[i, max_index] > 0.0:
                    detections_frame.iloc[max_index, 1] = tracks_frame.iloc[i, 1]
            
            # For all rows in the next frame that have an ID of -1, assign a new ID
            for i, row in detections_frame.iterrows():
                if row['id'] == -1:
                    detections_frame.loc[i, 'id'] = detections_frame['id'].max() + 1
            
            # Update the IDs in the original input DataFrame
            df.loc[df['frame'] == frame + 1, 'id'] = detections_frame['id'].values

        return df

    def hungarian_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Matches tracks to detections using the Hungarian algorithm.
        Args:
            det_df: A pandas DataFrame containing the detections.
        Returns:
            A pandas DataFrame containing the matches.
        """
        frames = df['frame'].unique()
        
        for frame in range(1, frames.shape[0]):
            # Assign IDs to the first frame
            if frame == 1:
                df.loc[df['frame'] == frame, 'id'] = np.arange(df[df['frame'] == frame].shape[0])

            # Get current frame n
            tracks_frame = self.get_tracks(df, frame)
            tracks_frame_bboxes: list[BoundingBox] = TrackerUtils.df_to_bbox_list(tracks_frame)
            # Get next frame n+1
            detections_frame = self.get_detections(df, frame)
            detections_frame_bboxes: list[BoundingBox] = TrackerUtils.df_to_bbox_list(detections_frame)
            # Get similarity matrix between n and n+1
            sim_matrix = TrackerUtils.similarity_matrix(tracks_frame_bboxes, detections_frame_bboxes)
            
            row_ind, col_ind = linear_sum_assignment(1 - sim_matrix)

            # For all rows columns indices, assign the ID of the row index to the next frame
            for i, j in zip(row_ind, col_ind):
                detections_frame.iloc[j, 1] = tracks_frame.iloc[i, 1]
            
            # For all rows in the next frame that have an ID of -1, assign a new ID
            for i, row in detections_frame.iterrows():
                if row['id'] == -1:
                    detections_frame.loc[i, 'id'] = detections_frame['id'].max() + 1
            
            # Update the IDs in the original input DataFrame
            df.loc[df['frame'] == frame + 1, 'id'] = detections_frame['id'].values

        return df
    
    def get_tracks(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Gets the tracks of a DataFrame.
        Args:
            df: A pandas DataFrame containing the detections.
            n: The number of frames.
        Returns:
            A pandas DataFrame containing the tracks.
        """
        return df[df['frame'] == n]
    
    def get_detections(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Gets the detections of a DataFrame.
        Args:
            df: A pandas DataFrame containing the detections.
            n: The number of frames.
        Returns:
            A pandas DataFrame containing the detections.
        """
        return df[df['frame'] == n + 1]