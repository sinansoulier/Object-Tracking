import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image as Img

from src.utils.bounding_box import BoundingBox

class TrackerUtils:
    """
    Utility class for the trackers.
    """
    @staticmethod
    def IoU(bbox_1: BoundingBox, bbox_2: BoundingBox) -> float:
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.
        Args:
            bbox_1 (BoundingBox): Object representing the bounding box, with attributes left, top, width, and height.
            bbox_2 (BoundingBox): Object representing the bounding box, with attributes left, top, width, and height.
        Returns:
            (float): The IoU of the two bounding boxes.
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
            bbox_list_left (list[BoundingBox]): A list of bounding boxes.
            bbox_list_right (list[BoundingBox]): A list of bounding boxes.
        Returns:
            (np.ndarray): A numpy array containing the similarity matrix.
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
            (list[BoundingBox]): A list of bounding boxes.
        """
        return list(map(lambda x: BoundingBox(x[0], x[1], x[2], x[3]),
                        df[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values))

    @staticmethod
    def preprocess_patch(patch: Img.Image) -> torch.Tensor:
        """
        Preprocesses a patch.
        Args:
            patch (np.ndarray): The patch.
        Returns:
            (torch.Tensor): The preprocessed patch.
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
        ])

        patch = transform(patch)
        patch = torch.unsqueeze(patch, 0)

        return patch

    @staticmethod
    def extract_deep_features_from_patches(list_patches: list[torch.Tensor]) -> np.ndarray:
        """
        Extracts deep features from a patch.
        Args:
            patch (np.ndarray): The patch.
        Returns:
            (np.ndarray): The deep features.
        """
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        model.eval()

        batch = torch.cat(list_patches, dim=0).to(device)

        with torch.no_grad():
            features = model(batch)
        
        np_features = features.cpu().numpy()
        return np_features.reshape((np_features.shape[0], -1))

    @staticmethod
    def feature_similarity(feature_1: np.ndarray, feature_2: np.ndarray) -> float:
        """
        Calculates the feature similarity between two features.
        Args:
            feature_1 (np.ndarray): The first feature.
            feature_2 (np.ndarray): The second feature.
        Returns:
            (float): The feature similarity.
        """
        return 1 - cosine(feature_1, feature_2)

    @staticmethod
    def feature_similarity_matrix(feature_list_left: list[np.ndarray], feature_list_right: list[np.ndarray]) -> np.ndarray:
        """
        Calculates the feature similarity matrix between two lists of features.
        Args:
            feature_list_left (list[np.ndarray]): A list of features.
            feature_list_right (list[np.ndarray]): A list of features.
        Returns:
            (np.ndarray): A numpy array containing the feature similarity matrix.
        """
        sim_matrix = np.zeros((len(feature_list_left), len(feature_list_right)))
        for (i, feature_left) in enumerate(feature_list_left):
            for (j, feature_right) in enumerate(feature_list_right):
                sim_matrix[i, j] = TrackerUtils.feature_similarity(feature_left, feature_right)
        return sim_matrix

    def combined_similarity_matrices(similarity_matrix: np.ndarray, feature_similarity_matrix: np.ndarray, weights: list = [1, 1]) -> np.ndarray:
        """
        Combines the similarity matrix and the feature similarity matrix.
        Args:
            similarity_matrix (np.ndarray): The similarity matrix.
            feature_similarity_matrix (np.ndarray): The feature similarity matrix.
        Returns:
            (np.ndarray): The combined similarity matrix.
        """
        return (weights[0] * similarity_matrix + weights[1] * feature_similarity_matrix) / sum(weights)