import numpy as np

from src.utils.point import Point

class KalmanFilter:
    """
    A class representing a Kalman filter.
    """
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas, center: Point = None):
        self.u = [u_x, u_y]
        if center is not None:
            self.x_k = np.array([[center.x], [center.y], [0], [0]])
        else:
            self.x_k = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[(dt**2)/2, 0], [0, (dt**2)/2], [dt, 0], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                           [0, (dt**4)/4, 0, (dt**3)/2],
                           [(dt**3)/2, 0, dt**2, 0],
                           [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        self.R = np.array([[x_sdt_meas**2, 0], [0, y_sdt_meas**2]])
        self.P = np.eye(self.A.shape[1])
        self.I = np.eye(self.A.shape[1])

    def predict(self):
        """
        Predicts the next state.
        """
        x_k_minus = np.dot(self.A, self.x_k) + np.dot(self.B, self.u)
        P_minus = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return x_k_minus, P_minus

    def update(self, z_k: list[float]):
        """
        Updates the state.
        Args:
            z_k (list[float]): The measurements.
        Returns:
            (np.ndarray): The updated state.
        """
        x_k_minus, P_minus = self.predict()
        S_k = np.dot(np.dot(self.H, P_minus), self.H.T) + self.R
        K_k = np.dot(np.dot(P_minus, self.H.T), np.linalg.inv(S_k))

        self.x_k = x_k_minus + np.dot(K_k, (z_k - np.dot(self.H, x_k_minus)))
        self.P = np.dot((self.I - np.dot(K_k, self.H)), P_minus)

        return self.x_k
    
    @staticmethod
    def new(center: Point) -> 'KalmanFilter':
        return KalmanFilter(
            dt=0.1,
            u_x=1,
            u_y=1,
            std_acc=1,
            x_sdt_meas=0.1,
            y_sdt_meas=0.1,
            center=center
        )