import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sd_meas): 
        self.u = [u_x, u_y]
        self.x_k = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[(dt**2)/2, 0], [0, (dt**2)/2], [dt, 0], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                           [0, (dt**4)/4, 0, (dt**3)/2],
                           [(dt**3)/2, 0, dt**2, 0],
                           [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        self.R = np.array([[x_sdt_meas**2, 0], [0, y_sd_meas**2]])
        self.P = np.eye(self.A.shape[1])
        self.I = np.eye(self.A.shape[1])

    def predict(self):
        x_k_minus = np.dot(self.A, self.x_k) + np.dot(self.B, self.u)
        P_minus = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return x_k_minus, P_minus

    def update(self, z_k):
        x_k_minus, P_minus = self.predict()
        S_k = np.dot(np.dot(self.H, P_minus), self.H.T) + self.R
        K_k = np.dot(np.dot(P_minus, self.H.T), np.linalg.inv(S_k))

        self.x_k = x_k_minus + np.dot(K_k, (z_k - np.dot(self.H, x_k_minus)))
        self.P = np.dot((self.I - np.dot(K_k, self.H)), P_minus)