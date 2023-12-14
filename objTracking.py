from src.KalmanFilter import KalmanFilter
from src.Detector import detect
import cv2

if __name__ == '__main__':
    # Create KalmanFilter object
    dt = 0.1
    u_x = 1
    u_y = 1
    std_acc = 1
    x_dt_meas =0.1
    y_dt_meas = 0.1

    # Create video capture object
    cap = cv2.VideoCapture("data/randomball.avi")

    trajectory = []

    while True:
        ret, frame = cap.read()
        centers = detect(frame)

        if (len(centers) > 0):
            for center in centers:
                cv2.circle(frame, (int(center[0]), int(center[1])), 12, (0, 0, 255), 4)

                trajectory.append(center)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break