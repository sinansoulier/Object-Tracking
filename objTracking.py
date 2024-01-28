import time
import cv2

from src.KalmanFilter import KalmanFilter
from src.Detector import detect

if __name__ == '__main__':
    dt = 0.1
    u_x = 1
    u_y = 1
    std_acc = 1
    x_dt_meas =0.1
    y_dt_meas = 0.1
    kf = KalmanFilter(dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)

    cap = cv2.VideoCapture("data/randomball.avi")

    trajectory = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        centers = detect(frame)
        rect_padding = 12
       
        if (len(centers) > 0):
            x_meas, y_meas = centers[0]
            print(f"x_meas: {x_meas}, y_meas: {y_meas}")

            predicted_state = kf.predict()[0]
            estimated_state = kf.update([x_meas, y_meas])

            cv2.circle(frame, (int(x_meas), int(y_meas)), 13, (0, 255, 0), -1)

            cv2.rectangle(frame,
                          (int(predicted_state[0][0] - rect_padding), int(predicted_state[1][0]) - rect_padding),
                          (int(predicted_state[0][0] + 15), int(predicted_state[1][0] + 15)),
                          (255, 0, 0),
                          2)

            cv2.rectangle(frame,
                          (int(estimated_state[0][0] - rect_padding), int(estimated_state[1][0] - rect_padding)),
                          (int(estimated_state[0][0] + rect_padding), int(estimated_state[1][0] + rect_padding)),
                          (0, 0, 255),
                          2)

            trajectory.append((int(estimated_state[0][0]), int(estimated_state[1][0])))

            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 3)


        cv2.imshow('frame', frame)
        time.sleep(0.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()