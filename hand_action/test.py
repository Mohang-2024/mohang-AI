import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import threading

# Define actions and sequence length
actions = ['hungry', 'sleepy', 'Language_translation', 'go_anywhere', 'hurts']
seq_length = 30

# Load the TensorFlow model
model = load_model('models/model.keras')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the video capture
cap = cv2.VideoCapture(0)

seq = []
action_seq = []
this_action = None

# 프레임 스킵을 통해 처리 부하를 줄이는 코드
frame_skip = 2  # 2 프레임마다 1번 처리
frame_count = 0

def process_frame(frame):
    global seq, action_seq, this_action

    # Flip and preprocess the frame for model
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)

            # seq_length에 맞는 데이터가 쌓였을 때 예측
            if len(seq) < seq_length:
                return frame

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            # 모델 추론을 수행
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf >= 0.9:
                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) >= 3 and action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    if this_action:
        cv2.putText(frame, f'{this_action.upper()}', (int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(this_action)

    return frame

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # 프레임 스킵 처리
    if frame_count % frame_skip == 0:
        thread = threading.Thread(target=process_frame, args=(img,))
        thread.start()

    frame_count += 1

    # 보여주는 부분은 바로 렌더링
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
