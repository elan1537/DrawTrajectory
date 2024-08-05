import cv2
import numpy as np
from collections import deque
from vector import Vector  # 사용자 정의 Vector 클래스 사용

# Constants
RECENT_POINT = 10
DECAY_FACTOR = 0.9  # 속도가 감소하는 비율
MAX_SPEED = 20  # 속도의 최대값

# Global variables
drag_activate = False
h, w = 2000, 2000

canvas = np.zeros((h, w, 3), np.uint8)
x1, y1 = -1, -1
last_points = deque(maxlen=RECENT_POINT)
speed = 0  # 현재 속도를 저장할 변수

# 초기 중심점 설정
start_point = np.array([w // 2, h // 2], dtype=np.float64)  # 실수형 배열로 정의
current_point = start_point.copy()  # 현재 점의 위치


def canvas_motion_event(event, x, y, flags, param):
    global drag_activate, x1, y1, canvas, last_points, speed

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_activate = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_activate:
            last_points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drag_activate = False


cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", canvas_motion_event)


cv2.line(canvas, (0, h // 2), (w, h // 2), (255, 255, 255), 1)
cv2.line(canvas, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

while True:
    temp_canvas = canvas.copy()

    if drag_activate and len(last_points) > 1:
        v_tot = Vector(0, 0)
        for idx in range(len(last_points) - 1):
            p1 = last_points[idx]
            p2 = last_points[idx + 1]
            v_tot += Vector(p2[0] - p1[0], p2[1] - p1[1])

        l = v_tot.dist()
        # 평균 벡터 계산
        v_tot.x /= len(last_points) - 1
        v_tot.y /= len(last_points) - 1

        speed = l  # 드래그 중일 때 속도 업데이트

        # 속도 제한
        if speed > MAX_SPEED:
            speed = MAX_SPEED

        if l != 0:
            v_normal = Vector(v_tot.x / l, v_tot.y / l)

        # 현재 위치 업데이트
        current_point += np.array([v_normal.x * speed, v_normal.y * speed])

        # 위치가 화면을 벗어나지 않도록 조정
        current_point = np.clip(current_point, [0, 0], [w, h])

        # 라인과 텍스트를 temp_canvas에 그리기
        cv2.circle(temp_canvas, tuple(current_point.astype(int)), 5, (0, 255, 0), -1)
        last_points.clear()  # 점이 움직였으면 포인트 목록 초기화

    else:
        # 드래그가 끝난 후 속도 감소
        speed *= DECAY_FACTOR
        if speed < 0.1:  # 속도가 매우 작아지면 0으로 설정
            speed = 0

        # 속도와 방향에 따라 점의 위치 업데이트
        if speed > 0:
            current_point += np.array([v_normal.x * speed, v_normal.y * speed])
            current_point = np.clip(current_point, [0, 0], [w, h])

        # 현재 위치에 점 그리기
        cv2.circle(temp_canvas, tuple(current_point.astype(int)), 5, (0, 255, 0), -1)

    # 속도 표시
    cv2.putText(
        temp_canvas,
        f"Speed: {speed:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # 업데이트된 temp_canvas를 디스플레이
    cv2.imshow("Canvas", temp_canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
