import cv2
import numpy as np
import time
from collections import deque
from vector import Vector
from point import Point

# Constants
DT = 0.01
RECENT_POINT = 10

# Global variables
init_point = None
start_point = None
middle_target = None
target_point = None
now_p = None
trajectories = deque()
queue_point = deque()
is_drawing_start = False


def canvas_motion_event(event, x, y, flags, param):
    global is_drawing_start, queue_point, img, init_point, middle_target, target_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_drawing_start:
            is_drawing_start = True
            init_point = Point(x, y)
            middle_target = None
            target_point = None
        else:
            is_drawing_start = False
            queue_point.clear()
            target_point = Point(x, y)

    if is_drawing_start:
        queue_point.append(Point(x, y))


def position(point: Point, v: Vector, t):
    return Point(point.x + v.x * (t), point.y + v.y * (t))


def direction(points, normalize=True):
    if not points:
        return Vector(0, 0)

    v_tot = Point.to_vector(points[0], points[-1])
    return v_tot.normalize() if normalize else v_tot


def render(canvas):
    global target_point
    canvas.fill(0)

    for point in queue_point:
        cv2.circle(canvas, tuple(point), 3, (255, 255, 255), -1)

    for i in range(len(queue_point) - 1):
        p1, p2 = queue_point[i], queue_point[i + 1]
        cv2.line(canvas, tuple(p1), tuple(p2), (255, 255, 255), 1)

    if target_point:
        cv2.circle(canvas, (target_point.x, target_point.y), 25, (255, 0, 255), -1)

    if now_p:
        cv2.circle(canvas, (int(now_p.x), int(now_p.y)), 25, (0, 0, 255), -1)

    for tra in trajectories:
        cv2.circle(canvas, (int(tra.x), int(tra.y)), 3, (10, 255, 10), -1)

    cv2.imshow("Canvas", canvas)


def update_positions(diff):
    global now_p, init_point, start_point, middle_target, target_point, queue_point

    if len(queue_point) > RECENT_POINT:
        queue_point.popleft()

    if init_point:
        now_p = init_point
        start_point = init_point
        init_point = None

    if target_point:
        to_dir = direction([now_p, target_point], normalize=False)
        now_p = position(now_p, to_dir, diff)
        queue_point.clear()
    elif queue_point:
        middle_target = queue_point[-1]
        to_dir = direction([now_p, middle_target], normalize=False)
    else:
        to_dir = direction(queue_point, normalize=False)

    if now_p:
        now_p = position(now_p, to_dir, diff)
        trajectories.append(now_p)


if __name__ == "__main__":
    cv2.namedWindow("Canvas")
    cv2.setMouseCallback("Canvas", canvas_motion_event)

    start_time = time.time()
    canvas = np.zeros((2160, 3840, 3), np.uint8)

    while True:
        diff = time.time() - start_time
        if diff >= DT:
            update_positions(diff)
            render(canvas)
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
