import pygame
import numpy as np

from collections import deque
from vector import Vector

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

import argparse, time


# Constants
RECENT_POINT = 10
DECAY_FACTOR = 0.5  # 속도가 감소하는 비율
MAX_SPEED = 10  # 속도의 최대값


# Pygame 초기 설정
pygame.init()
h, w = 1200, 1200
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("TCP Speed Visualization")
clock = pygame.time.Clock()


# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Global variables
drag_activate = False
x1, y1 = -1, -1
last_points = deque(maxlen=RECENT_POINT)
speed = 0  # 현재 속도를 저장할 변수

# 초기 중심점 설정
start_point = np.array(
    [w // 2 + 224.91, h // 2 + 109.15], dtype=np.float64
)  # 실수형 배열로 정의
current_point = start_point.copy()  # 현재 점의 위치


def main_loop(robot_ip, velocity, acceleration):
    global speed, current_point, drag_activate

    # Initialize RTDE interfaces
    rtde_c = RTDEControl(robot_ip)
    rtde_r = RTDEReceive(robot_ip)

    device_points = set()

    # Robot initialization
    acceleration = 0.5
    joint_q = np.deg2rad([0, -116.27, -162.21, 8.48, 90.0, 0])
    rtde_c.moveJ(joint_q, acceleration, velocity)

    init_period = time.time()

    running = True

    while running:
        screen.fill(BLACK)

        # 중앙 원 그리기
        pygame.draw.circle(screen, GRAY, (w // 2, h // 2), 250)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drag_activate = True
                    last_points.clear()
                    device_points.clear()

            elif event.type == pygame.MOUSEMOTION:
                if drag_activate:
                    last_points.append(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drag_activate = False

        if drag_activate and len(last_points) > 1:
            v_tot = Vector(0, 0)
            for idx in range(len(last_points) - 1):
                p1 = last_points[idx]
                p2 = last_points[idx + 1]
                v_tot += Vector(p2[0] - p1[0], p2[1] - p1[1])

            # 평균 벡터 계산
            l = v_tot.dist()
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

        # TCP 속도 가져오기
        tcp_speed_vector = rtde_r.getActualTCPSpeed()[:3]
        tcp_speed = np.linalg.norm(tcp_speed_vector) * 100

        pygame.draw.circle(screen, GREEN, tuple(current_point.astype(int)), 5)

        for p in device_points:
            pygame.draw.circle(screen, RED, tuple(p), 2)

        robot_point = np.array(
            [current_point[0] - w // 2, -(current_point[1] - h // 2)]
        )
        to_ = [robot_point[0] * 0.001, robot_point[1] * 0.001, 0.0, 2.223, -2.222, 0.0]

        if speed and (time.time() - init_period > 1 / 60):
            joint_q = rtde_c.getInverseKinematics(to_)
            # acc, vel, t, lookahead_time, gain
            rtde_c.servoJ(joint_q, acceleration, velocity, 1 / 60, 1 / 60 * 10, 150)

            x, y = rtde_r.getActualTCPPose()[:2]
            device_points.add((int(x * 1000) + w // 2, -int(y * 1000) + h // 2))

        init_period = time.time()

        # 속도 표시
        font = pygame.font.SysFont(None, 24)
        speed_text = font.render(f"Mouse Speed: {speed:.2f} mm/s", True, WHITE)
        tcp_speed_text = font.render(f"TCP Speed: {tcp_speed:.2f} mm/s", True, WHITE)
        not_allowed_text = font.render(f"Not Allowed", True, WHITE)
        screen.blit(speed_text, (10, 10))
        screen.blit(tcp_speed_text, (10, 40))
        screen.blit(not_allowed_text, (w // 2 - 30, h // 2 - 7))

        pygame.display.flip()
        clock.tick(60)  # 초당 60프레임으로 업데이트

    pygame.quit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Draw curve with Universal Robot")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.01)
    args = parser.parse_args()

    main_loop(robot_ip=args.ip, velocity=args.velocity, acceleration=args.acceleration)
