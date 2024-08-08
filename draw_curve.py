import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from vector import Vector
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


import pygame_gui
import multiprocessing as mp
import argparse
import time

# Constants
RECENT_POINT = 10
MAX_SPEED = 20  # 속도의 최대값
SPEED_HISTORY_SIZE = 100  # 그래프에 표시할 속도 기록 수

# 초기 중심점 설정
start_point_offset = np.array([224.91, 109.15], dtype=np.float64)  # 오프셋 정의


def update_graph(frame, speed_data):
    plt.cla()
    plt.plot(speed_data, label="Mouse Speed")
    plt.ylim(0, MAX_SPEED)
    plt.xlabel("Time (frames)")
    plt.ylabel("Speed (mm/s)")
    plt.title("Real-time Mouse Speed")
    plt.legend(loc="upper right")
    plt.tight_layout()


def graph_process(speed_history):
    fig = plt.figure()
    ani = FuncAnimation(fig, update_graph, fargs=(speed_history,), interval=100)
    plt.show()


def main_loop(robot_ip, initial_velocity, initial_acceleration, speed_history):
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
    slider_active = False
    last_points = deque(maxlen=RECENT_POINT)
    trajectory = set()
    speed = 0  # 현재 속도를 저장할 변수

    # 초기 중심점 설정
    current_point = np.array([w // 2, h // 2]) + start_point_offset  # 초기 점 위치

    # 속도 표시
    font = pygame.font.SysFont(None, 24)
    moved = font.render(f"Move to initial point.", True, WHITE)
    screen.blit(moved, (w // 2 - 25, h // 2 - 4))

    device_points = set()

    # Robot initialization
    # 주석 처리되어 있지만 실제 로봇이 있다면 활성화
    rtde_c = RTDEControl(robot_ip)
    rtde_r = RTDEReceive(robot_ip)
    joint_q = np.deg2rad([0, -116.27, -162.21, 8.48, 90.0, 0])
    rtde_c.moveJ(joint_q, 0.5, 0.5)

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
                # 슬라이더가 클릭된 경우 드래그를 비활성화
                drag_activate = True
                last_points.clear()
                device_points.clear()
                last_position = event.pos  # 마우스가 눌렸을 때 초기 위치 설정

            elif event.type == pygame.MOUSEMOTION:
                if drag_activate and not slider_active:
                    last_points.append(event.pos)
                    trajectory.add((event.pos[0] + 224.91, event.pos[1] + 109.15))

            elif event.type == pygame.MOUSEBUTTONUP:
                if slider_active:
                    slider_active = False
                if event.button == 1:
                    drag_activate = False

        if drag_activate and last_position is not None:
            # 마우스 이동 거리 계산
            p1 = last_position
            p2 = pygame.mouse.get_pos()

            if p1 != p2:  # 마우스가 움직였을 때만 속도 계산
                v_tot = Vector(p2[0] - p1[0], p2[1] - p1[1])
                last_position = p2

                # 이동 속도 계산 (픽셀 단위 속도)
                speed = v_tot.dist()

                # 속도 기록 업데이트
                speed_history.append(speed)

                # 속도 제한
                if speed > MAX_SPEED:
                    speed = MAX_SPEED

                if speed != 0:
                    v_normal = Vector(v_tot.x / speed, v_tot.y / speed)

                # 현재 위치 업데이트
                current_point += np.array([v_normal.x * speed, v_normal.y * speed])

                # 위치가 화면을 벗어나지 않도록 조정
                current_point = np.clip(current_point, [0, 0], [w, h])

            else:
                # 마우스가 움직이지 않으면 속도 0으로 설정
                speed = 0
                speed_history.append(speed)

        else:
            # 드래그가 활성화되지 않은 경우 속도 기록을 0으로
            speed = 0
            speed_history.append(speed)

        trajectory.add(tuple(current_point))
        # TCP 속도 가져오기
        tcp_speed_vector = rtde_r.getActualTCPSpeed()[:3]
        tcp_speed = np.linalg.norm(tcp_speed_vector) * 100

        pygame.draw.circle(screen, GREEN, tuple(current_point.astype(int)), 5)

        for p in device_points:
            pygame.draw.circle(screen, RED, tuple(p), 2)

        for p in trajectory:
            pygame.draw.circle(screen, WHITE, p, 1)

        robot_point = np.array(
            [current_point[0] - w // 2, -(current_point[1] - h // 2)]
        )
        to_ = [robot_point[0] * 0.001, robot_point[1] * 0.001, 0.0, 2.223, -2.222, 0.0]

        if speed and (time.time() - init_period > 1 / 60):
            joint_q = rtde_c.getInverseKinematics(to_)
            # acc, vel, t, lookahead_time, gain
            rtde_c.servoJ(
                joint_q,
                initial_acceleration,
                initial_velocity,
                1 / 60,
                1 / 60 * 10,
                150,
            )

            x, y = rtde_r.getActualTCPPose()[:2]
            device_points.add((int(x * 1000) + w // 2, -int(y * 1000) + h // 2))

        init_period = time.time()

        speed_text = font.render(f"Mouse Speed: {speed:.2f} mm/s", True, WHITE)
        tcp_speed_text = font.render(f"TCP Speed: {tcp_speed:.2f} mm/s", True, WHITE)
        not_allowed_text = font.render(f"Not Allowed", True, WHITE)
        screen.blit(speed_text, (10, 10))
        screen.blit(tcp_speed_text, (10, 40))
        screen.blit(not_allowed_text, (w // 2 - 45, h // 2 - 4))

        pygame.display.flip()
        clock.tick(60)  # 초당 60프레임으로 업데이트

    pygame.quit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Draw curve with Universal Robot")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.01)
    args = parser.parse_args()

    # 속도 기록을 위한 공유 리스트 생성
    mp_manager = mp.Manager()
    speed_history = mp_manager.list([0] * SPEED_HISTORY_SIZE)

    # 그래프를 그리는 프로세스 시작
    graph_proc = mp.Process(target=graph_process, args=(speed_history,))
    graph_proc.start()

    main_loop(
        robot_ip=args.ip,
        initial_velocity=args.velocity,
        initial_acceleration=args.acceleration,
        speed_history=speed_history,
    )

    # 프로세스가 종료되기를 기다림
    graph_proc.join()
