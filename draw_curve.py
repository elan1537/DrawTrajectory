import argparse
import multiprocessing as mp
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pygame
import pygame_gui
from matplotlib.animation import FuncAnimation
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from vector import Vector

# Constants
RECENT_POINT = 10
SPEED_HISTORY_SIZE = 100  # 그래프에 표시할 속도 기록 수

# 초기 중심점 설정
start_point_offset = np.array([224.91, 109.15], dtype=np.float64)  # 오프셋 정의


def update_graphs(
    frame, speed_data, tcp_speed_data, total_tcp_speed, total_diff, ax1, ax2, ax3, ax4
):
    # 마우스 속도 그래프 업데이트
    ax1.cla()
    ax1.plot(speed_data, label="Mouse Speed")
    ax1.set_ylim(0, max(speed_data) * 1.1)  # 최대 속도에 맞춰 그래프 확대
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Speed (mm/s)")
    ax1.set_title("Real-time Mouse Speed")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # TCP 속도 그래프 업데이트 (x, y, z)
    ax2.cla()
    ax2.plot(tcp_speed_data[0], label="TCP X Speed")
    ax2.plot(tcp_speed_data[1], label="TCP Y Speed")
    ax2.plot(tcp_speed_data[2], label="TCP Z Speed")
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("TCP Speed (mm/s)")
    ax2.set_title("Real-time TCP Speed")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    # TCP 전체 속력 그래프 업데이트
    ax3.cla()
    ax3.plot(total_tcp_speed, label="Total TCP Speed", color="purple")
    ax3.set_ylim(0, max(total_tcp_speed) * 1.1)
    ax3.set_xlabel("Time (frames)")
    ax3.set_ylabel("Speed (mm/s)")
    ax3.set_title("Total TCP Speed")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    ax4.cla()
    ax4.plot(total_diff, label="Total Difference", color="red")
    ax4.set_ylim(0, max(total_diff) * 1.1)
    ax4.set_xlabel("Time (frames)")
    ax4.set_ylabel("Difference (mm)")
    ax4.set_title("Total Difference")
    ax4.legend(loc="upper right")
    ax4.grid(True)


def graph_process(
    speed_history, tcp_speed_history, total_tcp_speed_history, total_diff
):
    plt.ion()  # Interactive mode on to handle multiple figures

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))

    ani = FuncAnimation(
        fig,
        update_graphs,
        fargs=(
            speed_history,
            tcp_speed_history,
            total_tcp_speed_history,
            total_diff,
            ax1,
            ax2,
            ax3,
            ax4,
        ),
        interval=100,
    )

    plt.tight_layout()
    plt.show(block=True)  # Show figures in separate windows


def handle_events(
    drag_activate,
    slider_active,
    last_position,
    last_points,
    trajectory,
    manager,
    start_point_offset,
):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, drag_activate, slider_active, last_position

        # 키보드 이벤트 처리
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # 'q' 키를 눌렀을 때
                return False, drag_activate, slider_active, last_position

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 슬라이더가 클릭된 경우 드래그를 비활성화
            if (
                velocity_slider.rect.collidepoint(event.pos)
                or acceleration_slider.rect.collidepoint(event.pos)
                or dt_slider.rect.collidepoint(event.pos)
                or lookahead_slider.rect.collidepoint(event.pos)
                or gain_slider.rect.collidepoint(event.pos)
                or drag_speed_slider.rect.collidepoint(event.pos)
                or max_speed_slider.rect.collidepoint(event.pos)
                or sensitivity_slider.rect.collidepoint(event.pos)
            ):
                slider_active = True
            else:
                drag_activate = True
                last_points.clear()
                last_position = event.pos  # 마우스가 눌렸을 때 초기 위치 설정

        elif event.type == pygame.MOUSEMOTION:
            if drag_activate and not slider_active:
                last_points.append(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if slider_active:
                slider_active = False
            if event.button == 1:
                drag_activate = False

        manager.process_events(event)

    return True, drag_activate, slider_active, last_position, last_points


def main_loop(
    robot_ip,
    initial_velocity,
    initial_acceleration,
    speed_history,
    tcp_speed_history,
    total_tcp_speed_history,
    total_diff,
):
    # Pygame 초기 설정
    pygame.init()
    h, w = 1500, 1500
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("TCP Speed Visualization")
    clock = pygame.time.Clock()

    # Pygame GUI 설정
    manager = pygame_gui.UIManager((w, h))

    global velocity_slider, acceleration_slider, dt_slider, lookahead_slider, gain_slider, drag_speed_slider, max_speed_slider, sensitivity_slider

    # 슬라이더 생성
    velocity_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((50, 100), (300, 50)),
        start_value=initial_velocity,
        value_range=(0.0, 1.0),
        manager=manager,
    )

    acceleration_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((50, 160), (300, 50)),
        start_value=initial_acceleration,
        value_range=(0.0, 1.0),
        manager=manager,
    )

    dt_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((50, 220), (300, 50)),
        start_value=1 / 60,
        value_range=(0.008, 0.1),
        manager=manager,
    )

    lookahead_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((50, 280), (300, 50)),
        start_value=1 / 60 * 10,
        value_range=(0.03, 0.2),
        manager=manager,
    )

    gain_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((50, 340), (300, 50)),
        start_value=100,
        value_range=(100, 2000),
        manager=manager,
    )

    # 드래그 속도 조절 슬라이더
    drag_speed_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((50, 400), (300, 50)),
        start_value=1.0,
        value_range=(0.1, 5.0),
        manager=manager,
    )

    # 최대 속도 조절 슬라이더
    max_speed_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((50, 460), (300, 50)),
        start_value=10.0,
        value_range=(1.0, 20.0),
        manager=manager,
    )

    # 감도 조절 슬라이더
    sensitivity_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((50, 520), (300, 50)),
        start_value=1.0,
        value_range=(0.1, 3.0),
        manager=manager,
    )

    # Colors
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)

    # Global variables
    drag_activate = False
    slider_active = False
    last_position = None
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

        running, drag_activate, slider_active, last_position, last_points = (
            handle_events(
                drag_activate,
                slider_active,
                last_position,
                last_points,
                trajectory,
                manager,
                start_point_offset,
            )
        )

        time_delta = clock.tick(60) / 1000.0
        manager.update(time_delta)

        # 슬라이더로부터 velocity, acceleration, dt, lookahead_time, gain 값 읽기
        velocity = velocity_slider.get_current_value()
        acceleration = acceleration_slider.get_current_value()
        dt = dt_slider.get_current_value()
        lookahead_time = lookahead_slider.get_current_value()
        gain = gain_slider.get_current_value()
        lookahead_time = lookahead_slider.get_current_value()
        drag_speed_factor = drag_speed_slider.get_current_value()  # 드래그 속도 조절
        max_speed = max_speed_slider.get_current_value()  # 최대 속도 조절
        sensitivity = sensitivity_slider.get_current_value()  # 감도 조절

        if drag_activate and last_position is not None:
            # 마우스 이동 거리 계산
            p1 = last_position
            p2 = pygame.mouse.get_pos()

            if p1 != p2:  # 마우스가 움직였을 때만 속도 계산
                v_tot = Vector(
                    (p2[0] - p1[0]) * sensitivity, (p2[1] - p1[1]) * sensitivity
                )
                last_position = p2

                # 이동 속도 계산 (픽셀 단위 속도)
                speed = (v_tot.dist() * drag_speed_factor) / 10  # 드래그 속도 조절
                print(speed)

                # 속도 기록 업데이트
                speed_history.append(speed)

                # 속도 제한
                if speed > max_speed:
                    speed = max_speed

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

        tcp_speed_history[0].append(tcp_speed_vector[0])
        tcp_speed_history[1].append(tcp_speed_vector[1])
        tcp_speed_history[2].append(tcp_speed_vector[2])

        total_speed = np.linalg.norm(tcp_speed_vector)
        total_tcp_speed_history.append(total_speed)

        pygame.draw.circle(screen, GREEN, tuple(current_point.astype(int)), 5)

        for p in device_points:
            pygame.draw.circle(screen, RED, tuple(p), 2)

        for p in trajectory:
            pygame.draw.circle(screen, WHITE, p, 1)

        robot_point = np.array(
            [current_point[0] - w // 2, -(current_point[1] - h // 2)]
        )
        to_ = [robot_point[0] * 0.001, robot_point[1] * 0.001, 0.0, 2.223, -2.222, 0.0]

        x, y = rtde_r.getActualTCPPose()[:2]
        diff = np.linalg.norm((robot_point[0] / 1000 - x, robot_point[1] / 1000 - y))
        device_points.add((int(x * 1000) + w // 2, -int(y * 1000) + h // 2))
        total_diff.append(diff)

        if diff > 0 and time.time() - init_period > 1 / 60:
            joint_q = rtde_c.getInverseKinematics(to_)
            # acc, vel, t, lookahead_time, gain
            rtde_c.servoJ(joint_q, acceleration, velocity, dt, lookahead_time, gain)

        init_period = time.time()

        speed_text = font.render(f"Mouse Speed: {speed:.2f} mm/s", True, WHITE)
        tcp_speed_text = font.render(
            f"TCP Speed: {np.linalg.norm(tcp_speed_vector):.2f} mm/s", True, WHITE
        )
        not_allowed_text = font.render(f"Not Allowed", True, WHITE)
        screen.blit(speed_text, (10, 10))
        screen.blit(tcp_speed_text, (10, 40))
        screen.blit(not_allowed_text, (w // 2 - 45, h // 2 - 4))

        # 슬라이더 값 텍스트
        velocity_text = font.render(f"Velocity: {velocity:.3f} m/s", True, WHITE)
        acceleration_text = font.render(
            f"Acceleration: {acceleration:.3f} m/s²", True, WHITE
        )
        dt_text = font.render(f"dt: {dt:.3f} s", True, WHITE)
        lookahead_text = font.render(f"Lookahead: {lookahead_time:.3f} s", True, WHITE)
        gain_text = font.render(f"Gain: {int(gain)}", True, WHITE)
        drag_speed_text = font.render(
            f"Drag Speed: {drag_speed_factor:.1f}", True, WHITE
        )
        max_speed_text = font.render(f"Max Speed: {max_speed:.1f} mm/s", True, WHITE)
        sensitivity_text = font.render(f"Sensitivity: {sensitivity:.1f}", True, WHITE)
        screen.blit(velocity_text, (360, 95))
        screen.blit(acceleration_text, (360, 155))
        screen.blit(dt_text, (360, 215))
        screen.blit(lookahead_text, (360, 275))
        screen.blit(gain_text, (360, 335))
        screen.blit(drag_speed_text, (360, 395))
        screen.blit(max_speed_text, (360, 455))
        screen.blit(sensitivity_text, (360, 515))

        manager.draw_ui(screen)
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
    tcp_speed_history = [
        mp_manager.list([0] * SPEED_HISTORY_SIZE) for _ in range(3)
    ]  # x, y, z 각각의 속도를 저장
    total_tcp_speed_history = mp_manager.list(
        [0] * SPEED_HISTORY_SIZE
    )  # 전체 TCP 속력을 저장

    total_diff = mp_manager.list([0] * SPEED_HISTORY_SIZE)

    # 그래프를 그리는 프로세스 시작
    graph_proc = mp.Process(
        target=graph_process,
        args=(speed_history, tcp_speed_history, total_tcp_speed_history, total_diff),
    )
    graph_proc.start()

    main_loop(
        robot_ip=args.ip,
        initial_velocity=args.velocity,
        initial_acceleration=args.acceleration,
        speed_history=speed_history,
        tcp_speed_history=tcp_speed_history,
        total_tcp_speed_history=total_tcp_speed_history,
        total_diff=total_diff,
    )

    # 프로세스가 종료되기를 기다림
    graph_proc.join()
