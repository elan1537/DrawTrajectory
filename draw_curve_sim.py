import pygame
import numpy as np

# Pygame 초기화
pygame.init()

# 화면 설정
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# 자석 A 초기화 (윈도우 중앙)
position_a = np.array([screen_width // 2, screen_height // 2], dtype=float)
velocity_a = np.array([0.0, 0.0], dtype=float)

# 상수 설정
force_constant = 1000  # 자석 간의 기본 힘 상수
mass_a = 1.0  # 자석 A의 질량
damping = 0.95  # 속도 감소율 (마찰을 모사하기 위해 사용)
max_speed = 7.0  # 최대 속도 제한
stop_distance = 5.0  # 자석들이 멈추는 임계 거리
n = 80.0  # 힘 조절 상수 (거리에 따른 빠른 접근 조절)

# 마우스 관련 변수
mouse_pressed = False
mouse_start_pos = np.array([screen_width // 2, screen_height // 2], dtype=float)
mouse_position = np.array([screen_width // 2, screen_height // 2], dtype=float)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            mouse_pressed = False
        elif event.type == pygame.K_q:
            running = False
            mouse_pressed = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pressed = True
            # 마우스 클릭 시작 위치
            mouse_start_pos = np.array(pygame.mouse.get_pos(), dtype=float)
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False

    if mouse_pressed:
        # 현재 마우스 위치 가져오기
        current_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)

        # 마우스의 상대적 움직임 계산
        relative_movement = current_mouse_pos - mouse_start_pos

        # 상대적 움직임을 누적하여 mouse_position 업데이트
        mouse_position += relative_movement

        # 업데이트 후, 새로운 마우스 시작 위치를 현재 위치로 설정
        mouse_start_pos = current_mouse_pos

        # 자석 A와 B 사이의 거리 벡터 및 거리 계산
        distance_vector = mouse_position - position_a
        distance = np.linalg.norm(distance_vector)

        # 자석들이 멈추는 조건
        if distance <= stop_distance:
            velocity_a = np.array([0.0, 0.0])  # 자석 A의 속도를 0으로 설정
        else:
            # 힘 계산 (로그 스케일 사용)
            if distance > 0:  # 거리 0을 방지
                force_direction = distance_vector / distance
                # 로그 스케일로 감속 조절
                if distance > n:
                    force_magnitude = force_constant / np.log(distance / n)
                else:
                    force_magnitude = force_constant / (distance / n)
                force = force_direction * force_magnitude
            else:
                force = np.array([0.0, 0.0])

            # 가속도 = 힘 / 질량
            acceleration = force / mass_a

            # 속도 업데이트 (가속도 반영)
            velocity_a += acceleration

            # 최대 속도 제한 적용
            speed = np.linalg.norm(velocity_a)
            if speed > max_speed:
                velocity_a = velocity_a / speed * max_speed

            # 속도 감소 (댐핑 효과)
            velocity_a *= damping

        # 위치 업데이트
        position_a += velocity_a

    # 화면 그리기
    screen.fill((0, 0, 0))  # 배경을 검은색으로 초기화
    pygame.draw.circle(screen, (255, 0, 0), position_a.astype(int), 20)  # 자석 A
    pygame.draw.circle(
        screen, (0, 255, 0), mouse_position.astype(int), 10
    )  # 자석 B (마우스)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
