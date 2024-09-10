import pygame
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt


class Slider:
    def __init__(self, x, y, w, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.handle_rect = pygame.Rect(
            x + int((initial_val - min_val) / (max_val - min_val) * w) - 5,
            y - 5,
            10,
            20,
        )
        self.dragging = False
        self.label = label

    def draw(self, screen):
        pygame.draw.rect(screen, (100, 100, 100), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.handle_rect)
        font = pygame.font.Font(None, 24)
        label_surface = font.render(
            f"{self.label}: {self.val:.2f}", True, (255, 255, 255)
        )
        screen.blit(label_surface, (self.rect.x, self.rect.y - 30))

    def update(self, mouse_pos, mouse_pressed):
        if mouse_pressed and self.rect.collidepoint(mouse_pos):
            self.dragging = True
        if not mouse_pressed:
            self.dragging = False

        if self.dragging:
            new_x = min(max(mouse_pos[0], self.rect.x), self.rect.x + self.rect.w)
            self.handle_rect.x = new_x - 5
            self.val = self.min_val + (new_x - self.rect.x) / self.rect.w * (
                self.max_val - self.min_val
            )


# Pygame 관련 설정
def pygame_process(shared_velocities, screen_width=1000, screen_height=1000):
    # Pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    # 슬라이더 초기화 (상단 배치)
    sliders = [
        Slider(50, 50, 400, 0.01, 2.0, 0.05, "Force Constant"),  # Force Constant
        Slider(50, 100, 400, 0.1, 1.0, 0.9, "Damping"),  # Damping
        Slider(50, 150, 400, 1.0, 50.0, 20.0, "Max Speed"),  # Max Speed
        Slider(50, 200, 400, 0.1, 10.0, 2.0, "Stop Distance"),  # Stop Distance
    ]

    # 자석 A 초기화 (윈도우 중앙)
    position_a = np.array([screen_width // 2, screen_height // 2], dtype=float)
    velocity_a = np.array([0.0, 0.0], dtype=float)

    # 마우스 관련 변수
    mouse_pressed = False
    mouse_start_pos = np.array([screen_width // 2, screen_height // 2], dtype=float)
    mouse_position = np.array([screen_width // 2, screen_height // 2], dtype=float)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]

        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # 슬라이더 영역을 클릭할 때 타겟점이 이동하지 않도록 슬라이더 클릭 여부 체크
        slider_being_dragged = any(slider.dragging for slider in sliders)

        # 슬라이더 값 업데이트
        for slider in sliders:
            slider.update(mouse_pos, mouse_pressed)

        # 슬라이더 값 읽기
        force_constant = sliders[0].val
        damping = sliders[1].val
        max_speed = sliders[2].val
        stop_distance = sliders[3].val

        # 자석 A의 움직임 처리
        if mouse_pressed and not slider_being_dragged:
            current_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
            mouse_position += current_mouse_pos - mouse_start_pos
            mouse_start_pos = current_mouse_pos

        # 자석 A와 B 사이의 거리 벡터 및 거리 계산
        distance_vector = mouse_position - position_a
        distance = np.linalg.norm(distance_vector)

        # 자석들이 멈추는 조건
        if distance <= stop_distance:
            velocity_a = np.array([0.0, 0.0])  # 자석 A의 속도를 0으로 설정
        else:
            # 힘 계산 (비례 제어 기반)
            if distance > 0:
                force_direction = distance_vector / distance
                force_magnitude = force_constant * distance
                force = force_direction * force_magnitude
            else:
                force = np.array([0.0, 0.0])

            acceleration = force
            velocity_a += acceleration
            speed = np.linalg.norm(velocity_a)
            if speed > max_speed:
                velocity_a = velocity_a / speed * max_speed

            velocity_a *= damping

        position_a += velocity_a

        # `velocity_a`의 값을 공유된 리스트에 추가
        if len(shared_velocities) > 150:  # 최대 150개의 값만 유지
            shared_velocities.pop(0)
        shared_velocities.append(np.linalg.norm(velocity_a))

        # 화면 그리기
        screen.fill((0, 0, 0))  # 배경을 검은색으로 초기화
        pygame.draw.circle(screen, (255, 0, 0), position_a.astype(int), 20)  # 자석 A
        pygame.draw.circle(
            screen, (0, 255, 0), mouse_position.astype(int), 10
        )  # 자석 B

        # 슬라이더 그리기 (상단)
        for slider in sliders:
            slider.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# Matplotlib 관련 설정
def matplotlib_process(shared_velocities):
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()

    while True:
        ax.clear()
        ax.plot(shared_velocities)
        ax.set_title("Velocity over Time (Last 60 Frames)")
        ax.set_xlabel("Frame (Last 120)")
        ax.set_ylabel("Velocity")
        ax.set_ylim(
            0, max(shared_velocities) * 1.2 if shared_velocities else 1
        )  # 속도를 안정적으로 표현하기 위한 y축 범위 설정
        plt.draw()
        plt.pause(0.01)


# 메인 함수
if __name__ == "__main__":
    with mp.Manager() as manager:
        # 공유된 리스트 생성
        shared_velocities = manager.list()

        # 두 프로세스 생성
        pygame_proc = mp.Process(target=pygame_process, args=(shared_velocities,))
        matplotlib_proc = mp.Process(
            target=matplotlib_process, args=(shared_velocities,)
        )

        # 두 프로세스 시작
        pygame_proc.start()
        matplotlib_proc.start()

        # 두 프로세스가 끝날 때까지 대기
        pygame_proc.join()
        matplotlib_proc.join()
