import pygame
import numpy as np


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
        if mouse_pressed and self.handle_rect.collidepoint(mouse_pos):
            self.dragging = True
        if not mouse_pressed:
            self.dragging = False

        if self.dragging:
            new_x = min(max(mouse_pos[0], self.rect.x), self.rect.x + self.rect.w)
            self.handle_rect.x = new_x - 5
            self.val = self.min_val + (new_x - self.rect.x) / self.rect.w * (
                self.max_val - self.min_val
            )

    def get_value(self):
        return self.val


def get_mouse_state():
    return (
        np.array(pygame.mouse.get_pos(), dtype=float),
        pygame.mouse.get_pressed()[0],
    )


def get_slider_values(sliders):
    return sliders[0].get_value(), sliders[1].get_value(), sliders[2].get_value()


def handle_events(state, mouse_pos, mouse_pressed):
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if not state.slider_area.collidepoint(mouse_pos):
                state.dragging = True
                state.mouse_offset = state.target_pos - mouse_pos
            else:
                for slider in state.sliders:
                    slider.update(mouse_pos, mouse_pressed)
        elif event.type == pygame.MOUSEBUTTONUP:
            state.dragging = False
            for slider in state.sliders:
                slider.update(mouse_pos, mouse_pressed)
        elif event.type == pygame.MOUSEMOTION:
            if state.slider_area.collidepoint(mouse_pos):
                for slider in state.sliders:
                    slider.update(mouse_pos, mouse_pressed)


def draw_velocities(screen, state, font):
    target_speed = np.linalg.norm(state.target_vel)
    tcp_speed = np.linalg.norm(state.v)

    target_text = font.render(
        f"Target Speed: {target_speed / state.pixels_per_meter:.2f} m/s",
        True,
        (0, 255, 0),
    )
    tcp_text = font.render(
        f"TCP Speed: {tcp_speed / state.pixels_per_meter:.2f} m/s", True, (255, 0, 0)
    )

    screen.blit(target_text, (10, 200))
    screen.blit(tcp_text, (10, 230))


def draw_screen(screen, state, font):
    pygame.draw.rect(screen, (50, 50, 50), state.slider_area)
    pygame.draw.circle(screen, (255, 0, 0), state.tcp_pos.astype(int), 20)
    pygame.draw.circle(screen, (0, 255, 0), state.target_pos.astype(int), 10)

    k_text = font.render(
        f"K: {state.sliders[0].get_value():.2f}", True, (255, 255, 255)
    )
    screen.blit(k_text, (150, 10))

    for slider in state.sliders:
        slider.draw(screen)

    # 속도 정보 그리기
    draw_velocities(screen, state, font)
