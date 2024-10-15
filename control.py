import pygame
import numpy as np
import multiprocessing as mp
import time

from util import Slider

import graph
import util


class SimulationState:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tcp_pos = np.array([screen_width // 2, screen_height // 2], dtype=float)
        self.target_pos = self.tcp_pos.copy()
        self.v = np.array([0.0, 0.0], dtype=float)
        self.m = 1.0

        self.pixels_per_meter = 1000
        self.max_speed = 1.5 * self.pixels_per_meter
        self.max_accel = 15 * self.pixels_per_meter

        self.dragging = False
        self.mouse_offset = np.array([0.0, 0.0], dtype=float)
        self.sliders = [
            Slider(10, 50, 400, 0, 20, 6.25, "k"),
            Slider(10, 100, 400, 0, 20, 5, "c"),
            Slider(10, 150, 400, 0.0, 10.0, 0.0, "stop_distance"),
        ]
        self.slider_area = pygame.Rect(0, 0, 420, 200)
        self.mouse_vel = np.array([0.0, 0.0], dtype=float)
        self.prev_mouse_pos = np.array([0.0, 0.0], dtype=float)
        self.target_vel = np.array([0.0, 0.0], dtype=float)
        self.prev_target_pos = np.array(
            [screen_width // 2, screen_height // 2], dtype=float
        )


def update_simulation(state, mouse_pos, delta_t, **kwargs):
    k, c, stop_distance = util.get_slider_values(state.sliders)

    update_target_position(state, mouse_pos, delta_t, **kwargs)
    update_tcp_position(state, k, c, stop_distance, delta_t)


def update_target_position(state, mouse_pos, delta_t, screen_width, screen_height):
    if not state.dragging:
        state.target_vel = np.array([0.0, 0.0], dtype=float)
        state.prev_target_pos = state.target_pos.copy()
        return state.target_pos, state.target_vel

    temp_target_pos = np.array(mouse_pos, dtype=float) + state.mouse_offset
    temp_target_pos = np.clip(temp_target_pos, [0, 0], [screen_width, screen_height])

    if state.slider_area.collidepoint(temp_target_pos):
        state.target_vel = np.array([0.0, 0.0], dtype=float)
        return state.target_pos, state.target_vel

    state.target_pos = temp_target_pos
    state.target_vel = (state.target_pos - state.prev_target_pos) / delta_t
    target_vel_magnitude = np.linalg.norm(state.target_vel)
    if target_vel_magnitude > state.max_speed:
        state.target_vel = state.target_vel * (state.max_speed / target_vel_magnitude)

    state.prev_target_pos = state.target_pos.copy()


def update_tcp_position(state, k, c, stop_distance, delta_t):
    e = state.target_pos - state.tcp_pos
    de = state.target_vel - state.v

    if np.linalg.norm(e) <= stop_distance:
        state.tcp_pos = state.target_pos.copy()
        state.v = np.array([0.0, 0.0], dtype=float)
    else:
        a = (k * e + c * de) / state.m
        a_magnitude = np.linalg.norm(a)
        if a_magnitude > state.max_accel:
            a = a * (state.max_accel / a_magnitude)

        state.v += a * delta_t
        v_magnitude = np.linalg.norm(state.v)
        if v_magnitude > state.max_speed:
            state.v = state.v * (state.max_speed / v_magnitude)

        temp_tcp_pos = state.tcp_pos + state.v * delta_t
        if not state.slider_area.collidepoint(temp_tcp_pos):
            state.tcp_pos = temp_tcp_pos
        else:
            state.v = np.array([0.0, 0.0], dtype=float)


def main(pos_data, vel_data, screen_width=1000, screen_height=1000):
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    state = SimulationState(screen_width, screen_height)
    center = np.array([screen_width // 2, screen_height // 2], dtype=float)

    running = True
    while running:
        screen.fill((0, 0, 0))

        mouse_pos, mouse_pressed = util.get_mouse_state()

        util.handle_events(state, mouse_pos, mouse_pressed)

        delta_t = min(clock.tick(60) / 1000.0, 0.016)

        update_simulation(
            state,
            mouse_pos,
            delta_t,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        graph.update_shared_data(pos_data, vel_data, state.tcp_pos, state.v, center)

        util.draw_screen(screen, state, font)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    with mp.Manager() as manager:
        pos_data = manager.list()
        vel_data = manager.list()

        p1 = mp.Process(target=main, args=(pos_data, vel_data))
        p2 = mp.Process(target=graph.matplotlib_process, args=(pos_data, vel_data))

        p1.start()
        p2.start()

        p1.join()
        p2.join()
