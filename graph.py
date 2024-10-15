import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation


def matplotlib_process(pos_data, vel_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    pos_lines = ax1.plot([], [], "r-", [], [], "g-")
    vel_lines = ax2.plot([], [], "r-", [], [], "g-")

    ax1.set_ylim(-500, 500)
    ax2.set_ylim(-3, 3)
    ax1.set_title("TCP Position (relative to center)")
    ax2.set_title("TCP Velocity")

    ax1.legend(["X", "Y"])
    ax2.legend(["X", "Y"])

    plt.tight_layout()

    def init():
        for line in pos_lines + vel_lines:
            line.set_data([], [])
        return pos_lines + vel_lines

    def update(frame):
        if len(pos_data) > 0 and len(vel_data) > 0:
            pos_array = np.array(pos_data)
            vel_array = np.array(vel_data)

            # 데이터 길이를 일치시킵니다
            min_length = min(len(pos_array), len(vel_array))
            pos_array = pos_array[-min_length:]
            vel_array = vel_array[-min_length:]

            x = np.arange(min_length)

            pos_lines[0].set_data(x, pos_array[:, 0])
            pos_lines[1].set_data(x, pos_array[:, 1])
            vel_lines[0].set_data(x, vel_array[:, 0])
            vel_lines[1].set_data(x, vel_array[:, 1])

            for ax in (ax1, ax2):
                ax.relim()
                ax.autoscale_view()

        return pos_lines + vel_lines

    ani = FuncAnimation(
        fig, update, frames=None, init_func=init, blit=True, interval=50
    )
    plt.show()


def update_shared_data(pos_data, vel_data, tcp_pos, tcp_vel, center, max_length=150):
    pos_data.append((tcp_pos - center).tolist())
    vel_data.append((tcp_vel / 1000).tolist())

    while len(pos_data) > max_length:
        pos_data.pop(0)
    while len(vel_data) > max_length:
        vel_data.pop(0)
