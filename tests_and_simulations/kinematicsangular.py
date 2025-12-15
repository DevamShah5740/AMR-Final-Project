import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Diff-drive robot: wheel dynamics + PID + kinematics + animation
# Case 2: Circular motion (v=x, omega=y)
# -----------------------------

class PID:
    def __init__(self, kp, ki, kd, u_min=-np.inf, u_max=np.inf):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.u_min, self.u_max = u_min, u_max
        self.i = 0.0
        self.prev_e = 0.0

    def step(self, e, dt):
        self.i += e * dt
        de = (e - self.prev_e) / dt if dt > 0 else 0.0
        self.prev_e = e
        u = self.kp * e + self.ki * self.i + self.kd * de
        return float(np.clip(u, self.u_min, self.u_max))

def simulate(v_cmd, w_cmd, T=12.0, dt=0.01):
    r = 0.05
    b = 0.15

    J = 0.02
    B = 0.1
    K = 0.5
    u_limit = 5.0

    pid_L = PID(8.0, 10.0, 0.1, -u_limit, u_limit)
    pid_R = PID(8.0, 10.0, 0.1, -u_limit, u_limit)

    x, y, th = 0.0, 0.0, 0.0
    wL, wR = 0.0, 0.0

    N = int(T / dt)
    xs = np.zeros(N); ys = np.zeros(N); ths = np.zeros(N)
    v_log = np.zeros(N); w_log = np.zeros(N)

    for k in range(N):
        wR_des = (v_cmd + b * w_cmd) / r
        wL_des = (v_cmd - b * w_cmd) / r

        uR = pid_R.step(wR_des - wR, dt)
        uL = pid_L.step(wL_des - wL, dt)

        wR += ((K * uR - B * wR) / J) * dt
        wL += ((K * uL - B * wL) / J) * dt

        v = (r / 2.0) * (wR + wL)
        w = (r / (2.0 * b)) * (wR - wL)

        x  += v * np.cos(th) * dt
        y  += v * np.sin(th) * dt
        th += w * dt

        xs[k], ys[k], ths[k] = x, y, th
        v_log[k], w_log[k] = v, w

    return xs, ys, ths, v_log, w_log, dt

def triangle_points(x, y, th, L=0.08, W=0.05):
    tip = np.array([L, 0.0])
    left = np.array([-0.5 * L,  W / 2])
    right = np.array([-0.5 * L, -W / 2])
    P = np.vstack([tip, left, right])

    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    P = (P @ R.T) + np.array([x, y])
    return P

if __name__ == "__main__":
    # ---- Commands ----
    x_linear_velocity = 0.6  # v = x [m/s]
    y_angular_velocity = 0.5 # omega = y [rad/s]

    if y_angular_velocity != 0:
        print(f"Expected circle radius R ≈ v/ω = {x_linear_velocity/y_angular_velocity:.3f} m")

    xs, ys, ths, v_log, w_log, dt = simulate(
        v_cmd=x_linear_velocity,
        w_cmd=y_angular_velocity,
        T=12.0,
        dt=0.01
    )

    # ---- Animation ----
    fig, ax = plt.subplots()
    ax.set_title("Circular Motion (v=x, ω=y) | PID + Wheel Dynamics")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.axis("equal")

    pad = 0.4
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)

    path_line, = ax.plot([], [], lw=2)
    robot_patch = plt.Polygon(triangle_points(xs[0], ys[0], ths[0]), closed=True)
    ax.add_patch(robot_patch)

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    step_skip = 2

    def init():
        path_line.set_data([], [])
        robot_patch.set_xy(triangle_points(xs[0], ys[0], ths[0]))
        info.set_text("")
        return path_line, robot_patch, info

    def update(frame):
        i = frame * step_skip
        if i >= len(xs):
            i = len(xs) - 1

        path_line.set_data(xs[:i + 1], ys[:i + 1])
        robot_patch.set_xy(triangle_points(xs[i], ys[i], ths[i]))
        info.set_text(f"t={i*dt:.2f}s\nv={v_log[i]:.2f} m/s\nω={w_log[i]:.2f} rad/s")
        return path_line, robot_patch, info

    frames = (len(xs) + step_skip - 1) // step_skip
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=20, blit=True)

    plt.show()
