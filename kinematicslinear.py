import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Diff-drive robot: wheel dynamics + PID + kinematics + animation + plots
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


def simulate(v_cmd, w_cmd, T=10.0, dt=0.01):
    # Geometry
    r = 0.05   # wheel radius [m]
    b = 0.15   # half wheelbase [m]

    # Wheel dynamics: w_dot = (K*u - B*w)/J
    J = 0.02
    B = 0.1
    K = 0.5
    u_limit = 5.0

    pid_L = PID(8.0, 10.0, 0.1, -u_limit, u_limit)
    pid_R = PID(8.0, 10.0, 0.1, -u_limit, u_limit)

    x, y, th = 0.0, 0.0, 0.0
    wL, wR = 0.0, 0.0

    N = int(T / dt)
    t = np.arange(N) * dt

    xs = np.zeros(N); ys = np.zeros(N); ths = np.zeros(N)
    v_log = np.zeros(N); w_log = np.zeros(N)

    wL_log = np.zeros(N); wR_log = np.zeros(N)
    wL_des_log = np.zeros(N); wR_des_log = np.zeros(N)
    uL_log = np.zeros(N); uR_log = np.zeros(N)

    for k in range(N):
        # Inverse kinematics -> desired wheel speeds
        wR_des = (v_cmd + b * w_cmd) / r
        wL_des = (v_cmd - b * w_cmd) / r

        # PID wheel-speed control
        uR = pid_R.step(wR_des - wR, dt)
        uL = pid_L.step(wL_des - wL, dt)

        # Wheel dynamics update
        wR += ((K * uR - B * wR) / J) * dt
        wL += ((K * uL - B * wL) / J) * dt

        # Forward kinematics -> body velocities
        v = (r / 2.0) * (wR + wL)
        w = (r / (2.0 * b)) * (wR - wL)

        # Pose integration
        x  += v * np.cos(th) * dt
        y  += v * np.sin(th) * dt
        th += w * dt

        xs[k], ys[k], ths[k] = x, y, th
        v_log[k], w_log[k] = v, w

        wL_log[k], wR_log[k] = wL, wR
        wL_des_log[k], wR_des_log[k] = wL_des, wR_des
        uL_log[k], uR_log[k] = uL, uR

    logs = {
        "t": t,
        "x": xs, "y": ys, "th": ths,
        "v": v_log, "w": w_log,
        "wL": wL_log, "wR": wR_log,
        "wL_des": wL_des_log, "wR_des": wR_des_log,
        "uL": uL_log, "uR": uR_log
    }
    return logs


def triangle_points(x, y, th, L=0.08, W=0.05):
    """Small heading triangle for the robot."""
    tip = np.array([L, 0.0])
    left = np.array([-0.5 * L,  W / 2])
    right = np.array([-0.5 * L, -W / 2])
    P = np.vstack([tip, left, right])

    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    P = (P @ R.T) + np.array([x, y])
    return P


def make_plots(logs, v_cmd, w_cmd):
    t = logs["t"]

    # 1) Trajectory
    plt.figure()
    plt.title("Trajectory (x-y)")
    plt.plot(logs["x"], logs["y"])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.axis("equal")

    # 2) Pose vs time
    plt.figure()
    plt.title("Pose vs Time")
    plt.plot(t, logs["x"], label="x [m]")
    plt.plot(t, logs["y"], label="y [m]")
    plt.plot(t, logs["th"], label="θ [rad]")
    plt.xlabel("time [s]")
    plt.grid(True)
    plt.legend()

    # 3) Body velocity tracking
    plt.figure()
    plt.title("Body Velocities vs Time")
    plt.plot(t, logs["v"], label="v actual [m/s]")
    plt.plot(t, np.full_like(t, v_cmd), label="v command [m/s]")
    plt.plot(t, logs["w"], label="ω actual [rad/s]")
    plt.plot(t, np.full_like(t, w_cmd), label="ω command [rad/s]")
    plt.xlabel("time [s]")
    plt.grid(True)
    plt.legend()

    # 4) Wheel speed tracking
    plt.figure()
    plt.title("Wheel Speeds vs Time")
    plt.plot(t, logs["wL"], label="ω_L actual [rad/s]")
    plt.plot(t, logs["wL_des"], label="ω_L desired [rad/s]")
    plt.plot(t, logs["wR"], label="ω_R actual [rad/s]")
    plt.plot(t, logs["wR_des"], label="ω_R desired [rad/s]")
    plt.xlabel("time [s]")
    plt.grid(True)
    plt.legend()

    # 5) Wheel speed error
    plt.figure()
    plt.title("Wheel Speed Error vs Time")
    eL = logs["wL_des"] - logs["wL"]
    eR = logs["wR_des"] - logs["wR"]
    plt.plot(t, eL, label="e_L = ω_L_des - ω_L")
    plt.plot(t, eR, label="e_R = ω_R_des - ω_R")
    plt.xlabel("time [s]")
    plt.grid(True)
    plt.legend()

    # 6) Control input (u)
    plt.figure()
    plt.title("Control Inputs (u) vs Time")
    plt.plot(t, logs["uL"], label="u_L")
    plt.plot(t, logs["uR"], label="u_R")
    plt.xlabel("time [s]")
    plt.grid(True)
    plt.legend()


def animate_robot(logs, dt):
    xs, ys, ths = logs["x"], logs["y"], logs["th"]
    v_log, w_log = logs["v"], logs["w"]

    fig, ax = plt.subplots()
    ax.set_title("Straight Line | PID + Wheel Dynamics (Animation)")
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

    step_skip = 2  # increase -> faster playback

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


if __name__ == "__main__":
    # ---- Commands ----
    v_cmd = 0.6   # [m/s]
    w_cmd = 0.0   # [rad/s]
    T = 10.0
    dt = 0.01

    logs = simulate(v_cmd=v_cmd, w_cmd=w_cmd, T=T, dt=dt)

    # Show animation first (close the animation window to continue)
    animate_robot(logs, dt)

    # Then show plots
    make_plots(logs, v_cmd=v_cmd, w_cmd=w_cmd)
    plt.show()

