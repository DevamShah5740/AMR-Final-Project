import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def unit(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def closest_inflated_distance(p, obstacles, buffer):
    """Distance to closest inflated obstacle boundary (r + buffer)."""
    best = np.inf
    for obs in obstacles:
        c = obs["c"]
        r = obs["r"] + buffer
        d = np.linalg.norm(p - c) - r
        best = min(best, d)
    return best

def field_force(p, goal, obstacles,
                buffer=0.9,
                k_att=2.0,
                k_rep=3.0,
                k_tan=0.25,
                rep_cap=8.0,
                tan_cap=4.0):
    """
    Potential field force with:
    - inflated obstacles = no-go zone (r + buffer)
    - capped repulsion so it won't overpower attraction far away
    """

    to_goal = goal - p
    F_att = k_att * to_goal

    F_rep = np.zeros(2)
    F_tan = np.zeros(2)
    ghat = unit(to_goal)

    for obs in obstacles:
        c = obs["c"]
        r_inf = obs["r"] + buffer  # inflated radius (dotted zone)

        vec = p - c
        dist_center = np.linalg.norm(vec)
        away = unit(vec)

        d = dist_center - r_inf  # distance to inflated boundary

        if d <= 0:
            # inside no-go: push out hard
            F_rep += 80.0 * away

            perp = np.array([-away[1], away[0]])
            sign = np.sign(np.cross(np.append(away, 0), np.append(ghat, 0))[2]) or 1.0
            F_tan += 20.0 * sign * perp
            continue

        # Repulsion strength: strong near boundary, decays away
        mag = k_rep * (1.0 / (d**2))

        # Cap repulsion magnitude per obstacle to keep goal attraction dominant
        mag = min(mag, rep_cap)

        F_rep += mag * away

        # Tangential (swirl) term: also capped
        perp = np.array([-away[1], away[0]])
        sign = np.sign(np.cross(np.append(away, 0), np.append(ghat, 0))[2]) or 1.0
        tmag = min(k_tan * mag, tan_cap)
        F_tan += tmag * sign * perp

    return F_att + F_rep + F_tan

def simulate(start=(0, 0), goal=(6, 6), obstacles=None,
             T=35.0, dt=0.02,
             v_max=1.0, w_max=2.5,
             buffer=0.9,
             goal_tol=0.05,
             final_approach_radius=1.0,
             safe_clearance=0.25):
    """
    final_approach_radius: within this distance of goal -> turn off swirl & soften repulsion
    safe_clearance: if too close to inflated obstacle near goal, keep avoidance on
    """

    if obstacles is None:
        obstacles = []
    start = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)

    x, y = start
    theta = 0.0

    N = int(T / dt)
    xs = np.zeros(N); ys = np.zeros(N); ths = np.zeros(N)
    vs = np.zeros(N); ws = np.zeros(N)
    dgoal = np.zeros(N)

    reached = False
    reach_idx = N - 1

    for k in range(N):
        p = np.array([x, y])
        to_goal = goal - p
        dist_goal = np.linalg.norm(to_goal)
        dgoal[k] = dist_goal

        # Stop exactly (within tolerance)
        if dist_goal <= goal_tol:
            reached = True
            reach_idx = k
            xs[k], ys[k], ths[k] = x, y, theta
            vs[k], ws[k] = 0.0, 0.0
            break

        # Check proximity to inflated obstacles
        d_obs = closest_inflated_distance(p, obstacles, buffer)

        # --------------------------
        # FINAL APPROACH MODE:
        # Close to goal AND not near inflated obstacles -> ignore swirl & strong repulsion
        # --------------------------
        final_mode = (dist_goal < final_approach_radius) and (d_obs > safe_clearance)

        if final_mode:
            # Pure go-to-goal direction
            des_theta = np.arctan2(to_goal[1], to_goal[0])
        else:
            # Normal obstacle avoidance
            F = field_force(
                p, goal, obstacles,
                buffer=buffer,
                k_att=2.0,
                k_rep=3.0,
                k_tan=0.25,
                rep_cap=8.0,
                tan_cap=4.0
            )
            des_dir = unit(F)
            des_theta = np.arctan2(des_dir[1], des_dir[0])

        # Heading control
        e = wrap_to_pi(des_theta - theta)
        omega = 3.2 * e
        omega = np.clip(omega, -w_max, w_max)

        # Speed control: do NOT let it stall completely while turning
        align = np.clip(np.cos(e), -1.0, 1.0)

        # base speed keeps moving; more when aligned
        v_min = 0.15  # key fix: prevents "spin and drift away"
        v = v_min + (v_max - v_min) * max(0.0, align)

        # Slow down near goal smoothly
        if dist_goal < 1.2:
            v *= dist_goal / 1.2

        # If in final mode, damp turning a bit and prioritize moving straight to target
        if final_mode:
            omega = np.clip(omega, -1.6, 1.6)

        # Integrate unicycle kinematics
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        xs[k], ys[k], ths[k] = x, y, theta
        vs[k], ws[k] = v, omega

    if reached:
        xs = xs[:reach_idx+1]
        ys = ys[:reach_idx+1]
        ths = ths[:reach_idx+1]
        vs = vs[:reach_idx+1]
        ws = ws[:reach_idx+1]
        dgoal = dgoal[:reach_idx+1]

    return xs, ys, ths, vs, ws, dgoal, reached

def triangle_points(x, y, th, L=0.10, W=0.06):
    tip = np.array([L, 0.0])
    left = np.array([-0.6 * L,  W / 2])
    right = np.array([-0.6 * L, -W / 2])
    P = np.vstack([tip, left, right])
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    return (P @ R.T) + np.array([x, y])

if __name__ == "__main__":
    start = (0.0, 0.0)
    goal  = (6.0, 6.0)

    obstacles = [
        {"c": np.array([2.6, 2.9]), "r": 0.55},
        {"c": np.array([4.1, 4.0]), "r": 0.65},
    ]

    buffer = 0.9  # dotted zone = no-go

    xs, ys, ths, vs, ws, dgoal, reached = simulate(
        start=start,
        goal=goal,
        obstacles=obstacles,
        T=35.0,
        dt=0.02,
        v_max=1.0,
        w_max=2.5,
        buffer=buffer,
        goal_tol=0.05,
        final_approach_radius=1.0,
        safe_clearance=0.25
    )

    print("Reached goal:", reached, "| Final dist:", float(dgoal[-1]))

    # ---- Animation ----
    fig, ax = plt.subplots()
    ax.set_title("Obstacle Avoidance (2 Obstacles) — Reaches Goal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(True); ax.axis("equal")

    ax.plot(start[0], start[1], "o", markersize=6)
    ax.plot(goal[0], goal[1], "*", markersize=12)

    for obs in obstacles:
        c, r0 = obs["c"], obs["r"]
        ax.add_patch(plt.Circle((c[0], c[1]), r0, fill=False, linewidth=2))
        ax.add_patch(plt.Circle((c[0], c[1]), r0 + buffer, fill=False, linestyle="--", linewidth=1))

    pad = 0.8
    ax.set_xlim(min(xs.min(), start[0], goal[0]) - pad, max(xs.max(), start[0], goal[0]) + pad)
    ax.set_ylim(min(ys.min(), start[1], goal[1]) - pad, max(ys.max(), start[1], goal[1]) + pad)

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

        path_line.set_data(xs[:i+1], ys[:i+1])
        robot_patch.set_xy(triangle_points(xs[i], ys[i], ths[i]))

        info.set_text(
            f"t={i*0.02:.2f}s\nv={vs[i]:.2f} m/s\nω={ws[i]:.2f} rad/s\n"
            f"dist_to_goal={dgoal[i]:.2f} m"
        )
        return path_line, robot_patch, info

    frames = (len(xs) + step_skip - 1) // step_skip
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=20, blit=True)

    plt.show()
