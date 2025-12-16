import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# AMR Navigation: Static + Dynamic Obstacles (Crossing obstacle, no chase)
# - Dynamic obstacle crosses Start->Goal line
# - Robot yields (slows/stops) if collision risk predicted
# - Potential fields + hard safety (never enters inflated obstacles)
# - Exact replay animation (logs obstacle states)
# - Plots: dist-to-goal, v(t), ω(t)
# ============================================================

# -----------------------------
# Helpers
# -----------------------------
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def unit(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def triangle_points(x, y, th, L=0.14, W=0.08):
    tip = np.array([L, 0.0])
    left = np.array([-0.6 * L,  W / 2])
    right = np.array([-0.6 * L, -W / 2])
    P = np.vstack([tip, left, right])
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    return (P @ R.T) + np.array([x, y])

def inside_inflated(p, centers, radii, buffer):
    for c, r in zip(centers, radii):
        if np.linalg.norm(p - c) <= (r + buffer):
            return True
    return False

def closest_inflated_distance(p, centers, radii, buffer):
    best = np.inf
    for c, r in zip(centers, radii):
        d = np.linalg.norm(p - c) - (r + buffer)
        best = min(best, d)
    return best

def point_to_segment_distance(p, a, b):
    """
    Distance from point p to line segment a->b.
    Used for "is obstacle near the start-goal straight path?"
    """
    ap = p - a
    ab = b - a
    denom = np.dot(ab, ab) + 1e-12
    t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj), t, proj

# -----------------------------
# Obstacles
# -----------------------------
class StaticObstacle:
    def __init__(self, center, radius):
        self.c = np.array(center, dtype=float)
        self.r = float(radius)

class DynamicObstacle:
    """
    Moving circle obstacle with bouncing in bounds (NO chase).
    """
    def __init__(self, center, radius, velocity, bounds):
        self.c = np.array(center, dtype=float)
        self.r = float(radius)
        self.v = np.array(velocity, dtype=float)
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)

    def step(self, dt):
        self.c += self.v * dt
        xmin, xmax, ymin, ymax = self.bounds

        if self.c[0] - self.r < xmin:
            self.c[0] = xmin + self.r
            self.v[0] *= -1
        if self.c[0] + self.r > xmax:
            self.c[0] = xmax - self.r
            self.v[0] *= -1
        if self.c[1] - self.r < ymin:
            self.c[1] = ymin + self.r
            self.v[1] *= -1
        if self.c[1] + self.r > ymax:
            self.c[1] = ymax - self.r
            self.v[1] *= -1

# -----------------------------
# Potential field (static + dynamic predictive)
# -----------------------------
def force_field(p, goal,
                static_centers, static_radii,
                dyn_centers, dyn_radii, dyn_vels,
                buffer,
                k_att=1.6,
                k_rep_s=4.0,
                k_rep_d=7.0,
                rep_cap=18.0,
                k_tan=0.35,
                tan_cap=7.0,
                lookahead=0.7):
    to_goal = goal - p
    F_att = k_att * to_goal

    F_rep = np.zeros(2)
    F_tan = np.zeros(2)
    ghat = unit(to_goal)

    # Static
    for c, r in zip(static_centers, static_radii):
        r_inf = r + buffer
        vec = p - c
        dist = np.linalg.norm(vec)
        away = unit(vec)
        d = dist - r_inf

        if d <= 0:
            F_rep += 160.0 * away
            perp = np.array([-away[1], away[0]])
            sign = np.sign(np.cross(np.append(away, 0), np.append(ghat, 0))[2]) or 1.0
            F_tan += 35.0 * sign * perp
            continue

        mag = min(k_rep_s / (d**2), rep_cap)
        F_rep += mag * away

        perp = np.array([-away[1], away[0]])
        sign = np.sign(np.cross(np.append(away, 0), np.append(ghat, 0))[2]) or 1.0
        F_tan += min(k_tan * mag, tan_cap) * sign * perp

    # Dynamic (predictive)
    for c, r, v in zip(dyn_centers, dyn_radii, dyn_vels):
        c_pred = c + v * lookahead
        r_inf = r + buffer
        vec = p - c_pred
        dist = np.linalg.norm(vec)
        away = unit(vec)
        d = dist - r_inf

        if d <= 0:
            F_rep += 200.0 * away
            perp = np.array([-away[1], away[0]])
            sign = np.sign(np.cross(np.append(away, 0), np.append(ghat, 0))[2]) or 1.0
            F_tan += 45.0 * sign * perp
            continue

        mag = min(k_rep_d / (d**2), rep_cap * 1.25)
        F_rep += mag * away

        perp = np.array([-away[1], away[0]])
        sign = np.sign(np.cross(np.append(away, 0), np.append(ghat, 0))[2]) or 1.0
        F_tan += min(0.55 * k_tan * mag, tan_cap + 1.0) * sign * perp

    return F_att + F_rep + F_tan

# -----------------------------
# Collision-risk / yield logic
# -----------------------------
def collision_risk_yield(robot_p, robot_theta, v_cmd,
                         dyn_center, dyn_vel, dyn_r, buffer,
                         horizon=1.2, dt=0.1,
                         yield_dist=0.55):
    """
    Predict robot and obstacle positions forward for a short horizon.
    If predicted distance < (r+buffer+yield_dist) => yield.
    """
    p = robot_p.copy()
    c = dyn_center.copy()

    for _ in range(int(horizon / dt)):
        # predict robot forward along its heading (simple model)
        p = p + np.array([np.cos(robot_theta), np.sin(robot_theta)]) * (v_cmd * dt)
        # predict obstacle
        c = c + dyn_vel * dt

        if np.linalg.norm(p - c) < (dyn_r + buffer + yield_dist):
            return True
    return False

# -----------------------------
# Simulation (logs dynamic obstacles exactly)
# -----------------------------
def simulate(start, goal, static_obs, dynamic_obs,
             bounds=(0, 9, 0, 9),
             T=55.0, dt=0.03,
             buffer=0.70,
             v_max=1.25, w_max=3.0,
             goal_tol=0.15,
             slow_radius=2.0,
             final_approach_radius=1.3,
             safe_clearance=0.45):

    start = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)

    # static arrays
    sC = np.array([o.c for o in static_obs], dtype=float) if static_obs else np.zeros((0, 2))
    sR = np.array([o.r for o in static_obs], dtype=float) if static_obs else np.zeros((0,))

    # robot state
    x, y = start
    theta = np.pi/4  # face roughly toward goal

    N = int(T / dt)
    t = np.arange(N) * dt

    xs = np.zeros(N); ys = np.zeros(N); ths = np.zeros(N)
    vs = np.zeros(N); ws = np.zeros(N); dgoal = np.zeros(N)

    # log dynamic obstacle states
    M = len(dynamic_obs)
    dynC_log = np.zeros((N, M, 2))
    dynV_log = np.zeros((N, M, 2))
    dynR = np.array([o.r for o in dynamic_obs], dtype=float) if M > 0 else np.zeros((0,))

    # yield flag (for display)
    yielded = np.zeros(N, dtype=bool)

    reached = False
    reach_idx = N - 1

    for k in range(N):
        # step dynamic obstacles and log
        for o in dynamic_obs:
            o.step(dt)

        if M > 0:
            dynC = np.array([o.c for o in dynamic_obs], dtype=float)
            dynV = np.array([o.v for o in dynamic_obs], dtype=float)
            dynC_log[k] = dynC
            dynV_log[k] = dynV
        else:
            dynC = np.zeros((0, 2))
            dynV = np.zeros((0, 2))

        p = np.array([x, y])
        to_goal = goal - p
        dist_goal = np.linalg.norm(to_goal)
        dgoal[k] = dist_goal

        if dist_goal <= goal_tol:
            reached = True
            reach_idx = k
            xs[k], ys[k], ths[k] = x, y, theta
            vs[k], ws[k] = 0.0, 0.0
            break

        # combine obstacles for safety checks
        allC = np.vstack([sC, dynC]) if (len(sC) + len(dynC)) > 0 else np.zeros((0, 2))
        allR = np.hstack([sR, dynR]) if (len(sR) + len(dynR)) > 0 else np.zeros((0,))
        d_obs = closest_inflated_distance(p, allC, allR, buffer) if len(allC) > 0 else np.inf

        # final approach mode
        final_mode = (dist_goal < final_approach_radius) and (d_obs > safe_clearance)

        # compute desired heading
        if final_mode:
            des_theta = np.arctan2(to_goal[1], to_goal[0])
        else:
            F = force_field(p, goal, sC, sR, dynC, dynR, dynV, buffer=buffer)
            des_dir = unit(F)
            des_theta = np.arctan2(des_dir[1], des_dir[0])

        # heading control
        e = wrap_to_pi(des_theta - theta)
        omega = np.clip(3.2 * e, -w_max, w_max)

        # base speed control
        align = np.clip(np.cos(e), -1.0, 1.0)
        v_min = 0.20
        v = v_min + (v_max - v_min) * max(0.0, align)

        # slow near goal
        if dist_goal < slow_radius:
            v *= dist_goal / slow_radius

        # ----------------------------------------------------
        # YIELD BEHAVIOR:
        # If a dynamic obstacle is predicted to cross near the robot's forward path,
        # slow down / stop briefly, then continue (creates visible "robot waits" effect).
        # ----------------------------------------------------
        will_yield = False
        if M > 0:
            # Check the first dynamic obstacle (the "crossing" one) for clear demo
            j = 0
            will_yield = collision_risk_yield(
                robot_p=p,
                robot_theta=theta,
                v_cmd=v,
                dyn_center=dynC[j],
                dyn_vel=dynV[j],
                dyn_r=dynR[j],
                buffer=buffer,
                horizon=1.4,
                dt=0.12,
                yield_dist=0.50
            )

        if will_yield and dist_goal > 1.0:
            yielded[k] = True
            # stop or slow strongly
            v = 0.02
            # small steering to keep stable (or set omega=0)
            omega = np.clip(omega, -1.6, 1.6)

        # HARD SAFETY (never step inside inflated obstacles)
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        p_next = np.array([x_next, y_next])

        if len(allC) > 0 and inside_inflated(p_next, allC, allR, buffer):
            # rotate-only / creep
            v = 0.02
            omega = np.clip(omega + np.sign(e) * 1.2, -w_max, w_max)

            x_next = x + v * np.cos(theta) * dt
            y_next = y + v * np.sin(theta) * dt
            p_next = np.array([x_next, y_next])

            if inside_inflated(p_next, allC, allR, buffer):
                x_next, y_next = x, y  # rotate only

        # integrate
        x, y = x_next, y_next
        theta = theta + omega * dt

        # keep in bounds
        xmin, xmax, ymin, ymax = bounds
        x = np.clip(x, xmin + 0.05, xmax - 0.05)
        y = np.clip(y, ymin + 0.05, ymax - 0.05)

        xs[k], ys[k], ths[k] = x, y, theta
        vs[k], ws[k] = v, omega

    K = reach_idx + 1 if reached else N
    return (t[:K], xs[:K], ys[:K], ths[:K], vs[:K], ws[:K], dgoal[:K],
            dynC_log[:K], dynV_log[:K], dynR, yielded[:K], reached, buffer)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    bounds = (0.0, 9.0, 0.0, 9.0)
    start = (0.8, 0.8)
    goal  = (8.3, 8.3)

    # Static obstacles
    static_obs = [
        StaticObstacle((3.0, 2.8), 0.75),
        StaticObstacle((5.2, 5.2), 0.90),
        StaticObstacle((7.2, 3.4), 0.70),
    ]

    # Dynamic obstacle #0 crosses the start-goal straight path:
    # Start->Goal is roughly diagonal; so we move obstacle roughly "downward" near that diagonal.
    dynamic_obs = [
        DynamicObstacle(center=(4.4, 6.2), radius=0.45, velocity=(0.0, -0.85), bounds=bounds),  # crosses diagonal
        DynamicObstacle(center=(6.9, 7.2), radius=0.50, velocity=(-0.65, -0.55), bounds=bounds),
        DynamicObstacle(center=(4.4, 1.6), radius=0.40, velocity=(0.75, 0.55), bounds=bounds),
    ]

    (t, xs, ys, ths, vs, ws, dgoal,
     dynC_log, dynV_log, dynR, yielded, reached, buffer) = simulate(
        start=start, goal=goal,
        static_obs=static_obs, dynamic_obs=dynamic_obs,
        bounds=bounds,
        T=55.0, dt=0.03,
        buffer=0.70,
        v_max=1.25, w_max=3.0,
        goal_tol=0.15
    )

    print("Reached goal:", reached, "| Final dist:", float(dgoal[-1]))

    # ============================================================
    # Colored Animation
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.8, 7.8))
    ax.set_title("AMR Navigation: Static + Dynamic Obstacles (Yield + Avoid + Reach Goal)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.35)
    ax.axis("equal")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    # Start & Goal
    ax.plot(start[0], start[1], "o", markersize=9, color="tab:blue", label="Start")
    ax.plot(goal[0], goal[1], "*", markersize=18, color="gold", markeredgecolor="black", label="Goal")
    goal_ring = plt.Circle((goal[0], goal[1]), 0.15, fill=False, linewidth=2, edgecolor="gold", alpha=0.9)
    ax.add_patch(goal_ring)

    # Draw ideal straight-line path (dotted)
    ax.plot([start[0], goal[0]], [start[1], goal[1]], linestyle=":", linewidth=2, alpha=0.5, label="Direct path")

    # Static obstacles (filled gray + dashed buffer)
    for o in static_obs:
        solid = plt.Circle((o.c[0], o.c[1]), o.r, fill=True, alpha=0.18,
                           color="tab:gray", edgecolor="black", linewidth=2)
        buff  = plt.Circle((o.c[0], o.c[1]), o.r + buffer, fill=False,
                           linestyle="--", linewidth=1.5, edgecolor="tab:gray", alpha=0.9)
        ax.add_patch(solid)
        ax.add_patch(buff)

    # Dynamic obstacles patches + buffer rings + arrows (red)
    M = dynC_log.shape[1]
    dyn_patches = []
    dyn_buffers = []
    dyn_arrows = []

    for j in range(M):
        c0 = dynC_log[0, j]
        r0 = dynR[j]

        solid = plt.Circle((c0[0], c0[1]), r0, fill=True, alpha=0.22,
                           color="tab:red", edgecolor="darkred", linewidth=2)
        buff  = plt.Circle((c0[0], c0[1]), r0 + buffer, fill=False,
                           linestyle="--", linewidth=1.5, edgecolor="tab:red", alpha=0.9)
        ax.add_patch(solid)
        ax.add_patch(buff)
        dyn_patches.append(solid)
        dyn_buffers.append(buff)

        v0 = dynV_log[0, j]
        arr = ax.arrow(c0[0], c0[1], 0.35*v0[0], 0.35*v0[1],
                       head_width=0.12, head_length=0.15, length_includes_head=True,
                       color="tab:red", alpha=0.9)
        dyn_arrows.append(arr)

    # Robot path + robot
    path_line, = ax.plot([], [], lw=2.8, color="tab:green", label="Robot Path")
    robot_patch = plt.Polygon(triangle_points(xs[0], ys[0], ths[0]), closed=True,
                              facecolor="tab:blue", edgecolor="navy", alpha=0.85)
    ax.add_patch(robot_patch)

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                   bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85))

    yield_text = ax.text(0.02, 0.86, "", transform=ax.transAxes, va="top",
                         bbox=dict(boxstyle="round", fc="lightyellow", ec="olive", alpha=0.85))

    ax.legend(loc="lower right", framealpha=0.9)

    step_skip = 1

    def init():
        path_line.set_data([], [])
        robot_patch.set_xy(triangle_points(xs[0], ys[0], ths[0]))
        info.set_text("")
        yield_text.set_text("")
        return [path_line, robot_patch, info, yield_text] + dyn_patches + dyn_buffers + dyn_arrows

    def update(frame):
        i = frame * step_skip
        if i >= len(xs):
            i = len(xs) - 1

        # Robot update
        path_line.set_data(xs[:i+1], ys[:i+1])
        robot_patch.set_xy(triangle_points(xs[i], ys[i], ths[i]))

        # Dynamic obstacles update (exact replay)
        for j in range(M):
            c = dynC_log[i, j]
            dyn_patches[j].center = (c[0], c[1])
            dyn_buffers[j].center = (c[0], c[1])

            dyn_arrows[j].remove()
            v = dynV_log[i, j]
            dyn_arrows[j] = ax.arrow(c[0], c[1], 0.35*v[0], 0.35*v[1],
                                     head_width=0.12, head_length=0.15,
                                     length_includes_head=True,
                                     color="tab:red", alpha=0.9)

        info.set_text(
            f"t={t[i]:.2f}s\nv={vs[i]:.2f} m/s\nω={ws[i]:.2f} rad/s\n"
            f"dist_to_goal={dgoal[i]:.2f} m"
        )

        # Yield message (robot slows/stops)
        if yielded[i]:
            yield_text.set_text("🛑 Yielding: dynamic obstacle crossing path")
        else:
            yield_text.set_text("")

        return [path_line, robot_patch, info, yield_text] + dyn_patches + dyn_buffers + dyn_arrows

    frames = (len(xs) + step_skip - 1) // step_skip
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=20, blit=False)
    plt.show()

    # ============================================================
    # Plots: distance, v(t), ω(t)
    # ============================================================
    plt.figure()
    plt.title("Distance to Goal vs Time")
    plt.plot(t, dgoal, linewidth=2)
    plt.xlabel("time [s]")
    plt.ylabel("distance [m]")
    plt.grid(True, alpha=0.35)

    plt.figure()
    plt.title("Linear Velocity v(t) vs Time")
    plt.plot(t, vs, linewidth=2)
    plt.xlabel("time [s]")
    plt.ylabel("v [m/s]")
    plt.grid(True, alpha=0.35)

    plt.figure()
    plt.title("Angular Velocity ω(t) vs Time")
    plt.plot(t, ws, linewidth=2)
    plt.xlabel("time [s]")
    plt.ylabel("ω [rad/s]")
    plt.grid(True, alpha=0.35)

    plt.show()
