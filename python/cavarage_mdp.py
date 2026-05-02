#!/usr/bin/env python3
import heapq
import math
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow


FREE = 0
WALL = 1

UNKNOWN = -1
KNOWN_FREE = 0
KNOWN_WALL = 1

H, W = 80, 120
RES = 0.1

LIDAR_RANGE = 10.0
LIDAR_BEAMS = 91

STEP_PER_FRAME = 1
GOAL_REPLAN_DIST = 2

UNKNOWN_GAIN = 20.0
DIST_GAIN = 0.02
MDP_GAMMA = 0.96
MDP_STEP_COST = 1.0
MDP_INVALID_ACTION_COST = 5.0
MDP_VALUE_ITERATIONS = 80
MDP_VALUE_TOL = 1e-3

MAX_FRONTIER_CANDIDATES = 250
MAX_EVALUATED_FRONTIER_CANDIDATES = 80
RANDOM_SEED = None  # 固定したい場合は 2 などにする

SHOW_FRONTIER_CANDIDATES = True
MAX_CANDIDATE_COST_LABELS = 40
CANDIDATE_UPDATE_INTERVAL = 3
PATH_WIDEN_PROB = 0.14
EXTRA_WALL_SEGMENTS = 28
WALL_SEGMENT_MIN_LEN = 14
WALL_SEGMENT_MAX_LEN = 42
RUBBLE_DENSITY = 0.025
START_CLEARANCE = 6
MIN_REACHABLE_FREE_RATIO = 0.38
REMOVE_ISOLATED_WALL_CELLS = True


def make_random_maze():
    grid = np.ones((H, W), dtype=np.int8)

    maze_h = H if H % 2 == 1 else H - 1
    maze_w = W if W % 2 == 1 else W - 1

    visited = np.zeros((maze_h, maze_w), dtype=bool)

    stack = [(1, 1)]
    visited[1, 1] = True
    grid[1, 1] = FREE

    while stack:
        y, x = stack[-1]

        candidates = []
        for dy, dx in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            ny = y + dy
            nx = x + dx

            if 1 <= ny < maze_h - 1 and 1 <= nx < maze_w - 1:
                if not visited[ny, nx]:
                    candidates.append((ny, nx, dy, dx))

        if candidates:
            ny, nx, dy, dx = random.choice(candidates)

            grid[y + dy // 2, x + dx // 2] = FREE
            grid[ny, nx] = FREE
            visited[ny, nx] = True

            stack.append((ny, nx))
        else:
            stack.pop()

    # 通路を少しだけ広げる。低めにして細い路地と壁量を残す。
    widened = grid.copy()
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if grid[y, x] == FREE:
                for ny in range(max(1, y - 1), min(H - 1, y + 2)):
                    for nx in range(max(1, x - 1), min(W - 1, x + 2)):
                        if random.random() < PATH_WIDEN_PROB:
                            widened[ny, nx] = FREE

    grid = widened

    # 長い壁を多めに追加し、狭い抜け道を少しだけ残す。
    for _ in range(EXTRA_WALL_SEGMENTS):
        candidate_grid = grid.copy()
        y = random.randint(8, H - 9)
        x = random.randint(8, W - 9)
        length = random.randint(WALL_SEGMENT_MIN_LEN, WALL_SEGMENT_MAX_LEN)

        if random.random() < 0.5:
            x2 = min(W - 2, x + length)
            candidate_grid[y, x:x2] = WALL

            if x2 - x > 6:
                gap = random.randint(x + 2, x2 - 2)
                candidate_grid[y, max(1, gap - 1):min(W - 1, gap + 2)] = FREE
        else:
            y2 = min(H - 2, y + length)
            candidate_grid[y:y2, x] = WALL

            if y2 - y > 6:
                gap = random.randint(y + 2, y2 - 2)
                candidate_grid[max(1, gap - 1):min(H - 1, gap + 2), x] = FREE

        if reachable_free_count(candidate_grid, (1, 1)) >= H * W * MIN_REACHABLE_FREE_RATIO:
            grid = candidate_grid

    # 小さい瓦礫を増やして袋小路や細い路地を作る。
    for _ in range(int(H * W * RUBBLE_DENSITY)):
        y = random.randint(2, H - 4)
        x = random.randint(2, W - 4)

        if grid[y, x] == FREE:
            candidate_grid = grid.copy()
            candidate_grid[y:y + 2, x:x + 2] = WALL

            if reachable_free_count(candidate_grid, (1, 1)) >= H * W * MIN_REACHABLE_FREE_RATIO:
                grid = candidate_grid

    # スタート周辺を空ける
    grid[1:START_CLEARANCE, 1:START_CLEARANCE] = FREE

    # 外周壁
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    if REMOVE_ISOLATED_WALL_CELLS:
        remove_isolated_wall_cells(grid)

    # 追加壁で分断された孤立通路は壁として扱い、到達可能な迷路密度を上げる。
    keep_reachable_free_area(grid, (1, 1))

    return grid


def remove_isolated_wall_cells(grid):
    isolated = []

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if grid[y, x] != WALL:
                continue

            has_wall_neighbor = False

            for ny, nx in neighbors((y, x)):
                if grid[ny, nx] == WALL:
                    has_wall_neighbor = True
                    break

            if not has_wall_neighbor:
                isolated.append((y, x))

    for y, x in isolated:
        grid[y, x] = FREE


def keep_reachable_free_area(grid, start):
    reachable = reachable_free_cells(grid, start)
    grid[(grid == FREE) & (~reachable)] = WALL


def reachable_free_count(grid, start):
    return int(np.sum(reachable_free_cells(grid, start)))


def reachable_free_cells(grid, start):
    sy, sx = start

    reachable = np.zeros_like(grid, dtype=bool)

    if grid[sy, sx] == WALL:
        return reachable

    stack = [start]
    reachable[sy, sx] = True

    while stack:
        y, x = stack.pop()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue

            if reachable[ny, nx] or grid[ny, nx] == WALL:
                continue

            reachable[ny, nx] = True
            stack.append((ny, nx))

    return reachable

def bresenham(y0, x0, y1, x1):
    points = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx + dy
    x, y = x0, y0

    while True:
        points.append((y, x))

        if x == x1 and y == y1:
            break

        e2 = 2 * err

        if e2 >= dy:
            err += dy
            x += sx

        if e2 <= dx:
            err += dx
            y += sy

    return points


def simulate_lidar(true_map, known_map, pose):
    y, x, yaw = pose
    max_cells = int(LIDAR_RANGE / RES)

    angles = np.linspace(-math.pi, math.pi, LIDAR_BEAMS)

    for a in angles:
        th = yaw + a

        ey = int(round(y + max_cells * math.sin(th)))
        ex = int(round(x + max_cells * math.cos(th)))

        for cy, cx in bresenham(int(y), int(x), ey, ex):
            if cy < 0 or cy >= H or cx < 0 or cx >= W:
                break

            if true_map[cy, cx] == WALL:
                known_map[cy, cx] = KNOWN_WALL
                break

            known_map[cy, cx] = KNOWN_FREE


def neighbors(p):
    y, x = p

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = y + dy
        nx = x + dx

        if 0 <= ny < H and 0 <= nx < W:
            yield ny, nx


def astar(known_map, start, goal):
    sy, sx = start
    gy, gx = goal

    if known_map[sy, sx] == KNOWN_WALL:
        return []

    if known_map[gy, gx] == KNOWN_WALL:
        return []

    def h(p):
        return abs(p[0] - gy) + abs(p[1] - gx)

    pq = []
    heapq.heappush(pq, (0, start))

    came = {}
    cost = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)

        if cur == goal:
            path = [cur]

            while cur in came:
                cur = came[cur]
                path.append(cur)

            return path[::-1]

        for nb in neighbors(cur):
            ny, nx = nb

            if known_map[ny, nx] == KNOWN_WALL:
                continue

            if known_map[ny, nx] == UNKNOWN:
                continue

            new_cost = cost[cur] + 1

            if nb not in cost or new_cost < cost[nb]:
                cost[nb] = new_cost
                prio = new_cost + h(nb)
                heapq.heappush(pq, (prio, nb))
                came[nb] = cur

    return []


def bfs_distances(known_map, start):
    sy, sx = start
    dist = {}
    came = {}

    if known_map[sy, sx] != KNOWN_FREE:
        return dist, came

    q = deque([start])
    dist[start] = 0

    while q:
        cur = q.popleft()

        for nb in neighbors(cur):
            ny, nx = nb

            if known_map[ny, nx] != KNOWN_FREE:
                continue

            if nb in dist:
                continue

            dist[nb] = dist[cur] + 1
            came[nb] = cur
            q.append(nb)

    return dist, came


def rebuild_path(came, start, goal):
    if start == goal:
        return [start]

    if goal not in came:
        return []

    path = [goal]
    cur = goal

    while cur != start:
        cur = came[cur]
        path.append(cur)

    return path[::-1]


def build_mdp_solution(known_map, robot_cell):
    fs = ranked_frontier_candidates(known_map, robot_cell)

    if not fs:
        return {}, {}, [], {}

    dist_map, _ = bfs_distances(known_map, robot_cell)
    frontier_reward = {}

    for y, x, unknown_count in fs[:MAX_EVALUATED_FRONTIER_CANDIDATES]:
        state = (y, x)

        if state not in dist_map:
            continue

        frontier_reward[state] = UNKNOWN_GAIN * unknown_count

    if not frontier_reward:
        return {}, {}, [], {}

    states = list(dist_map.keys())
    value = {s: 0.0 for s in states}
    policy = {}

    for _ in range(MDP_VALUE_ITERATIONS):
        delta = 0.0
        next_value = {}

        for state in states:
            if state in frontier_reward:
                next_value[state] = frontier_reward[state]
                policy[state] = state
                delta = max(delta, abs(next_value[state] - value[state]))
                continue

            best_action_value = -1e18
            best_next = state

            for nb in neighbors(state):
                if nb in value:
                    if nb in frontier_reward:
                        action_value = -MDP_STEP_COST + frontier_reward[nb]
                    else:
                        action_value = -MDP_STEP_COST + MDP_GAMMA * value[nb]
                    next_state = nb
                else:
                    action_value = -MDP_INVALID_ACTION_COST + MDP_GAMMA * value[state]
                    next_state = state

                if action_value > best_action_value:
                    best_action_value = action_value
                    best_next = next_state

            next_value[state] = best_action_value
            policy[state] = best_next
            delta = max(delta, abs(best_action_value - value[state]))

        value = next_value

        if delta < MDP_VALUE_TOL:
            break

    candidates = []
    for state, reward in frontier_reward.items():
        dist = dist_map[state]
        discount = MDP_GAMMA ** dist
        step_cost = MDP_STEP_COST * (1.0 - discount) / max(1e-9, 1.0 - MDP_GAMMA)
        score = discount * reward - step_cost
        candidates.append((state[0], state[1], score, reward, dist, int(round(reward / UNKNOWN_GAIN))))

    candidates.sort(key=lambda v: v[2], reverse=True)

    return value, policy, candidates, frontier_reward


def extract_mdp_path(policy, frontier_reward, start):
    if start in frontier_reward:
        return [start], start

    path = [start]
    visited = {start}
    cur = start

    for _ in range(H * W):
        nxt = policy.get(cur)

        if nxt is None or nxt == cur or nxt in visited:
            break

        path.append(nxt)
        cur = nxt

        if cur in frontier_reward:
            return path, cur

        visited.add(cur)

    return path, cur if cur in frontier_reward else None


def frontier_cells(known_map):
    fs = []

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if known_map[y, x] != KNOWN_FREE:
                continue

            unknown_count = 0

            for ny, nx in neighbors((y, x)):
                if known_map[ny, nx] == UNKNOWN:
                    unknown_count += 1

            if unknown_count > 0:
                fs.append((y, x, unknown_count))

    return fs


def ranked_frontier_candidates(known_map, robot_cell):
    fs = frontier_cells(known_map)

    if not fs:
        return []

    ry, rx = robot_cell

    fs.sort(
        key=lambda v: (
            -v[2],
            abs(v[0] - ry) + abs(v[1] - rx),
        )
    )

    return fs[:MAX_FRONTIER_CANDIDATES]


def choose_frontier_goal(known_map, robot_cell):
    goal, path, _ = choose_frontier_goal_with_candidates(known_map, robot_cell)

    return goal, path


def choose_frontier_goal_with_candidates(known_map, robot_cell):
    _, policy, candidates, frontier_reward = build_mdp_solution(known_map, robot_cell)

    if not candidates:
        return None, [], []

    path, goal = extract_mdp_path(policy, frontier_reward, robot_cell)

    if goal is None:
        goal = (candidates[0][0], candidates[0][1])
        path = astar(known_map, robot_cell, goal)

    return goal, path, candidates


def evaluated_frontier_candidates(known_map, robot_cell):
    _, _, candidates = choose_frontier_goal_with_candidates(known_map, robot_cell)

    return candidates


def coverage_stats(known_map, true_map):
    free_total = np.sum(true_map == FREE)
    known_free = np.sum((known_map == KNOWN_FREE) & (true_map == FREE))
    known_cells = np.sum(known_map != UNKNOWN)

    return {
        "free": known_free / max(1, free_total),
        "map": known_cells / known_map.size,
        "known_free": int(known_free),
        "free_total": int(free_total),
        "known_cells": int(known_cells),
        "map_total": int(known_map.size),
    }


def known_coverage_ratio(known_map, true_map):
    return coverage_stats(known_map, true_map)["free"]


def find_start(true_map):
    for y in range(2, H - 2):
        for x in range(2, W - 2):
            if true_map[y, x] == FREE:
                return [float(y), float(x), 0.0]

    return [5.0, 5.0, 0.0]


def build_rgb_map(known_map):
    vis = np.empty((H, W, 3), dtype=np.float32)

    vis[:, :, :] = 0.35
    vis[known_map == KNOWN_FREE] = [1.0, 1.0, 1.0]
    vis[known_map == KNOWN_WALL] = [0.0, 0.0, 0.0]

    return vis


def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

    true_map = make_random_maze()
    known_map = np.full((H, W), UNKNOWN, dtype=np.int8)

    pose = find_start(true_map)

    goal = None
    path = []
    candidates = []
    traj_y = []
    traj_x = []

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(right=0.74)

    simulate_lidar(true_map, known_map, pose)

    img = ax.imshow(build_rgb_map(known_map), origin="upper", animated=True)

    path_line, = ax.plot([], [], linewidth=2, animated=True)
    traj_line, = ax.plot([], [], linewidth=1, animated=True)
    candidate_points = ax.scatter(
        [],
        [],
        marker=".",
        s=34,
        c=[],
        cmap="viridis_r",
        alpha=0.85,
        animated=True,
        zorder=3,
    )
    candidate_cost_labels = [
        ax.text(
            0,
            0,
            "",
            fontsize=6,
            color="tab:blue",
            ha="center",
            va="center",
            animated=True,
            zorder=4,
        )
        for _ in range(MAX_CANDIDATE_COST_LABELS)
    ]
    goal_point = ax.scatter([], [], marker="*", s=160, animated=True)
    robot_point = ax.scatter([], [], s=80, animated=True)

    robot_arrow = FancyArrow(
        pose[1],
        pose[0],
        3.0,
        0.0,
        width=0.2,
        head_width=1.5,
        length_includes_head=True,
        animated=True,
    )
    ax.add_patch(robot_arrow)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    text = ax.text(
        2,
        3,
        "",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        animated=True,
    )
    cost_info_text = ax.text(
        1.03,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        clip_on=False,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"),
        animated=True,
    )

    def update(frame):
        nonlocal pose, goal, path, candidates, robot_arrow

        simulate_lidar(true_map, known_map, pose)

        robot_cell = (int(round(pose[0])), int(round(pose[1])))

        need_replan = False

        if goal is None:
            need_replan = True
        elif len(path) <= 1:
            need_replan = True
        elif math.hypot(robot_cell[0] - goal[0], robot_cell[1] - goal[1]) < GOAL_REPLAN_DIST:
            need_replan = True

        if need_replan:
            goal, path, candidates = choose_frontier_goal_with_candidates(known_map, robot_cell)

        for _ in range(STEP_PER_FRAME):
            if len(path) <= 1:
                break

            next_cell = path[1]

            dy = next_cell[0] - pose[0]
            dx = next_cell[1] - pose[1]

            pose[2] = math.atan2(dy, dx)
            pose[0] = float(next_cell[0])
            pose[1] = float(next_cell[1])

            path.pop(0)

            traj_y.append(pose[0])
            traj_x.append(pose[1])

            simulate_lidar(true_map, known_map, pose)

        if not need_replan and frame % CANDIDATE_UPDATE_INTERVAL == 0:
            robot_cell = (int(round(pose[0])), int(round(pose[1])))
            candidates = evaluated_frontier_candidates(known_map, robot_cell)

        img.set_data(build_rgb_map(known_map))

        if path:
            py = [p[0] for p in path]
            px = [p[1] for p in path]
            path_line.set_data(px, py)
        else:
            path_line.set_data([], [])

        if traj_x:
            traj_line.set_data(traj_x, traj_y)
        else:
            traj_line.set_data([], [])

        if SHOW_FRONTIER_CANDIDATES and candidates:
            candidate_offsets = np.array([[x, y] for y, x, *_ in candidates])
            candidate_costs = np.array([cost for _, _, cost, *_ in candidates])
            candidate_points.set_offsets(candidate_offsets)
            candidate_points.set_array(candidate_costs)
            candidate_points.set_clim(float(candidate_costs.min()), float(candidate_costs.max()))
        else:
            candidate_points.set_offsets(np.empty((0, 2)))
            candidate_points.set_array(np.array([]))

        label_candidates = candidates[:MAX_CANDIDATE_COST_LABELS] if SHOW_FRONTIER_CANDIDATES else []
        for label, candidate in zip(candidate_cost_labels, label_candidates):
            y, x, cost, *_ = candidate
            label.set_position((x, y))
            label.set_text(f"{int(cost)}")

        for label in candidate_cost_labels[len(label_candidates):]:
            label.set_text("")

        if goal is not None:
            goal_point.set_offsets(np.array([[goal[1], goal[0]]]))
        else:
            goal_point.set_offsets(np.empty((0, 2)))

        robot_point.set_offsets(np.array([[pose[1], pose[0]]]))

        robot_arrow.remove()
        robot_arrow = FancyArrow(
            pose[1],
            pose[0],
            3.0 * math.cos(pose[2]),
            3.0 * math.sin(pose[2]),
            width=0.2,
            head_width=1.5,
            length_includes_head=True,
            animated=True,
        )
        ax.add_patch(robot_arrow)

        cov = coverage_stats(known_map, true_map)
        text.set_text(
            f"free coverage: {cov['free'] * 100:.1f}%  "
            f"map known: {cov['map'] * 100:.1f}%  "
            f"candidates: {len(candidates)}"
        )
        cost_info_text.set_text(
            "candidate number: integer part of MDP value\n"
            "state: known free cell, action: 4-neighbor move\n"
            "reward: unknown_gain * unknown_neighbors at frontier candidates\n"
            "policy: value iteration chooses the action with max expected reward\n"
            "star: candidate reached by following the MDP policy"
        )

        if cov["free"] > 0.995:
            print("Coverage completed.")
            ani.event_source.stop()

        return (
            img,
            path_line,
            traj_line,
            candidate_points,
            *candidate_cost_labels,
            goal_point,
            robot_point,
            robot_arrow,
            text,
            cost_info_text,
        )

    ani = FuncAnimation(
        fig,
        update,
        interval=50,
        blit=True,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
