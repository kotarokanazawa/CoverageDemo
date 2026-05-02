#!/usr/bin/env python3
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow

import cavarage as greedy
import cavarage_cpp as cpp
import cavarage_info_gain as info_gain
import cavarage_mdp as mdp
import cavarage_pomdp as pomdp


RANDOM_SEED = None
TARGET_FREE_COVERAGE = 0.995
MAP_SCALE = 0.5
GRAPH_HISTORY_LIMIT = 2000
COMPARE_ANIMATED = False


@dataclass
class MethodState:
    name: str
    module: object
    known_map: np.ndarray
    pose: list
    goal: Optional[tuple] = None
    path: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    traj_y: list = field(default_factory=list)
    traj_x: list = field(default_factory=list)
    time_history: list = field(default_factory=list)
    distance_history: list = field(default_factory=list)
    coverage_history: list = field(default_factory=list)
    travel_distance: float = 0.0
    robot_arrow: Optional[FancyArrow] = None
    color: str = "tab:blue"


@dataclass
class PanelArtists:
    img: object
    path_line: object
    traj_line: object
    candidate_points: object
    candidate_cost_labels: list
    goal_point: object
    robot_point: object
    text: object
    info_text: object


def apply_scaled_map_size(modules):
    scaled_h = max(10, int(greedy.H * MAP_SCALE))
    scaled_w = max(10, int(greedy.W * MAP_SCALE))

    for module in modules:
        module.H = scaled_h
        module.W = scaled_w


def make_coverage_graph(ax, states):
    lines = {}

    for state in states:
        line, = ax.plot([], [], label=state.name, color=state.color, linewidth=1.8)
        lines[state.name] = line

    ax.set_title("Coverage over travel distance")
    ax.set_xlabel("travel distance [cell]")
    ax.set_ylabel("free coverage [%]")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    return lines


def make_panel(ax, state, info_text):
    img = ax.imshow(greedy.build_rgb_map(state.known_map), origin="upper", animated=COMPARE_ANIMATED)
    path_line, = ax.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.4,
        animated=COMPARE_ANIMATED,
        zorder=5,
    )
    traj_line, = ax.plot(
        [],
        [],
        color="tab:green",
        linewidth=1.3,
        alpha=0.8,
        animated=COMPARE_ANIMATED,
        zorder=4,
    )
    candidate_points = ax.scatter(
        [],
        [],
        marker=".",
        s=34,
        c=[],
        cmap="viridis_r",
        alpha=0.85,
        animated=COMPARE_ANIMATED,
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
            animated=COMPARE_ANIMATED,
            zorder=4,
        )
        for _ in range(greedy.MAX_CANDIDATE_COST_LABELS)
    ]
    goal_point = ax.scatter(
        [],
        [],
        marker="*",
        s=220,
        c="gold",
        edgecolors="black",
        linewidths=0.8,
        animated=COMPARE_ANIMATED,
        zorder=7,
    )
    robot_point = ax.scatter(
        [],
        [],
        marker="o",
        s=95,
        c="red",
        edgecolors="white",
        linewidths=0.9,
        animated=COMPARE_ANIMATED,
        zorder=8,
    )

    state.robot_arrow = FancyArrow(
        state.pose[1],
        state.pose[0],
        3.0,
        0.0,
        width=0.2,
        head_width=1.5,
        length_includes_head=True,
        animated=COMPARE_ANIMATED,
        color="red",
        zorder=8,
    )
    ax.add_patch(state.robot_arrow)

    ax.set_title(state.name)
    ax.set_xlim(0, greedy.W)
    ax.set_ylim(greedy.H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    text = ax.text(
        0.0,
        1.02,
        "",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        clip_on=False,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        animated=COMPARE_ANIMATED,
    )
    info = ax.text(
        0.01,
        -0.015,
        info_text,
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
        clip_on=False,
        animated=COMPARE_ANIMATED,
    )

    return PanelArtists(
        img=img,
        path_line=path_line,
        traj_line=traj_line,
        candidate_points=candidate_points,
        candidate_cost_labels=candidate_cost_labels,
        goal_point=goal_point,
        robot_point=robot_point,
        text=text,
        info_text=info,
    )


def update_method(state, true_map, frame):
    module = state.module
    module.simulate_lidar(true_map, state.known_map, state.pose)

    robot_cell = (int(round(state.pose[0])), int(round(state.pose[1])))
    need_replan = False

    if state.goal is None:
        need_replan = True
    elif len(state.path) <= 1:
        need_replan = True
    elif math.hypot(robot_cell[0] - state.goal[0], robot_cell[1] - state.goal[1]) < module.GOAL_REPLAN_DIST:
        need_replan = True

    if need_replan:
        state.goal, state.path, state.candidates = module.choose_frontier_goal_with_candidates(
            state.known_map,
            robot_cell,
        )

    for _ in range(module.STEP_PER_FRAME):
        if len(state.path) <= 1:
            break

        next_cell = state.path[1]
        dy = next_cell[0] - state.pose[0]
        dx = next_cell[1] - state.pose[1]

        state.pose[2] = math.atan2(dy, dx)
        state.pose[0] = float(next_cell[0])
        state.pose[1] = float(next_cell[1])
        state.travel_distance += math.hypot(dy, dx)
        state.path.pop(0)
        state.traj_y.append(state.pose[0])
        state.traj_x.append(state.pose[1])
        module.simulate_lidar(true_map, state.known_map, state.pose)

    if not need_replan and frame % module.CANDIDATE_UPDATE_INTERVAL == 0:
        robot_cell = (int(round(state.pose[0])), int(round(state.pose[1])))
        state.candidates = module.evaluated_frontier_candidates(state.known_map, robot_cell)


def draw_method(state, artists, true_map):
    artists.img.set_data(greedy.build_rgb_map(state.known_map))

    if state.path:
        py = [p[0] for p in state.path]
        px = [p[1] for p in state.path]
        artists.path_line.set_data(px, py)
    else:
        artists.path_line.set_data([], [])

    if state.traj_x:
        artists.traj_line.set_data(state.traj_x, state.traj_y)
    else:
        artists.traj_line.set_data([], [])

    if state.candidates:
        candidate_offsets = np.array([[x, y] for y, x, *_ in state.candidates])
        candidate_values = np.array([value for _, _, value, *_ in state.candidates])
        artists.candidate_points.set_offsets(candidate_offsets)
        artists.candidate_points.set_array(candidate_values)
        artists.candidate_points.set_clim(float(candidate_values.min()), float(candidate_values.max()))
    else:
        artists.candidate_points.set_offsets(np.empty((0, 2)))
        artists.candidate_points.set_array(np.array([]))

    label_candidates = state.candidates[:greedy.MAX_CANDIDATE_COST_LABELS]
    for label, candidate in zip(artists.candidate_cost_labels, label_candidates):
        y, x, value, *_ = candidate
        label.set_position((x, y))
        label.set_text(f"{int(value)}")

    for label in artists.candidate_cost_labels[len(label_candidates):]:
        label.set_text("")

    if state.goal is not None:
        artists.goal_point.set_offsets(np.array([[state.goal[1], state.goal[0]]]))
    else:
        artists.goal_point.set_offsets(np.empty((0, 2)))

    artists.robot_point.set_offsets(np.array([[state.pose[1], state.pose[0]]]))

    state.robot_arrow.remove()
    state.robot_arrow = FancyArrow(
        state.pose[1],
        state.pose[0],
        3.0 * math.cos(state.pose[2]),
        3.0 * math.sin(state.pose[2]),
        width=0.2,
        head_width=1.5,
        length_includes_head=True,
        animated=COMPARE_ANIMATED,
        color="red",
        zorder=8,
    )
    artists.img.axes.add_patch(state.robot_arrow)

    cov = greedy.coverage_stats(state.known_map, true_map)
    artists.text.set_text(
        f"free coverage: {cov['free'] * 100:.1f}%\n"
        f"map known: {cov['map'] * 100:.1f}%\n"
        f"candidates: {len(state.candidates)}"
    )

    return [
        artists.img,
        artists.path_line,
        artists.traj_line,
        artists.candidate_points,
        *artists.candidate_cost_labels,
        artists.goal_point,
        artists.robot_point,
        state.robot_arrow,
        artists.text,
        artists.info_text,
    ]


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    method_specs = [
        (
            "Greedy frontier score",
            greedy,
            "num: path len | star: max score",
            "tab:blue",
        ),
        (
            "MDP value iteration",
            mdp,
            "num: MDP value | star: policy goal",
            "tab:orange",
        ),
        (
            "Coverage path planning",
            cpp,
            "num: sweep rank | star: sweep goal",
            "tab:green",
        ),
        (
            "POMDP next-best-view",
            pomdp,
            "num: belief utility | star: max utility",
            "tab:red",
        ),
        (
            "Information gain / NBV",
            info_gain,
            "num: visible unknown utility | star: max utility",
            "tab:purple",
        ),
    ]

    apply_scaled_map_size([spec[1] for spec in method_specs])

    true_map = greedy.make_random_maze()
    start_pose = greedy.find_start(true_map)

    states = [
        MethodState(
            name=name,
            module=module,
            known_map=np.full((greedy.H, greedy.W), greedy.UNKNOWN, dtype=np.int8),
            pose=start_pose.copy(),
            color=color,
        )
        for name, module, _, color in method_specs
    ]

    for state in states:
        state.module.simulate_lidar(true_map, state.known_map, state.pose)

    fig = plt.figure(figsize=(20, 17))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.15, 1.15, 0.50])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
    ]
    unused_ax = fig.add_subplot(gs[2, 1])
    unused_ax.axis("off")
    graph_ax = fig.add_subplot(gs[3, :])
    fig.subplots_adjust(left=0.03, right=0.99, top=0.96, bottom=0.06, hspace=0.43, wspace=0.04)

    panels = [
        make_panel(ax, state, info_text)
        for ax, state, (_, _, info_text, _) in zip(axes, states, method_specs)
    ]
    coverage_lines = make_coverage_graph(graph_ax, states)
    start_time = time.perf_counter()

    def update(frame):
        artists = []
        coverages = []
        elapsed = time.perf_counter() - start_time

        for state, panel in zip(states, panels):
            update_method(state, true_map, frame)
            artists.extend(draw_method(state, panel, true_map))
            cov = greedy.coverage_stats(state.known_map, true_map)["free"]
            coverages.append(cov)
            state.time_history.append(elapsed)
            state.distance_history.append(state.travel_distance)
            state.coverage_history.append(cov * 100.0)

            if len(state.time_history) > GRAPH_HISTORY_LIMIT:
                state.time_history = state.time_history[-GRAPH_HISTORY_LIMIT:]
                state.distance_history = state.distance_history[-GRAPH_HISTORY_LIMIT:]
                state.coverage_history = state.coverage_history[-GRAPH_HISTORY_LIMIT:]

            line = coverage_lines[state.name]
            line.set_data(state.distance_history, state.coverage_history)
            artists.append(line)

        max_distance = max((state.travel_distance for state in states), default=0.0)
        graph_ax.set_xlim(0.0, max(10.0, max_distance + 5.0))
        graph_ax.set_title(
            f"Coverage over travel distance  elapsed: {elapsed:.1f} s"
        )

        if all(cov > TARGET_FREE_COVERAGE for cov in coverages):
            print("All methods completed coverage.")
            ani.event_source.stop()

        return artists

    ani = FuncAnimation(
        fig,
        update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
