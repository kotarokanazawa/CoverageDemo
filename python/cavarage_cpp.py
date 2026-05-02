#!/usr/bin/env python3
import cavarage as base

from cavarage import *  # noqa: F401,F403


CPP_SWEEP_BAND_HEIGHT = 4
CPP_DISTANCE_GAIN = 0.15
CPP_FRONTIER_GAIN = 2.0


def sweep_rank(y, x):
    band = y // CPP_SWEEP_BAND_HEIGHT

    if band % 2 == 0:
        return band * W + x

    return band * W + (W - 1 - x)


def choose_frontier_goal_with_candidates(known_map, robot_cell):
    fs = ranked_frontier_candidates(known_map, robot_cell)

    if not fs:
        return None, [], []

    dist_map, came = bfs_distances(known_map, robot_cell)
    candidates = []

    for y, x, unknown_count in fs[:MAX_EVALUATED_FRONTIER_CANDIDATES]:
        goal = (y, x)
        dist = dist_map.get(goal)

        if dist is None:
            continue

        rank = sweep_rank(y, x)
        score = -rank - CPP_DISTANCE_GAIN * dist + CPP_FRONTIER_GAIN * unknown_count
        candidates.append((y, x, rank, score, dist, unknown_count))

    if not candidates:
        return None, [], []

    candidates.sort(key=lambda v: v[2])
    best = max(candidates, key=lambda v: v[3])
    goal = (best[0], best[1])
    path = rebuild_path(came, robot_cell, goal)

    return goal, path, candidates


def evaluated_frontier_candidates(known_map, robot_cell):
    _, _, candidates = choose_frontier_goal_with_candidates(known_map, robot_cell)

    return candidates


def choose_frontier_goal(known_map, robot_cell):
    goal, path, _ = choose_frontier_goal_with_candidates(known_map, robot_cell)

    return goal, path


def main():
    base.choose_frontier_goal_with_candidates = choose_frontier_goal_with_candidates
    base.evaluated_frontier_candidates = evaluated_frontier_candidates
    base.choose_frontier_goal = choose_frontier_goal
    base.main()


if __name__ == "__main__":
    main()
