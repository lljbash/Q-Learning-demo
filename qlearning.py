#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pickle
import tqdm
import os
import cv2


os.system('mkdir -p episode')
os.system('mkdir -p output')

nrow = 3
ncol = 3
ncr = nrow * ncol
turn = 100

alpha = 0.2
gamma = 0.9

train_iter = 20000
save_iter = 100


def rc2coord(r, c):
    return r * ncol + c


def coord2rc(coord):
    return coord // ncol, coord % ncol


dcoord = (
        -ncol, # 0: up
        ncol,  # 1: down
        -1,    # 2: left
        1,     # 3: right
        )

valid_op = []

for r in range(ncol):
    for c in range(nrow):
        op = []
        if r != 0:
            op.append(0)
        if r != nrow - 1:
            op.append(1)
        if c != 0:
            op.append(2)
        if c != ncol - 1:
            op.append(3)
        valid_op.append(np.array(op))


# state: player r/c, treasure r/c

def extract_coord_from_state(state):
    coord = state % ncr;
    state //= ncr;
    return coord, state


class Info:
    pass


def parse_state(state):
    info = Info
    info.trap3, state = extract_coord_from_state(state)
    info.trap2, state = extract_coord_from_state(state)
    info.trap1, state = extract_coord_from_state(state)
    info.treasure, state = extract_coord_from_state(state)
    info.player, state = extract_coord_from_state(state)
    return info


def pack_state(info):
    state = info.player
    state *= ncr
    state += info.treasure
    state *= ncr
    state += info.trap1
    state *= ncr
    state += info.trap2
    state *= ncr
    state += info.trap3
    return state


def calc_reward(info):
    if info.player == info.treasure:
        return 100
    elif info.player == info.trap1 or info.player == info.trap2 or info.player == info.trap3:
        return -100
    else:
        return 0


def new_board(info):
    indices = list(range(ncr - 1))
    np.random.shuffle(indices)
    i = 0
    for index in indices:
        if i == info.player:
            i += 1
        if index == 0:
            info.treasure = i
        elif index == 1:
            info.trap1 = i
        elif index == 2:
            info.trap2 = i
        elif index == 3:
            info.trap3 = i
        i += 1


def best_op(Q, state, ops):
    return ops[np.argmax(Q[state, ops])]


def eps_greedy(Q, state, ops, eps):
    e = np.random.rand()
    if e < eps:
        return best_op(Q, state, ops)
    else:
        return np.random.choice(ops)


def episode(Q, eps, save=None):
    Q = Q.copy()
    info = Info
    info.player = 0
    new_board(info)
    state = pack_state(info)
    states = [state]
    pts = [0]
    for i in range(turn):
        op = eps_greedy(Q, state, valid_op[info.player], eps)
        info.player += dcoord[op]
        reward = calc_reward(info)
        if (info.player == info.treasure):
            new_board(info)
        new_state = pack_state(info)
        best_next_op = best_op(Q, new_state, valid_op[info.player])
        Q[state, op] = (1 - alpha) * Q[state, op] + alpha * (reward + gamma * Q[new_state, best_next_op])
        state = new_state
        states.append(state)
        pts.append(pts[-1] + reward)
    if save is not None:
        with open(save, 'wb') as f:
            pickle.dump(Q, f)
    return Q, states, pts


def train():
    Q = np.zeros([ncr**5, 4])
    for i in tqdm.trange(train_iter):
        save = None
        if (i + 1) % save_iter == 0:
            save = 'episode/episode%d.pkl' % (i + 1)
        Q, _, _ = episode(Q, 1 / train_iter * i * 0.6 + 0.2, save)


def view(save):
    grid_size = 100
    radius = grid_size // 2
    height = grid_size * nrow + radius
    width = grid_size * ncol
    text_coord = (10, height - 10)

    with open(save, 'rb') as f:
        Q = pickle.load(f)
    _, states, pts = episode(Q, 1, None)

    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('output.avi', fourcc, 2, (height, width))
    for i, (state, pt) in enumerate(zip(states, pts)):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :] = [255, 255, 255]
        info = parse_state(state)
        r, c = coord2rc(info.trap3)
        cv2.circle(img, (c * grid_size + radius, r * grid_size + radius), radius, (0, 0, 255), -1)
        r, c = coord2rc(info.trap2)
        cv2.circle(img, (c * grid_size + radius, r * grid_size + radius), radius, (0, 0, 255), -1)
        r, c = coord2rc(info.trap1)
        cv2.circle(img, (c * grid_size + radius, r * grid_size + radius), radius, (0, 0, 255), -1)
        r, c = coord2rc(info.treasure)
        cv2.circle(img, (c * grid_size + radius, r * grid_size + radius), radius, (0, 255, 0), -1)
        r, c = coord2rc(info.player)
        cv2.circle(img, (c * grid_size + radius, r * grid_size + radius), radius, (0, 0, 0), -1)
        cv2.putText(img, str(pt), text_coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        # out.write(img)
        cv2.imwrite('output/img%03d.jpg'%i, img)
        cv2.imshow('view', img)
        cv2.waitKey(150)
    # out.release()
    print(pts[-1])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train()
    else:
        view(sys.argv[1])

