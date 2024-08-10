"""
@author: Viet Nguyen <nhviet1009@gmail.com>
Updated for Python 3.12 by Assistant
"""
import argparse
import os
import shutil
import csv
from typing import NamedTuple, Deque
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tensorboardX import SummaryWriter
from collections import deque

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris

class Transition(NamedTuple):
    state: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool

class Args(NamedTuple):
    width: int
    height: int
    block_size: int
    batch_size: int
    lr: float
    gamma: float
    initial_epsilon: float
    final_epsilon: float
    num_decay_epochs: float
    num_epochs: int
    save_interval: int
    replay_memory_size: int
    log_path: str
    saved_path: str
    score_log_path: str  # New field for score log path

def get_args() -> Args:
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epochs between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--score_log_path", type=str, default="score_log.csv")  # New argument for score log path

    args = parser.parse_args()
    return Args(**vars(args))

def train(opt: Args) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    optimizer = Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory: Deque[Transition] = deque(maxlen=opt.replay_memory_size)
    epoch = 0

    # Open CSV file for logging scores
    with open(opt.score_log_path, 'w', newline='') as score_log:
        score_writer = csv.writer(score_log)
        score_writer.writerow(['Epoch', 'Score', 'Tetrominoes', 'Cleared Lines'])  # Write header

        while epoch < opt.num_epochs:
            next_steps = env.get_next_states()
            # Exploration or exploitation
            epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
            u = random()
            random_action = u <= epsilon
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            model.eval()
            with torch.no_grad():
                predictions = model(next_states)[:, 0]
            model.train()
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                index = torch.argmax(predictions).item()

            next_state = next_states[index, :]
            action = next_actions[index]

            reward, done = env.step(action, render=True)

            if torch.cuda.is_available():
                next_state = next_state.cuda()
            replay_memory.append(Transition(state, reward, next_state, done))
            if done:
                final_score = env.score
                final_tetrominoes = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset()
                if torch.cuda.is_available():
                    state = state.cuda()

                # Log the score to CSV
                score_writer.writerow([epoch, final_score, final_tetrominoes, final_cleared_lines])
                score_log.flush()  # Ensure the data is written to the file
            else:
                state = next_state
                continue
            if len(replay_memory) < opt.replay_memory_size / 10:
                continue
            epoch += 1
            batch = Transition(*zip(*sample(replay_memory, min(len(replay_memory), opt.batch_size))))
            state_batch = torch.stack(batch.state)
            reward_batch = torch.from_numpy(np.array(batch.reward, dtype=np.float32)[:, None])
            next_state_batch = torch.stack(batch.next_state)

            if torch.cuda.is_available():
                state_batch = state_batch.cuda()
                reward_batch = reward_batch.cuda()
                next_state_batch = next_state_batch.cuda()

            q_values = model(state_batch)
            model.eval()
            with torch.no_grad():
                next_prediction_batch = model(next_state_batch)
            model.train()

            y_batch = torch.cat(
                tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                      zip(reward_batch, batch.done, next_prediction_batch)))[:, None]

            optimizer.zero_grad()
            loss = criterion(q_values, y_batch)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}/{opt.num_epochs}, Action: {action}, Score: {final_score}, Tetrominoes {final_tetrominoes}, Cleared lines: {final_cleared_lines}")
            writer.add_scalar('Train/Score', final_score, epoch - 1)
            writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
            writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

            if epoch > 0 and epoch % opt.save_interval == 0:
                torch.save(model, f"{opt.saved_path}/tetris_{epoch}")

        torch.save(model, f"{opt.saved_path}/tetris")

if __name__ == "__main__":
    opt = get_args()
    train(opt)