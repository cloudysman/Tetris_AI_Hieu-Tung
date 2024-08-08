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
    action: int
    next_state: torch.Tensor
    reward: float
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
    score_log_path: str
    resume: bool
    checkpoint_path: str

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
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--score_log_path", type=str, default="score_log.csv")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint for resuming training")

    args = parser.parse_args()
    return Args(**vars(args))

def resume_training(opt: Args, checkpoint_path: str):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    model = checkpoint['model']
    optimizer = Adam(model.parameters(), lr=opt.lr)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_epoch = checkpoint['epoch'] + 1
    epsilon = max(opt.final_epsilon, opt.initial_epsilon - (start_epoch / opt.num_decay_epochs) * (opt.initial_epsilon - opt.final_epsilon))
    
    return model, optimizer, start_epoch, epsilon

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
    
    if opt.resume and opt.checkpoint_path:
        model, optimizer, start_epoch, epsilon = resume_training(opt, opt.checkpoint_path)
    else:
        model = DeepQNetwork()
        optimizer = Adam(model.parameters(), lr=opt.lr)
        start_epoch = 0
        epsilon = opt.initial_epsilon

    criterion = nn.MSELoss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory: Deque[Transition] = deque(maxlen=opt.replay_memory_size)

    with open(opt.score_log_path, 'a', newline='') as score_log:
        score_writer = csv.writer(score_log)
        if start_epoch == 0:  # Only write header if starting from scratch
            score_writer.writerow(['Epoch', 'Score', 'Tetrominoes', 'Cleared Lines'])

        for epoch in range(start_epoch, opt.num_epochs):
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
            replay_memory.append(Transition(state, action, next_state, reward, done))
            if done:
                final_score = env.score
                final_tetrominoes = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset()
                if torch.cuda.is_available():
                    state = state.cuda()
                
                score_writer.writerow([epoch, final_score, final_tetrominoes, final_cleared_lines])
                score_log.flush()
            else:
                state = next_state
                continue
            if len(replay_memory) < opt.replay_memory_size / 10:
                continue
            
            batch = Transition(*zip(*sample(replay_memory, min(len(replay_memory), opt.batch_size))))
            state_batch = torch.stack(batch.state)
            action_batch = torch.LongTensor(batch.action)
            reward_batch = torch.FloatTensor(batch.reward)
            next_state_batch = torch.stack(batch.next_state)
            done_batch = torch.FloatTensor(batch.done)

            if torch.cuda.is_available():
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                next_state_batch = next_state_batch.cuda()
                done_batch = done_batch.cuda()

            q_values = model(state_batch)
            model.eval()
            with torch.no_grad():
                next_prediction_batch = model(next_state_batch)
            model.train()

            y_batch = reward_batch + torch.mul((opt.gamma * torch.max(next_prediction_batch, 1)[0]), (1 - done_batch))
            
            optimizer.zero_grad()
            loss = criterion(q_values, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}/{opt.num_epochs}, Action: {action}, Score: {final_score}, Tetrominoes {final_tetrominoes}, Cleared lines: {final_cleared_lines}")
            writer.add_scalar('Train/Score', final_score, epoch - 1)
            writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
            writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

            if epoch > 0 and epoch % opt.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f"{opt.saved_path}/tetris_{epoch}")

        torch.save({
            'epoch': opt.num_epochs,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f"{opt.saved_path}/tetris_final")

if __name__ == "__main__":
    opt = get_args()
    train(opt)