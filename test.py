import argparse
from typing import NamedTuple
import torch
import cv2
from src.tetris import Tetris

class Args(NamedTuple):
    width: int
    height: int
    block_size: int
    fps: int
    saved_path: str
    output: str

def get_args() -> Args:
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return Args(**vars(args))

def test(opt: Args) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load(f"{opt.saved_path}/tetris")
    else:
        model = torch.load(f"{opt.saved_path}/tetris", map_location=torch.device('cpu'))
    model.eval()

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()

    if torch.cuda.is_available():
        model.cuda()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(opt.output, fourcc, opt.fps, 
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))

    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        if torch.cuda.is_available():
            next_states = next_states.cuda()

        with torch.no_grad():
            predictions = model(next_states)[:, 0]

        index = torch.argmax(predictions).item()
        action = next_actions[index]

        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)