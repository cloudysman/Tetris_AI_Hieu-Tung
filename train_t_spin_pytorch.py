import numpy as np
from game import Game
from common import GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH
from src.deep_q_network import DQNAgent

def preprocess_state(state):
    # Ensure the state is always flattened to 400 elements
    flattened = np.array(state[0]).flatten()
    if flattened.shape[0] != 400:
        # Pad or truncate to 400 elements
        padded = np.zeros(400)
        padded[:min(400, flattened.shape[0])] = flattened[:min(400, flattened.shape[0])]
        flattened = padded
    #print(f"State shape: {flattened.shape}")  # Debug print
    return flattened

def train_dqn():
    env = Game()
    initial_state = env.reset()
    preprocessed_state = preprocess_state(initial_state)
    state_size = preprocessed_state.shape[0]
    print(f"State size: {state_size}")  # Debug print
    
    action_size = len(env.get_all_possible_gamestates()[1])
    print(f"Action size: {action_size}")  # Debug print
    
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward = env.step(chosen=action)
            next_state = preprocess_state(next_state)
            done = env.is_done()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print(f"episode: {e}/{episodes}, score: {total_reward}, epsilon: {agent.epsilon:.2f}")

        if e % 10 == 0:
            agent.update_target_model()

    agent.save("tetris_dqn.h5")

if __name__ == "__main__":
    train_dqn()