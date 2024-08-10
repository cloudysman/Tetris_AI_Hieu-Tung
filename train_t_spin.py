import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from game import Game
from tetromino import Tetromino
import random
import pickle
from common import *
from gui import Gui
import time
import multiprocessing
from queue import Empty as QueueEmpty
from tqdm import tqdm
import threading

# size dependent
shape_main_grid = (1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 1)
if STATE_INPUT == 'short':
    shape_hold_next = (1, 1 * 2 + 1 + 6 * Tetromino.pool_size())
    split_hold_next = 1 * 2 + 1
else:
    shape_hold_next = (1, GAME_BOARD_WIDTH * 2 + 1 + 6 * Tetromino.pool_size())
    split_hold_next = GAME_BOARD_WIDTH * 2 + 1

gamma = 0.95
epsilon = 0.06

current_avg_score = 0
rand = random.Random()

penalty = -500
reward_coef = [1.0, 0.5, 0.4, 0.3]

def make_model_conv2d():
    main_grid_input = keras.Input(shape=shape_main_grid[1:], name="main_grid_input")
    a = layers.Conv2D(64, 6, activation="relu", input_shape=shape_main_grid[1:])(main_grid_input)
    a1 = layers.MaxPool2D(pool_size=(15, 5), strides=(1, 1))(a)
    a1 = layers.Flatten()(a1)
    a2 = layers.AveragePooling2D(pool_size=(15, 5))(a)
    a2 = layers.Flatten()(a2)

    b = layers.Conv2D(256, 4, activation="relu", input_shape=shape_main_grid[1:])(main_grid_input)
    b1 = layers.MaxPool2D(pool_size=(17, 7), strides=(1, 1))(b)
    b1 = layers.Flatten()(b1)
    b2 = layers.AveragePooling2D(pool_size=(17, 7))(b)
    b2 = layers.Flatten()(b2)

    hold_next_input = keras.Input(shape=shape_hold_next[1:], name="hold_next_input")

    x = layers.concatenate([a1, a2, b1, b2, hold_next_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    critic_output = layers.Dense(1)(x)

    model = keras.Model(inputs=[main_grid_input, hold_next_input], outputs=critic_output)
    return model

def load_model(filepath=None):
    model_loaded = make_model_conv2d()

    model_loaded.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='huber',
        metrics=['mean_squared_error']
    )
    if filepath is not None:
        model_loaded.load_weights(filepath)
    else:
        model_loaded.save(FOLDER_NAME + 'whole_model/outer_0.keras')
        print('model initial state has been saved')

    return model_loaded

def get_data_from_playing_cnn2d(model_filename, target_size=8000, max_steps_per_episode=2000, proc_num=4, queue=None, headless=True):
    print(f"Worker {proc_num} starting...")
    start_time = time.time()
    try:
        print(f"Worker {proc_num}: Attempting to load model from {model_filename}")
        try:
            model = keras.models.load_model(model_filename)
            print(f"Worker {proc_num}: Model loaded successfully")
        except Exception as e:
            print(f"Worker {proc_num}: Failed to load model. Error: {str(e)}")
            print(f"Worker {proc_num}: Creating a new model.")
            model = make_model_conv2d()
            model.compile(
                optimizer=keras.optimizers.Adam(0.001),
                loss='huber',
                metrics=['mean_squared_error']
            )
            print(f"Worker {proc_num}: New model created and compiled.")

        print(f"Worker {proc_num}: Creating game environment with GUI")
        gui = Gui(is_display=False)  # Adjust delay as needed
        #env = Game(gui=gui)
        env = Game(headless=True)

        print(f"Worker {proc_num}: Game environment created successfully")

        global epsilon
        if proc_num == 0:
            epsilon = 0
        print(f"Worker {proc_num}: Epsilon set to {epsilon}")

        data = list()
        episode_max = 1000
        total_score = 0
        avg_score = 0
        t_spins = 0

        print(f"Worker {proc_num}: Starting game loop")
        for episode in range(episode_max):
            print(f"Worker {proc_num}: Starting episode {episode + 1}")
            env.reset()
            episode_data = list()
            episode_score = 0
            episode_lines = 0
            episode_t_spins = 0

            for step in range(max_steps_per_episode):
                if step % 100 == 0:
                    print(f"Worker {proc_num}: Episode {episode + 1}, Step {step}")
                
                s = env.get_state_dqn_conv2d(env.current_state)
                #print(f"Type of s: {type(s)}")
                # if isinstance(s, tuple):
                #     print(f"Shape of s[0]: {s[0].shape}, Shape of s[1]: {s[1].shape}")
                # else:
                #     print(f"Shape of s: {s.shape}")
                possible_states, add_scores, dones, is_include_hold, is_new_hold, _ = env.get_all_possible_states_conv2d()
                rewards = get_reward(add_scores, dones)

                pool_size = Tetromino.pool_size()

                q = rewards + model(possible_states, training=False).numpy()
                for j in range(len(dones)):
                    if dones[j]:
                        q[j] = rewards[j]
                best = tf.argmax(q).numpy()[0] + 0

                if is_include_hold and not is_new_hold:
                    possible_states[1][:-1, -pool_size:] = 0
                else:
                    possible_states[1][:, -pool_size:] = 0

                rand_fl = random.random()
                if rand_fl > epsilon:
                    chosen = best
                else:
                    chosen = random.randint(0, len(dones) - 1)

                #print(f"Shape of s: {np.array(s).shape}")
                # print(f"Shape of possible_states[0][chosen]: {np.array(possible_states[0][chosen]).shape}")
                # print(f"Shape of possible_states[1][chosen]: {np.array(possible_states[1][chosen]).shape}")
                episode_data.append((s[0],  # main grid
                         s[1],  # hold/next piece info
                         possible_states[0][chosen],  # next main grid
                         possible_states[1][chosen],  # next hold/next piece info
                         add_scores[chosen], 
                         dones[chosen]))

                if add_scores[chosen] != int(add_scores[chosen]):
                    t_spins += 1
                    episode_t_spins += 1

                try:
                    state, reward, done, info = env.step(chosen=chosen)
                    episode_score += reward
                    episode_lines += env.current_state.lines - episode_lines
                    #env.render()  # Render the game state
                except Exception as e:
                    print(f"Worker {proc_num}: Error during game step. Error: {str(e)}")
                    raise

                if done or step == max_steps_per_episode - 1:
                    data += episode_data
                    total_score += episode_score
                    print(f"Worker {proc_num}, Episode {episode + 1}: Score = {episode_score:.2f}, Lines = {episode_lines}, T-spins = {episode_t_spins}")
                    break

            if len(data) > target_size:
                print(f'Worker {proc_num}: Reached target size. Episodes:{episode + 1}, Avg score:{total_score / (episode + 1):.2f}, Data size:{len(data)}, T-spins:{t_spins}')
                avg_score = total_score / (episode + 1)
                break

            if episode % 50 == 0:
                elapsed_time = time.time() - start_time
                print(f"Worker {proc_num}: Processed {episode} episodes in {elapsed_time:.2f} seconds. Current data size: {len(data)}")

        print(f"Worker {proc_num} finished. Data size: {len(data)}. Total time: {time.time() - start_time:.2f} seconds")
        if queue is not None:
            queue.put((data, avg_score))
        return data, avg_score
    except Exception as e:
        print(f"Worker {proc_num} encountered an error: {str(e)}")
        import traceback
        traceback.print_exc()
        if queue is not None:
            queue.put((str(e), None))
        return None, None


    
# def collect_samples_multiprocess_queue(model_filename, outer=0, target_size=10000):
#     print(f"Starting data collection for outer={outer}, target_size={target_size}")
#     print("Using 1 CPU core")
    
#     start_time = time.time()
    
#     try:
#         data, avg_score = get_data_from_playing_cnn2d(model_filename, target_size, 1000, 0, None)
#         if data is None:
#             print("Error: No data collected.")
#             return []
#         print(f'Data collection finished. Data length: {len(data)} | avg score: {avg_score}')
#         return data
#     except Exception as e:
#         print(f"Error in data collection: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return []
#     finally:
#         print(f'Total time: {time.time() - start_time:.2f} seconds')

from functools import partial

def collect_samples_multiprocess_queue(model_filename, outer=0, target_size=10000 , headless=True):
    print(f"Starting data collection for outer={outer}, target_size={target_size}")
    timeout = 3600  # 1 hour
    cpu_count = min(multiprocessing.cpu_count(), CPU_MAX)
    print(f"Using {cpu_count} CPU cores")
    
    start_time = time.time()
    all_data = []
    all_scores = []
    
    for i in range(cpu_count):
        print(f"Starting collection for CPU {i}")
        try:
            data, avg_score = get_data_from_playing_cnn2d(model_filename, target_size // cpu_count, 1000, i, None, headless=True)
            if data is None:
                print(f"Error: No data collected for CPU {i}")
                continue
            all_data.extend(data)
            all_scores.append(avg_score)
            print(f'CPU {i} finished. Data length: {len(data)} | avg score: {avg_score}')
        except Exception as e:
            print(f"Error in data collection for CPU {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not all_data:
        print("Error: No data collected from any CPU")
        return []
    
    global current_avg_score
    current_avg_score = max(all_scores) if all_scores else 0
    
    print(f'Data collection finished. Total data length: {len(all_data)} | max avg score: {current_avg_score}')
    print(f'Total time: {time.time() - start_time:.2f} seconds')
    
    return all_data
        
def train(model, outer_start=0, outer_max=100):
    inner_max = 2
    epoch_training = 2
    batch_training = 512

    buffer_new_size = 20
    buffer_outer_max = 1
    history = None

    for outer in tqdm(range(outer_start + 1, outer_start + 1 + outer_max), desc="Outer iterations"):
        print(f'\n======== outer = {outer}/{outer_start + outer_max} ========')
        time_outer_begin = time.time()

        print("Collecting data...")
        buffer = list()

        new_buffer = collect_samples_multiprocess_queue(model_filename=FOLDER_NAME + f'whole_model/outer_{outer - 1}.keras',
                                                        outer=outer - 1, target_size=buffer_new_size , headless=True)
        if new_buffer:
            save_buffer_to_file(FOLDER_NAME + f'dataset/buffer_{outer}.pkl', new_buffer)
            buffer += new_buffer

        for i in tqdm(range(max(1, outer - buffer_outer_max + 1), outer), desc="Loading additional samples"):
            try:
                buffer += load_buffer_from_file(filename=FOLDER_NAME + f'dataset/buffer_{i}.pkl')
            except FileNotFoundError:
                print(f"Warning: Buffer file for outer {i} not found. Skipping.")

        if not buffer:
            print("Error: No data collected or loaded. Skipping this iteration.")
            continue

        random.shuffle(buffer)

        print("Processing buffer...")
        s1, s2, s1_, s2_, r_, dones_ = process_buffer_best(buffer)
        
        print(f"Shapes: s1 {s1.shape}, s2 {s2.shape}, s1_ {s1_.shape}, s2_ {s2_.shape}, r_ {r_.shape}, dones_ {dones_.shape}")

        buffer_size = len(buffer)
        new_buffer_size = len(new_buffer)
        del buffer
        del new_buffer

        for inner in range(inner_max):
            print(f"      ======== inner = {inner + 1}/{inner_max} =========")
            target = list()
            total_batches = int(s1.shape[0] / batch_training) + 1
            for i in tqdm(range(total_batches), desc="Calculating target"):
                start = i * batch_training
                end = min((i + 1) * batch_training, s1.shape[0])
                target.append(
                    model([s1_[start:end], s2_[start:end]], training=False).numpy().reshape(-1) + r_[start:end].reshape(-1))

            target = np.concatenate(target)
            
            print("Adjusting target for game over states...")
            for i in tqdm(range(len(dones_)), desc="Adjusting target"):
                if dones_[i]:
                    target[i] = r_[i]

            target = target * gamma

            print("Training model...")
            history = model.fit([s1, s2], target, batch_size=batch_training, epochs=epoch_training, verbose=1)
            print(f'      loss = {history.history["loss"][-1]:8.3f}   mse = {history.history["mean_squared_error"][-1]:8.3f}')

        print("Saving model...")
        model.save(FOLDER_NAME + f'whole_model/outer_{outer}.keras')

        time_outer_end = time.time()
        text_ = f'outer = {outer:>4d}/{outer_start + outer_max:>4d} | pre-training avg score = {current_avg_score:>8.3f} | loss = {history.history["loss"][-1]:>8.3f} | mse = {history.history["mean_squared_error"][-1]:>8.3f} |' \
                f' dataset size = {buffer_size:>7d} | new dataset size = {new_buffer_size:>7d} | time elapsed: {time_outer_end - time_outer_begin:>6.1f} sec | coef = {reward_coef} | penalty = {penalty:>7d} | gamma = {gamma:>6.3f}\n'
        append_record(text_)
        print('   ' + text_)

    print("Training completed successfully")

def save_buffer_to_file(filename, buffer):
    from pathlib import Path
    Path(FOLDER_NAME + 'dataset').mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)

def load_buffer_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def process_buffer_best(buffer):
    if not buffer:
        return [], [], [], [], [], []

    s1, s2, s1_, s2_, add_scores, dones_ = zip(*buffer)

    # Xử lý s1 và s2
    s1 = np.array([np.array(s).reshape(20, 10, 1) for s in s1])
    s2 = np.array([np.array(s).reshape(-1) for s in s2])

    # Xử lý s1_ và s2_
    s1_ = np.array([np.array(s).reshape(20, 10, 1) for s in s1_])
    s2_ = np.array([np.array(s).reshape(-1) for s in s2_])

    # Chuyển đổi add_scores và dones_ thành numpy arrays
    add_scores = np.array(add_scores)
    dones_ = np.array(dones_)
    
    r_ = get_reward(add_scores, dones_)
    
    print(f"Shapes after processing:")
    print(f"s1: {s1.shape}, s2: {s2.shape}")
    print(f"s1_: {s1_.shape}, s2_: {s2_.shape}")
    print(f"r_: {r_.shape}, dones_: {dones_.shape}")
    
    return s1, s2, s1_, s2_, r_, dones_
def check_same_state(s1, s2):
    s1_ = s1.reshape(-1)
    s2_ = s2.reshape(-1)
    return np.array_equal(s1_, s2_)

def append_record(text, filename=None):
    if filename is None:
        filename = FOLDER_NAME + 'record.txt'
    with open(filename, 'a') as f:
        f.write(text)

def get_reward(add_scores, dones):
    reward = list()
    for i in range(len(add_scores)):
        add_score = add_scores[i]

        if dones[i]:
            add_score += penalty
        reward.append(add_score)
    return np.array(reward).reshape([-1, 1])

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    

    # Check Game import
    try:
        from game import Game
        print("Game imported successfully")
    except Exception as e:
        print(f"Error importing Game: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)  # Exit if Game can't be imported

    # Check Game initialization
    try:
        test_game = Game()
        print("Game initialized successfully")
    except Exception as e:
        print(f"Error initializing Game: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)  # Exit if Game can't be initialized

    if MODE == 'human_player':
        game = Game(gui=Gui(), seed=None)
        game.restart()
        game.run()
    elif MODE == 'ai_player_training':
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # Tạo thư mục để lưu mô hình nếu chưa tồn tại
        os.makedirs(os.path.dirname(FOLDER_NAME + 'whole_model/'), exist_ok=True)
        
        model_path = FOLDER_NAME + 'whole_model/outer_0.keras'
        
        if OUT_START == 0 or not os.path.exists(model_path):
            print("Creating a new model...")
            model_load = make_model_conv2d()
            model_load.compile(
                optimizer=keras.optimizers.Adam(0.001),
                loss='huber',
                metrics=['mean_squared_error']
            )
            model_load.save(model_path)
            print(f'Initial model has been created and saved at {model_path}')
        else:
            print(f"Loading existing model from {model_path}")
            model_load = keras.models.load_model(model_path)
        
        print("Testing model loading in main thread...")
        try:
            test_model = keras.models.load_model(model_path)
            print("Model loaded successfully in main thread")
            test_model.summary()
        except Exception as e:
            print(f"Error loading model in main thread: {str(e)}")
            print(f"Error type: {type(e).__name__}")
        
        try:
            print("Starting training...")
            train(model_load, outer_start=OUT_START, outer_max=OUTER_MAX)
            print("Training completed successfully")
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            import traceback
            traceback.print_exc()

    elif MODE == 'ai_player_watching':
        try:
            model_load = keras.models.load_model(FOLDER_NAME + f'whole_model/outer_{OUT_START}.keras')
        except:
            print(f"Couldn't load model. Please ensure you have trained the model first.")
            exit(1)
        test(model_load, mode='step', is_gui_on=True)
    else:
        print(f"Unknown MODE: {MODE}")