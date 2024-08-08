# Tetris Deep Q-Learning Project
This project implements a Deep Q-Network (DQN) to play Tetris. The agent learns to play Tetris through reinforcement learning, improving its performance over time.

## 🚀 Features

- Deep Q-Learning implementation for Tetris
- Customizable Tetris environment
- TensorBoard integration for training visualization
- Score logging for performance tracking
- Flexible training parameters

## 📋 Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Parameters](#parameters)
- [Customization](#customization)

## 🛠 Installation

1. Ensure you have Python 3.12 installed on your system.

2. Clone this repository:
   ```bash
   git clone https://github.com/cloudysman/Tetris_AI_Hieu-Tung.git
   cd Tetris_AI_Hieu-Tung
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
tetris-dqn/
│
├── src/
│   ├── tetris.py         
├── train.py             
├── test.py              
├── requirements.txt     
└── README.md            
```

## 🎮 Usage

### Training

To train the agent, run the `train.py` script. You can customize various parameters (see [Parameters](#parameters) section).

Basic usage:
```bash
python train.py
```

Resume training:
```bash
python train.py  --checkpoint_path "path/to/checkpoint/tetris_3000
```

During training, the script will:
- Print progress information to the console
- Save the model periodically (default: every 1000 epochs)
- Log training metrics to TensorBoard
- Save game scores to a CSV file

### Testing

To test the trained agent, use the `test.py` script:

```bash
python test.py
```

This will load the trained model and play Tetris, saving a video of the gameplay.

## 🎛 Parameters

Key parameters for training (in `train.py`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--width` | Width of the Tetris board | 10 |
| `--height` | Height of the Tetris board | 20 |
| `--batch_size` | Number of samples per batch for training | 512 |
| `--lr` | Learning rate | 1e-3 |
| `--num_epochs` | Total number of training epochs | 3000 |
| `--saved_path` | Directory to save trained models | "trained_models" |
| `--score_log_path` | Path to save the score log CSV file | "score_log.csv" |

For a full list of parameters and their descriptions, run:
```bash
python train.py --help
```

## 🔧 Customization

- Modify `src/deep_q_network.py` to experiment with different network architectures.
- Adjust reward calculation in `src/tetris.py` to change the agent's learning objectives.
- Experiment with different hyperparameters in `train.py` to optimize learning.

## 📊 Results

Comming soon

## 📜 License

This project is VST_s2_TH licensed.

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) for the deep learning framework
---

<p align="center">
  Made with ❤️ by Vu Son Tung , Trong Hieu
</p>
