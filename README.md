# Autonomous-Robotic-Car-With-CNN-Obstacle-Avoidence

An autonomous driving car that is powered by CNNs and trained on collected data to avoid obstacles.

## About The Project

This project implements an autonomous driving system on the PiCar-X robot platform using a designed Convolutional Neural Network. It accepts real-time camera input to determine the driving, and the robot is able to drive autonomously and avoid obstacles. The training data of the CNN model were collected personally, so the model is specific to the environment and driving condition of the robot.

### Key Features

- **Real-time CNN inference** - Processes camera feed for immediate decision making
- **Custom data collection system** - Includes tools for gathering and labeling training data
- **WebSocket remote control** - Allows manual control for data collection and testing

### Built With

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [PiCar-X](https://www.sunfounder.com/) - Robotic car platform

## Getting Started

### Prerequisites

- PiCar-X robot with camera module
- Raspberry Pi with WiFi capability

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/Autonomous-Robotic-Car-With-CNN-Obstacle-Avoidence.git
   cd Autonomous-Robotic-Car-With-CNN-Obstacle-Avoidence
   ```

2. Install required packages
   ```bash
   pip install torch torchvision opencv-python pillow numpy pandas matplotlib scikit-learn
   pip install picarx robot_hat vilib sunfounder_controller websocket-client keyboard
   ```

3. Create data directory for training images
   ```bash
   mkdir data
   ```

## Usage

### 1. Data Collection

First, collect training data by manually controlling the robot:

```bash
# Run the data collection script on the robot
python car_response_script.py

# Run the remote control script on your computer
python car_control_script.py
```

Use WASD keys to control the robot:
- **W** - Forward
- **S** - Backward  
- **A** - Left (saves frame with 'left' label)
- **D** - Right (saves frame with 'right' label)

The system automatically saves camera frames and labels when turning left or right.

### 2. Train the Model

After collecting sufficient training data:

```bash
python neural_training.py
```

This will:
- Load and preprocess your collected images
- Train the CNN model using your data
- Save the trained model weights as `autonomous_model_weights.pth`
- Display training/validation accuracy plots

### 3. Run Autonomous Mode

Once the model is trained, run the autonomous driving system:

```bash
python autonomous_car.py
```
## Model Architecture

The CNN uses a custom architecture optimized for real-time driving decisions:

- **Input**: 32x32 RGB images from robot camera
- **Conv Layers**: 3 convolutional layers with ReLU activation and max pooling
- **FC Layers**: 3 fully connected layers with dropout for regularization
- **Output**: 2-class classification (left/right steering decisions)

## Known Issues (Work In Progress)

-Working on improvements for edge cases and unknown senarious the robot encounters


## 📫 Contact Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Joseph%20Cusumano-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/josephmcusumano)
