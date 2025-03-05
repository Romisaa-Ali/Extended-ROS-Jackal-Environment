# Extended-ROS-Jackal-Environment


This repository contains an extension to the environment setup in the original [ROS-Jackal](https://github.com/Daffan/ros_jackal) repository. The extension enhances the TD3-based reinforcement learning model by improving its generalization capability for robot navigation in diverse and unpredictable environments.

## Overview of the Extension

The key improvement in this repository is the introduction of randomized start and goal points during training. Unlike conventional setups that rely on fixed navigation paths, this approach dynamically alters the robot's path in each episode. By doing so, the model gains exposure to a wider range of navigation scenarios, making it more adaptable to unseen environments.

### Key Features:

- **Randomized Start and Goal Points:** Enhances adaptability by preventing overfitting to specific paths.
- **Improved Generalization:** Helps the robot handle new, previously unseen navigation challenges more effectively.
- **Bridging Simulation and Reality:** Supports a more robust training strategy that aligns better with real-world deployment.

## Repository Contents

This repository includes modifications in the following files from the original [ROS-Jackal](https://github.com/Daffan/ros_jackal) repository:

### Environment Files (Located in the `envs` folder):

- `gazebo_simulation.py`
- `jackal_gazebo_envs.py`
- `motion_control_envs.py`

### Main Package Files (Located in the root folder of `ros_jackal`):

- `plot_points3.py`

### Worlds Files (Located in the `jackal_helper` package):

- Updates and modifications in world definition files to enhance the diversity of training environments.

## Installation & Usage

1. Clone this repository:
   ```bash
   git clone <your-new-repo-url>
   cd <your-new-repo>
   ```
2. Ensure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Run your training with the updated environment:
   ```bash
   python train.py --config configs/e2e_default_TD3.yaml
   ```

## Notes

This extension builds on the existing [ROS-Jackal](https://github.com/Daffan/ros_jackal) framework and is intended to improve the adaptability and robustness of reinforcement learning-based navigation systems. It retains compatibility with the original repository’s training and testing pipelines.

For further details, refer to the original [ROS-Jackal repository](https://github.com/Daffan/ros_jackal).
