# Extended-ROS-Jackal-Environment

This repository contains an extension to the environment setup in the original [ROS-Jackal](https://github.com/Daffan/ros_jackal) repository. The extension enhances the TD3-based reinforcement learning model by improving its generalization capability for robot navigation in diverse and unpredictable environments.

## Overview of the Extension

The key improvement in this repository is the introduction of randomized start and goal points during training. Unlike conventional setups that rely on fixed navigation paths, this approach dynamically alters the robot's path in each episode. By doing so, the model gains exposure to a wider range of navigation scenarios, making it more adaptable to unseen environments.

### Key Features:

- **Randomized Start and Goal Points:** Enhances adaptability by preventing overfitting to specific paths.
- **Improved Generalization:** Helps the robot handle new, previously unseen navigation challenges more effectively.
- **Custom 16-Scenario Environment:** Introduces 16 unique static environments, each measuring 16m x 16m, designed to provide a diverse range of navigation challenges with randomized start and goal points.

## Repository Contents

This repository includes modifications in the following files from the original [ROS-Jackal](https://github.com/Daffan/ros_jackal) repository:

### Environment Files (Located in the `envs` folder before extension):

- `gazebo_simulation.py`
- `jackal_gazebo_envs.py`
- `motion_control_envs.py`

### Main Package Files (Located in the root folder of `ros_jackal` before extension):

- `plot_points3.py`: Added for visualization improvements and better tracking of randomized navigation points.

### Worlds Files (Located in the `jackal_helper` package of the original repository before extension):

- Updates and modifications in world definition files to enhance the diversity of training environments.
- Addition of 16 newly designed static environments with randomized start and goal points for enhanced generalization.

### Model Files (Located in the `model` folder in this repository):

- Extended TD3 model with improved generalization capability for robot navigation.
- Updated policy and critic network structures to handle dynamic navigation scenarios.
- Pretrained models available for evaluation and fine-tuning.
- **TensorBoard logs (`events.out.tfevents.1726307412.user-System-Product-Name`) for tracking training time and performance metrics.**

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
