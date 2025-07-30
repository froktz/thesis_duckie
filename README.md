# Duckietown RL Agent – Master's Thesis Project

This repository contains the first part of my Master's thesis project, developed as part of the Duckietown platform. The goal is to train a reinforcement learning (RL) agent to navigate autonomously within a simulated Duckietown environment, respecting the road structure and lane constraints.

## 🧠 Objective

The current objective is to teach the agent how to:
- Stay within the lane boundaries
- Navigate the map freely and safely
- Learn general road-following behavior without a predefined path

Future stages of the thesis will involve more complex tasks such as:
- Route planning
- Decision-making at intersections
- Goal-oriented navigation and optimization

## 🚗 Environment

The project is based on the [Duckietown Gym](https://github.com/duckietown/gym-duckietown) simulator. Custom wrappers and reward functions are used to facilitate training and better reflect the desired behaviors.

## 🛠️ Features

- DQN-based RL agent using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Custom observation space (e.g. vectorized state features like lateral offset, heading angle, and goal distance)
- Dynamic goal tile generation
- Reward shaping to encourage lane following and goal-seeking behavior
- Visual feedback and logging tools

## 🧪 Training Notes

The agent is trained on a custom 5x5 map, with varied tile types including straights, curves, and intersections. The main focus is on learning **generalizable road behavior** rather than overfitting to a specific route.

## 📁 Project Structure

