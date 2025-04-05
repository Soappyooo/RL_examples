import os
import time
import logging
import numpy as np
import torch
from utils.factory import create_env, create_agent
from utils.replay_buffer import ReplayBuffer


def seed_all(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_off_policy(config: dict, logger: logging.Logger) -> None:
    """
    Train the off-policy agent using the specified configuration.

    Args:
        config (dict): Configuration dictionary for the environment and agent.
        logger (logging.Logger): Logger instance for logging training progress.
    """

    logger.info(f"Starting training with config: {config}")
    # Set random seed for reproducibility
    seed_all(config["training"]["seed"])

    # Initialize environment and agent
    env = create_env(config["env"])
    agent = create_agent(config["agent"])

    # Initialize replay buffer
    buffer = ReplayBuffer(config["training"]["buffer_size"])

    # Get configs
    num_episodes = config["training"]["num_episodes"]
    batch_size = config["training"]["batch_size"]
    min_buffer_size = config["training"]["min_buffer_size"]
    log_interval = config["training"]["log_interval"]
    save_interval = config["training"]["save_interval"]
    train_interval_steps = config["training"]["train_interval_steps"]
    checkpoint_path = config["training"]["checkpoint_path"]
    checkpoint_path = os.path.abspath(
        os.path.join(
            checkpoint_path, f"{config['env']['name']}_{config['agent']['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    )
    os.makedirs(checkpoint_path, exist_ok=True)
    logger.info(f"Checkpoint save path: {checkpoint_path}")

    initial_act_interval = config["env"]["initial_act_interval"]
    act_interval_decay = config["env"]["act_interval_decay"]
    final_act_interval = config["env"]["final_act_interval"]
    current_act_interval = initial_act_interval

    # Training loop
    agent.train()
    logger.info("Training started.")
    for episode in range(num_episodes):
        state = env.reset()
        action = agent.act(state)
        done = False
        total_reward = 0
        total_loss = 0.0
        update_count = 0

        agent.begin_episode()

        while not done:
            next_state, reward, done, additional_info = env.step(action)
            buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if env.current_step % round(current_act_interval) == 0:
                action = agent.act(state)

            if len(buffer) > max(batch_size, min_buffer_size) and env.current_step % train_interval_steps == 0:
                batch = buffer.sample(batch_size)
                loss = agent.update(agent.buffer_tuple_to_dict(batch))["loss"]
                total_loss += loss
                update_count += 1

        avg_loss = total_loss / update_count if update_count > 0 else 0.0

        if (episode + 1) % log_interval == 0:
            logger.info(
                f"Episode {episode + 1}/{num_episodes}: Total Reward: {total_reward:.4f}, "
                + f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}, "
                + f"Lr: {agent.learning_rate:.6f}, Act Interval: {round(current_act_interval)}"
            )

        if (episode + 1) % save_interval == 0:
            checkpoint_file = os.path.abspath(os.path.join(checkpoint_path, f"checkpoint_{episode + 1}.pth"))
            agent.save_checkpoint(checkpoint_file)
            logger.info(f"Checkpoint saved at episode {episode + 1}, file: {checkpoint_file}")

        agent.end_episode()
        current_act_interval = max(final_act_interval, current_act_interval * act_interval_decay)

    logger.info("Training completed.")


def train_on_policy(config: dict, logger: logging.Logger) -> None:
    """
    Train the on-policy agent (like PPO) using the specified configuration.

    Args:
        config (dict): Configuration dictionary for the environment and agent.
        logger (logging.Logger): Logger instance for logging training progress.
    """
    logger.info(f"Starting on-policy training with config: {config}")
    # Set random seed for reproducibility
    seed_all(config["training"]["seed"])

    # Initialize environment and agent
    env = create_env(config["env"])
    agent = create_agent(config["agent"])

    # Get configs
    num_episodes = config["training"]["num_episodes"]
    horizon = config["training"]["horizon"]  # Number of steps to collect before update
    log_interval = config["training"]["log_interval"]
    save_interval = config["training"]["save_interval"]
    checkpoint_path = config["training"]["checkpoint_path"]
    checkpoint_path = os.path.abspath(
        os.path.join(
            checkpoint_path, f"{config['env']['name']}_{config['agent']['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    )
    os.makedirs(checkpoint_path, exist_ok=True)
    logger.info(f"Checkpoint save path: {checkpoint_path}")

    initial_act_interval = config["env"]["initial_act_interval"]
    act_interval_decay = config["env"]["act_interval_decay"]
    final_act_interval = config["env"]["final_act_interval"]
    current_act_interval = initial_act_interval

    # Training loop
    agent.train()
    logger.info("On-policy training started.")

    episode = 0
    total_steps = 0

    while episode < num_episodes:
        # Buffer for collecting trajectory data
        states_buffer = []
        actions_buffer = []
        rewards_buffer = []
        next_states_buffer = []
        dones_buffer = []
        log_probs_buffer = []

        steps_in_current_batch = 0
        batch_episodes = 0
        batch_total_reward = 0

        # Collect data for one update
        while steps_in_current_batch < horizon and episode < num_episodes:
            state = env.reset()
            episode_reward = 0
            done = False

            agent.begin_episode()

            while not done:
                # Select action
                if env.current_step % round(current_act_interval) == 0:
                    action = agent.act(state)
                    log_prob = agent.get_log_prob()

                # Take step in environment
                next_state, reward, done, additional_info = env.step(action)

                # Store transition
                states_buffer.append(state)
                actions_buffer.append(action)
                rewards_buffer.append(reward)
                next_states_buffer.append(next_state)
                dones_buffer.append(done)
                log_probs_buffer.append(log_prob)

                # Update state and counters
                state = next_state
                episode_reward += reward
                steps_in_current_batch += 1
                total_steps += 1

                # Break if we've collected enough steps
                if steps_in_current_batch >= horizon:
                    break

            # Episode ended
            batch_episodes += 1
            batch_total_reward += episode_reward

            # Adjust action interval
            current_act_interval = max(final_act_interval, current_act_interval * act_interval_decay)

            if done:
                agent.end_episode()
                episode += 1

                if episode % log_interval == 0:
                    logger.info(
                        f"Episode {episode}/{num_episodes}: Total Reward: {episode_reward:.4f}, "
                        + f"Steps: {total_steps}, Act Interval: {round(current_act_interval)}, "
                        + f"LR: {agent.policy_lr:.6f} | {agent.value_lr:.6f}"
                    )

                if episode % save_interval == 0:
                    checkpoint_file = os.path.abspath(os.path.join(checkpoint_path, f"checkpoint_{episode}.pth"))
                    agent.save_checkpoint(checkpoint_file)
                    logger.info(f"Checkpoint saved at episode {episode}, file: {checkpoint_file}")

        # Convert buffers to dictionary for agent update
        buffer_dict = {
            "states": states_buffer,
            "actions": actions_buffer,
            "rewards": rewards_buffer,
            "next_states": next_states_buffer,
            "dones": dones_buffer,
            "log_probs": log_probs_buffer,
        }

        # Update agent policy using collected data
        loss = agent.update(buffer_dict)

        # Log batch information
        logger.info(
            f"Update after {steps_in_current_batch} steps ({batch_episodes} episodes): "
            + f"Avg Episode Reward: {batch_total_reward / batch_episodes:.4f}, "
            + f"Loss: actor {loss["actor_loss"]:.4f}, critic {loss["critic_loss"]:.4f}"
        )

    logger.info("On-policy training completed.")
