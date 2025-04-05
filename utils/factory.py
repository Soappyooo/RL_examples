from agents import *
from envs import *
from utils.registry import AGENT_REGISTRY, ENV_REGISTRY


def create_env(config: dict) -> Env:
    """
    Create an environment based on the provided configuration.

    Args:
        config (dict): Configuration dictionary for the environment.

    Returns:
        Env: An instance of the specified environment.
    """

    env_name = config["name"]
    env_class = ENV_REGISTRY.get(env_name)
    if env_class is None:
        raise ValueError(f"Environment '{env_name}' not found in registry.")

    return env_class.from_config(config)  # Assuming from_config method exists in the env class


def create_agent(config: dict) -> Agent:
    """
    Create an agent based on the provided configuration.

    Args:
        config (dict): Configuration dictionary for the agent.

    Returns:
        agents.Agent: An instance of the specified agent.
    """
    agent_name = config["name"]
    agent_class = AGENT_REGISTRY.get(agent_name)
    if agent_class is None:
        raise ValueError(f"Agent '{agent_name}' not found in registry.")

    return agent_class.from_config(config)  # Assuming from_config method exists in the agent class
