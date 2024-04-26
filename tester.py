import gymnasium as gym

# Create and wrap the environment
env = gym.make('Walker2d-v4', render_mode="human")

total_num_episodes = 1000


for episode in range(total_num_episodes):
    # gymnasium v26 requires users to set seed while resetting the environment
    obs, info = env.reset()

    done = False
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    env.render()