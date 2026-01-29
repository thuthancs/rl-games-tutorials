import gymnasium as gym
import flappy_bird_gymnasium


def main():
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    obs, info = env.reset()

    try:
        while True:
            # random actions for now
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
