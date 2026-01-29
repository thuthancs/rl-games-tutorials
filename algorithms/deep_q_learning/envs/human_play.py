import gymnasium as gym
import flappy_bird_gymnasium  # registers env
import pygame

NOOP = 0
FLAP = 1


def main():
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    obs, info = env.reset()

    try:
        while True:
            action = NOOP

            # read pygame events (window must have focus)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = FLAP

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
