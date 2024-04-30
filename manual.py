import time
import gymnasium as gym
import pygame

    
env = gym.make("CartPole-v1", render_mode="human")
env.reset()
time.sleep(1)
score = 0
while True:
   
    pressed_keys = pygame.key.get_pressed()
    
    env.render()

    action = 1 if pressed_keys[pygame.K_SPACE] else 0

    # Run step
    _, _, done, _, _ = env.step(action)
    if done:
        break
    time.sleep(0.1)

    score += 1

print(f"Your score: {score}")