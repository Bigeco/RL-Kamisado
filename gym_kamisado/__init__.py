from gymnasium.envs.registration import register

register(
    id="Kamisado-v0",
    entry_point="gym_kamisado.envs.kamisado:KamisadoEnv",
    autoreset=True,
    order_enforce=True,
    max_episode_steps=100,
    reward_threshold=1,
)

register(
    id="CliffWalking-v0",
    entry_point="gym_kamisado.envs.test:CliffWalkingEnv"
)
