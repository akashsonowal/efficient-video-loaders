 print("---Launching MineRL enviroment (be patient)---")
    obs = env.reset()

    while True:
        minerl_action = agent.get_action(obs)
        obs, reward, done, info = env.step(minerl_action)
        env.render()