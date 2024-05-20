 print("---Launching MineRL enviroment (be patient)---")
 obs = env.reset()

 while True:
     minerl_action = agent.get_action(obs)
     obs, reward, done, info = env.step(minerl_action)
     env.render()

def main(model, weights, video_path, json_path, n_batches, n_frames):
    print(MESSAGE)
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    required_resolution = ENV_KWARGS["resolution"]
    cap = cv2.VideoCapture(video_path)

    json_index = 0
    with open(json_path) as json_file:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)

    for _ in range(n_batches):
        th.cuda.empty_cache()
        print("=== Loading up frames ===")
        frames = []
        recorded_actions = []
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            assert frame.shape[0] == required_resolution[1] and frame.shape[1] == required_resolution[0], "Video must be of resolution {}".format(required_resolution)
            # BGR -> RGB
            frames.append(frame[..., ::-1])
            env_action, _ = json_action_to_env_action(json_data[json_index])
            recorded_actions.append(env_action)
            json_index += 1
        frames = np.stack(frames)
        print("=== Predicting actions ===")
        predicted_actions = agent.predict_actions(frames)

        for i in range(n_frames):
            frame = frames[i]
            recorded_action = recorded_actions[i]
            cv2.putText(
                frame,
                f"name: prediction (true)",
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            for y, (action_name, action_array) in enumerate(predicted_actions.items()):
                current_prediction = action_array[0, i]
                cv2.putText(
                    frame,
                    f"{action_name}: {current_prediction} ({recorded_action[action_name]})",
                    (10, 25 + y * 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1
                )
            # RGB -> BGR again...
            cv2.imshow("MineRL IDM model predictions", frame[..., ::-1])
            cv2.waitKey(0)
    cv2.destroyAllWindows()
