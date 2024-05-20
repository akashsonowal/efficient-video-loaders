import json
import glob
import os
import random
from multiprocessing import Process, Queue, Event

def data_loader_worker(tasks_queue, output_queue, quit_workers_event):
    while True:
        task = tasks_queue.get()
        if task is None:
            break
        trajectory_id, video_path, json_path = task
        video = cv2.VideoCapture(video_path)

        with open(json_path) as json_file:
            json_lines = json_file.readlines()
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)

        for i in range(len(json_data)):
            if quit_workers_event.is_set():
                break
            step_data = json_data[i]

            action, is_null_action = json_action_to_env_action(step_data) # model

            # Read frame even if this is null so we progress forward
            ret, frame = video.read()

            if ret:
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                frame = resize_image(frame, AGENT_RESOLUTION)
                output_queue.put((trajectory_id, frame, action), timeout=QUEUE_TIMEOUT)
            else:
                print(f"Could not read frame from video {video_path}")

        video.release()
        if quit_workers_event.is_set():
            break
     output_queue.put(None)

class DataLoader:
    def __init__(self, dataset_dir, n_workers=8, batch_size=8, n_epochs=1, max_queue_size=16):
        assert n_workers >= batch_size, "Number of workers must be equal or greater than batch size"
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = unique_ids
        # Create tuples of (video_path, json_path) for each unique_id
        demonstration_tuples = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".jsonl"))
            demonstration_tuples.append((video_path, json_path))

        assert n_workers <= len(demonstration_tuples), f"n_workers should be lower or equal than number of demonstrations {len(demonstration_tuples)}"

        # Repeat dataset for n_epochs times, shuffling the order for
        # each epoch
        self.demonstration_tuples = []
        for i in range(n_epochs):
            random.shuffle(demonstration_tuples)
            self.demonstration_tuples += demonstration_tuples
        
        self.task_queue = Queue()
        self.n_steps_processed = 0
        for trajectory_id, task in enumerate(self.demonstration_tuples):
            self.task_queue.put((trajectory_id, *task))
        for _ in range(n_workers):
            self.task_queue.put(None)

        self.output_queues = [Queue(maxsize=max_queue_size) for _ in range(n_workers)]
        self.quit_workers_event = Event()
        self.processes = [
            Process(
                target=data_loader_worker,
                args=(
                    self.task_queue,
                    output_queue,
                    self.quit_workers_event,
                ),
                daemon=True
            )
            for output_queue in self.output_queues
        ]
        for process in self.processes:
            process.start()
    
    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = []
        batch_actions = []
        batch_episode_id = []

        for i in range(self.batch_size):
            workitem = self.output_queues[self.n_steps_processed % self.n_workers].get(timeout=QUEUE_TIMEOUT)
            if workitem is None:
                # Stop iteration when first worker runs out of work to do.
                # Yes, this has a chance of cutting out a lot of the work,
                # but this ensures batches will remain diverse, instead
                # of having bad ones in the end where potentially
                # one worker outputs all samples to the same batch.
                raise StopIteration()
            trajectory_id, frame, action = workitem
            batch_frames.append(frame)
            batch_actions.append(action)
            batch_episode_id.append(trajectory_id)
            self.n_steps_processed += 1
        return batch_frames, batch_actions, batch_episode_id
    
    def __del__(self):
        for process in self.processes:
            process.terminate()
            process.join()