# In video dataloaders, a single batch is a batch of frames not batch of videos.

# Video reader for single video

# Video loader for many videos
# Each video has a batch is segmented in chunks. So, many video files will result in n * n_batches per video
# import os
# from decord import VideoLoader
# from decord import VideoReader
# from decord import gpu, cpu

# video_files = os.listdir("data")

# video_file_paths = [os.path.join("data", video_file) for video_file in video_files][:1]

# for video_path in video_file_paths:
#     vr = VideoReader(video_path, ctx=cpu(0))
#     print(len(vr))
#     print(vr[:].shape)

# vl= VideoLoader(video_file_paths, ctx=[cpu(0)], shape=(10, 1080, 1920, 3), interval=1, skip=0, shuffle=0)
# print('Total batches:', len(vl))

# i = 0
# for batch in vl:
#     print(batch[0].shape)
#     i+=1

# print(i)

# print(vl)

# for batch in vl:
#     print(batch[0].shape)

## Large scale video pipeline

# CV2 is sequential reading


import os
import glob
import cv2
from multiprocessing import Queue, Process, Event

QUEUE_TIMEOUT = (
    10  # Queue is shared across processes so that's why need timeout for one process
)


def data_loader_worker(tasks_queue, output_queue, quit_workers_event):
    """
    Runs as an individual process
    """
    while True:
        task = tasks_queue.get()
        if task is None:
            break
        id, video_path = task
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):  # iterate on a single video
            if quit_workers_event.is_set():
                break

            ret, frame = video.read()
            if ret:
                output_queue.put(
                    (id, frame), timeout=QUEUE_TIMEOUT
                )  # handles if queue is full so waits to empty

        video.release()
        if quit_workers_event.is_set():
            break
    output_queue.put(None)


class DataLoader:
    """
    batch size is number of frames to pick to form a random batch.
    """

    def __init__(
        self, dataset_dir, n_workers=2, batch_size=2, n_epochs=1, max_queue_size=4
    ) -> None:
        assert (
            n_workers >= batch_size
        ), "Number of workers must be equal or greater than batch size"
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = unique_ids

        video_paths = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".mp4"))
            video_paths.append(video_path)

        self.task_queue = Queue()  # has video links

        self.n_steps_processed = 0
        for id, video_path in enumerate(video_paths):
            self.task_queue.put((id, video_path))

        for _ in range(n_workers):
            self.task_queue.put(None)

        self.output_queues = [
            Queue(maxsize=max_queue_size) for _ in range(n_workers)
        ]  # each queue is in a worker
        self.quit_workers_event = Event()

        self.processes = [
            Process(
                target=data_loader_worker,
                args=(self.task_queue, output_queue, self.quit_workers_event),
                daemon=True,
            )
            for output_queue in self.output_queues
        ]

        for process in self.processes:
            process.start()

    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = []
        batch_id = []

        for i in range(self.batch_size):
            workitem = self.output_queues[self.n_steps_processed % self.n_workers].get(
                timeout=QUEUE_TIMEOUT
            )  # handles if queue is empty so waits
            if workitem is None:
                raise StopIteration()

            id, frame = workitem
            batch_frames.append(frame)
            batch_id.append(id)
            self.n_steps_processed += 1
        return batch_frames, batch_id

    def __del__(self):
        for process in self.processes:
            process.terminate()
            process.join()


if __name__ == "__main__":
    data_loader = DataLoader(dataset_dir="data", n_workers=6, batch_size=2, n_epochs=1)

    for batch_i, (batch_images, batch_episode_id) in enumerate(data_loader):
        print(batch_i)
        print(batch_episode_id)
        print(len(batch_images))
        print("^^^^^^^^^^^^^^^^^")
