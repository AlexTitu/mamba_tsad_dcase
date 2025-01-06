import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, Dataset
import librosa
from typing import Tuple, Dict


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class AudioAnomalyDataset(Dataset):
    """Custom Dataset for loading and processing audio data for anomaly detection with window-based sampling."""

    def __init__(
            self,
            dataset: Tuple[any, any],
            sample_rate: int = 16000,
            window_length: int = 62,
            overlap_factor: float = 0.5,
            extension: str = '.wav',
            standardize: bool = False,
            scaling_range: Tuple[float, float] = (0, 1),
            is_testing: bool = False
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        dataset : Tuple[str, str]
            Dataset sources for normal and anomalous samples.
        sample_rate : int, optional
            Sampling rate for audio files, by default 16000.
        window_length : int, optional
            Number of samples in each window, by default 8000.
        overlap_factor : float, optional
            Overlap ratio between consecutive windows (0.0 to 1.0), by default 0.5.
        extension : str, optional
            File extension to consider, by default '.wav'.
        standardize : bool, optional
            Whether to apply min-max scaling, by default False.
        scaling_range : Tuple[float, float], optional
            Range for scaling, by default (0, 1).
        """
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.overlap_factor = overlap_factor
        self.extension = extension
        self.standardize = standardize
        self.scaling_range = scaling_range
        self.dataset_source = dataset[0]  # Normal audio files
        self.dataset_target = dataset[1]  # Anomalous audio files
        self.stride = int(window_length * (1 - overlap_factor))
        self.is_testing = is_testing
        # All files have the same length (160000 samples), so we can compute the number of windows directly
        if extension == '.wav':
            self.num_windows_per_file = (160000 - self.window_length) // self.stride + 1
        else:
            self.num_windows_per_file = (313 - self.window_length) // self.stride + 1
        self.total_files = len(self.dataset_source) + len(self.dataset_target)

    def preGenerateMelSpecs(self):
        if self.extension == '.wav':
            for audioPath in self.dataset_source + self.dataset_target:
                mel_spec = self.logmelspec(audioPath)
                # Get the directory and filename without the extension
                directory, filename = os.path.split(os.path.splitext(audioPath)[0])
                # Save the Mel spectrogram as a numpy array in the same directory as the audio file
                np.save(os.path.join(directory, filename + '.npy'), mel_spec)

        else:
            print(f'Pre-Generation of Mel Specs is not supported!')
            return

        return

    def logmelspec(self, path):
        audioFile, sr = librosa.load(path, sr=16000)

        # Ensure audio is 10 seconds long
        if len(audioFile) < sr * 10:
            audioFile = np.pad(audioFile, (0, sr * 10 - len(audioFile)), constant_values=(0, 0))
        else:
            audioFile = audioFile[:sr * 10]

        # Compute the Mel spectrogram with a fixed number of Mel frequency bands and a fixed hop length
        audiomelspec = librosa.feature.melspectrogram(y=audioFile, sr=sr, n_fft=1024, hop_length=512,
                                                      n_mels=128)  # 1251 / 5049
        audiomelspec_db = librosa.power_to_db(audiomelspec)

        if self.standardize:
            audiomelspec_db = ((audiomelspec_db - np.min(audiomelspec_db)) /
                               (np.max(audiomelspec_db) - np.min(audiomelspec_db)))

        audiomelspec_db = np.array([audiomelspec_db])

        return audiomelspec_db

    def __len__(self) -> int:
        """Return the total number of windows across all audio files."""
        return len(self.dataset_source)+len(self.dataset_target)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int] or tuple[torch.Tensor, torch.Tensor, int, str]:
        """Retrieve a fragment and its label."""
        # Determine which dataset (source or target) the idx belongs to
        if idx < len(self.dataset_source):
            # It's in the source dataset
            path = self.dataset_source[idx]
            origin = 'source'
        else:
            # It's in the target dataset
            idx -= len(self.dataset_source)  # Adjust index for the target dataset
            path = self.dataset_target[idx]
            origin = 'target'

        if self.extension == '.npy':
            # Directly load precomputed spectrogram
            signal = np.load(path)  # Shape: (1, 128, 313) B, M, T
            signal = np.transpose(np.squeeze(signal, 0))  # Shape: (1, 313, 128) B, T, M
        else:
            # Load the signal
            signal, _ = librosa.load(path, sr=self.sample_rate)

            # Pad or trim the signal to 160,000 samples
            if len(signal) < 160000:
                signal = np.pad(signal, (0, 160000 - len(signal)), mode='constant')
            else:
                signal = signal[:160000]  # Crop if longer than 160,000 samples

        # Standardize if required
        if self.standardize:
            min_val, max_val = signal.min(), signal.max()
            signal = (signal - min_val) / (max_val - min_val)
            signal = signal * (self.scaling_range[1] - self.scaling_range[0]) + self.scaling_range[0]

        # Convert to PyTorch tensors
        if self.extension == ".wav":
            real_signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(-1)  # Shape: (window_length, 1)
            target_signal = real_signal.clone()  # Target is the same as input for anomaly detection
        else:
            real_signal = torch.tensor(signal, dtype=torch.float32) # Shape: (window_length)
            target_signal = real_signal.clone()  # Target is the same as input for anomaly detection

        # Set the label (assuming anomalies are in 'anomaly' in the path)
        label = 1 if 'anomaly' in path else 0

        if not self.is_testing:
            return real_signal, target_signal, label
        else:
            return real_signal, target_signal, label, origin


# ensuring even split of classes in train and validation sets
def stratified_split(dataset_source: list, dataset_target: list, validation_size: float = 0.25, random_state: int = 42):
    """
        Stratify split only the file paths for source and target datasets.
        """
    # Split source file paths
    source_train, source_val = train_test_split(
        dataset_source,
        test_size=validation_size,
        random_state=random_state
    )

    # Split target file paths
    target_train, target_val = train_test_split(
        dataset_target,
        test_size=validation_size,
        random_state=random_state
    )

    return source_train, source_val, target_train, target_val


def load_dataset(root_dir: str, dataset_type: Tuple[str, str], extension: str, machineType: str = None):
    """Load dataset paths based on dataset type."""
    dataset_source = []
    dataset_target = []
    machine_data = os.listdir(root_dir)
    for index, machine_folder in enumerate(machine_data):
        folder_type, machine_type = machine_folder.split("_")
        if machineType is not None:
            if folder_type == dataset_type[0] and machine_folder == machineType:
                machine_sounds = os.path.join(root_dir, machine_folder, machine_type, dataset_type[1])
                for recording in os.listdir(machine_sounds):
                    if recording.endswith(extension):
                        if 'source' in recording:
                            dataset_source.append(os.path.join(machine_sounds, recording))
                        elif 'target' in recording:
                            dataset_target.append(os.path.join(machine_sounds, recording))
            else:
                continue
        else:
            if folder_type == dataset_type[0]:
                machine_sounds = os.path.join(root_dir, machine_folder, machine_type, dataset_type[1])
                for recording in os.listdir(machine_sounds):
                    if recording.endswith(extension):
                        if 'source' in recording:
                            dataset_source.append(os.path.join(machine_sounds, recording))
                        elif 'target' in recording:
                            dataset_target.append(os.path.join(machine_sounds, recording))

    return dataset_source, dataset_target


class RecDataset(IterableDataset):
    def __init__(self, data, label, window_length=-1, dtype=np.float32,
                 partition=False, timestamp=None, align="none", normalization_type="min-max",
                 down_sample_rate=None, xscaler=None, output_dim=None, shuffle=True):
        super(RecDataset, self).__init__()
        # preprocessing
        if down_sample_rate is not None and (len(data) // down_sample_rate + 1) >= window_length:
            data = data[::down_sample_rate]
        data = np.nan_to_num(data, 0)
        # T, D
        self.data = data.astype(dtype)
        # normalization
        if xscaler is not None:
            self.data = xscaler.transform(self.data)
        else:
            if normalization_type == "norm":
                xscaler = StandardScaler()
                self.data = xscaler.fit_transform(self.data)
            elif normalization_type == "min-max":
                xscaler = MinMaxScaler()
                self.data = xscaler.fit_transform(self.data)
            else:
                xscaler = None
        self.xscaler = xscaler
        # T, 1
        # 0: normal, 1: outlier
        label = label.astype(np.int32)
        if window_length == -1:
            window_length = self.data.shape[0]
        self.window_length = window_length
        if len(self.data) < self.window_length:
            self.window_length = len(self.data)

        # specify dimension
        self.input_dim = self.data.shape[-1]
        self.output_dim = self.input_dim if output_dim is None else output_dim

        self.sampler = DistributedSampler(partition=partition, shuffle=shuffle)
        self.align = align
        if self.align == "last":
            self.label = label[(window_length-1):]
        elif self.align == "begin":
            self.label = label[:-(window_length-1)]
        elif self.align in ["nonoverlap", "causal_pad", "none"]:
            self.label = label
        else:
            raise ValueError

        self.timestamp = timestamp

    def __len__(self):
        if self.align == "nonoverlap":
            return (len(self.data) // self.window_length) + bool(len(self.data) % self.window_length)
        elif "pad" in self.align:
            return len(self.data)
        else:
            return len(self.data) - (self.window_length - 1)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        indexes = self.sampler.sample(self.data[:self.__len__()])
        for index in indexes:
            if self.align == "nonoverlap":
                start_index = index * self.window_length
                end_index = min(start_index + self.window_length, len(self.data))
                x = self.data[start_index: end_index]
                y = self.data[start_index: end_index, :self.output_dim]
                len_x, len_y = len(x), len(y)
                # padding
                if len_x < self.window_length:
                    x = np.concatenate([x, np.zeros((self.window_length-len_x, x.shape[-1]))], dtype=x.dtype, axis=0)
                    y = x[:, :self.output_dim]
                if self.timestamp is not None:
                    t = self.timestamp[index: end_index, np.newaxis]
                    x = np.append(t, x, axis=1)
            elif self.align == "causal_pad":
                start_index = max(0, index - self.window_length + 1)
                pad_len = max(0, self.window_length - index - 1)
                end_index = index
                x = np.concatenate([np.zeros((pad_len, self.data.shape[-1])), self.data[start_index:end_index+1]],
                                   dtype=self.data.dtype, axis=0)
                y = self.data[index:index+1]
                len_x, len_y = len(x), len(y)
            else:
                # window data
                end_index = min(index + self.window_length, len(self.data))
                x = self.data[index: end_index]
                y = self.data[index: end_index, :self.output_dim]
                len_x, len_y = len(x), len(y)
                if self.timestamp is not None:
                    t = self.timestamp[index: end_index, np.newaxis]
                    x = np.append(t, x, axis=1)
            yield x, y, len_x, len_y