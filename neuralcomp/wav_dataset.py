import os
import re
from typing import List, Sequence, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset

# Regular expression for extracting numbers
numbers = re.compile(r'(\d+)')

def fast_scandir(path: str, exts: List[str], recursive: bool = False) -> Tuple[List[str], List[str]]:
    """
    Scans a directory and returns a list of subdirectories and files with specified extensions.

    Args:
        path (str): The directory path to scan.
        exts (List[str]): A list of file extensions to include in the scan.
        recursive (bool): If True, the function will scan subdirectories recursively.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists: one for subdirectories and one for file paths.
    """
    subfolders, files = [], []
    for f in os.scandir(path):
        if f.is_dir():
            subfolders.append(f.path)
        elif f.is_file() and os.path.splitext(f.name)[1].lower() in exts:
            files.append(f.path)
    if recursive:
        for dir in list(subfolders):
            sf, f = fast_scandir(dir, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)
    return subfolders, files

def get_all_wav_filenames(paths: Sequence[str], recursive: bool) -> List[str]:
    """
    Retrieves all WAV and FLAC filenames from the specified directories.

    Args:
        paths (Sequence[str]): A sequence of directory paths.
        recursive (bool): If True, searches directories recursively.

    Returns:
        List[str]: A list of all WAV and FLAC file paths found in the specified directories.
    """
    extensions = [".wav", ".flac"]
    filenames = []
    for path in paths:
        _, files = fast_scandir(path, extensions, recursive=recursive)
        filenames.extend(files)
    return filenames

class WAVDataset(Dataset):
    """
    A PyTorch Dataset for loading audio files in WAV format, with options for frame-based processing.
    """
    def __init__(
            self,
            noisy_path: Union[str, Sequence[str]],
            clean_path: Union[str, Sequence[str]],
            frame_length_ms: int = 100,
            overlap_percent: float = 0,
            sample_rate: int = 16000,
            check_silence: bool = True,
            with_ID3: bool = False,
            recursive: bool = False,
            noise: bool = False
    ):
        """
        Initialize the WAVDataset.

        Args:
            noisy_path: Path(s) to noisy audio files.
            clean_path: Path(s) to clean audio files.
            frame_length_ms: Frame length in milliseconds.
            overlap_percent: Overlap between frames as a percentage.
            sample_rate: Sampling rate of audio.
            check_silence: Whether to check for silence in audio frames.
            with_ID3: Whether to include ID3 metadata from audio files.
            recursive: Whether to search for audio files recursively.
            noise: Whether to include noise in the dataset.
        """
        self.paths = noisy_path if isinstance(noisy_path, (list, tuple)) else [noisy_path]
        self.clean_paths = clean_path if isinstance(clean_path, (list, tuple)) else [clean_path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.sample_rate = sample_rate or 16000  # Default to 16 kHz if not specified
        self.frame_length = int(self.sample_rate * frame_length_ms / 1000)
        self.overlap = int(self.frame_length * overlap_percent / 100)
        self.cumulative_frame_counts = self._calculate_cumulative_frame_counts()
        self.check_silence = check_silence
        self.with_ID3 = with_ID3
        self.noise = noise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the waveform tensor and its corresponding sample rate.
        """
        file_idx, frame_idx = self._get_file_and_frame_index(idx)
        clean_file_idx = numbers.findall(self.wavs[file_idx])[0]
        waveform, sample_rate = self._load_frame(f"{self.clean_paths[0]}/clnsp" + clean_file_idx + ".wav", frame_idx)
        if not self.noise:
            return waveform, sample_rate

        noisy_waveform, sample_rate = self._load_frame(self.wavs[file_idx], frame_idx)
        return noisy_waveform, waveform

    def _load_frame(self, file_path: str, frame_idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load a specific frame from an audio file.

        Args:
            file_path (str): Path to the audio file.
            frame_idx (int): The index of the frame to be loaded.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the waveform tensor of the frame and its sample rate.
        """
        start = frame_idx * (self.frame_length - self.overlap)
        waveform, sample_rate = torchaudio.load(file_path, frame_offset=start, num_frames=self.frame_length)
        return waveform, sample_rate

    def _calculate_cumulative_frame_counts(self) -> List[int]:
        """
        Calculate the cumulative count of frames for each audio file in the dataset.

        Returns:
            List[int]: A list of cumulative frame counts.
        """
        cumulative = 0
        cumulative_frame_counts = []
        for file_path in self.wavs:
            num_frames = self._get_num_frames(file_path)
            cumulative += num_frames
            cumulative_frame_counts.append(cumulative)
        return cumulative_frame_counts

    def _get_file_and_frame_index(self, idx: int) -> Tuple[int, int]:
        """
        Get the file index and frame index within that file corresponding to a dataset index.

        Args:
            idx (int): The dataset index.

        Returns:
            Tuple[int, int]: A tuple containing the file index and the frame index.
        """
        if not self.cumulative_frame_counts:
            raise IndexError("No files available")

        file_idx = self._binary_search_cumulative_frames(idx)
        if file_idx is not None:
            frame_idx = idx - (self.cumulative_frame_counts[file_idx - 1] if file_idx > 0 else 0)
            return file_idx, frame_idx
        else:
            raise IndexError("Index out of range")

    def _binary_search_cumulative_frames(self, idx: int) -> int:
        """
        Perform a binary search to find the file index for a given dataset index.

        Args:
            idx (int): The dataset index.

        Returns:
            int: The file index corresponding to the dataset index.
        """
        low, high = 0, len(self.cumulative_frame_counts)
        while low < high:
            mid = (low + high) // 2
            if self.cumulative_frame_counts[mid] <= idx:
                low = mid + 1
            else:
                high = mid
        if low > len(self.cumulative_frame_counts):
            raise IndexError("Index out of range")

        return low

    def _get_num_frames(self, file_path: str) -> int:
        """
        Calculate the number of frames in an audio file.

        Args:
            file_path (str): The path to the audio file.

        Returns:
            int: The number of frames in the audio file.
        """
        info = torchaudio.info(file_path)
        total_samples = info.num_frames
        return 1 + (total_samples - self.frame_length) // (self.frame_length - self.overlap)

    def __len__(self) -> int:
        """
        Get the total number of frames in the dataset.

        Returns:
            int: The total number of frames.
        """
        total_frames = 0
        for file_path in self.wavs:
            total_frames += self._get_num_frames(file_path)
        return total_frames
