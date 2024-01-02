import torch
import torchaudio
from model import ConvAutoEncoder, RunningMode
from wav_dataset import WAVDataset
from torch.utils.data import DataLoader

# Constants and Configuration
FRAME_LENGTH = 1600  # 100 ms at 16 kHz
OVERLAP = 0
MODEL_PATH = 'model.pt'
TEST_CLEAN_PATH = ['./audio/test/clean']
TEST_NOISY_PATH = ['./audio/test/noisy']
SAMPLE_RATE = 16000
BATCH_SIZE = 1

def reconstruct_full_audio(decompressed_frames: list, frame_length: int, overlap: int) -> torch.Tensor:
    """
    Reconstruct the full audio from decompressed frames.

    Args:
        decompressed_frames (list): List of decompressed audio frames.
        frame_length (int): Length of each frame.
        overlap (int): Overlap length between frames.

    Returns:
        torch.Tensor: Reconstructed full audio.
    """
    num_frames = len(decompressed_frames)
    full_length = frame_length * num_frames - overlap * (num_frames - 1)
    full_audio = torch.zeros(full_length)

    start = 0
    for frame in decompressed_frames:
        end = start + frame_length
        # Overlap handling by averaging the overlapped segments
        if start != 0:  # if not the first frame
            test1 = full_audio[start:start + overlap]
            test2 = frame.squeeze()[:overlap]
            full_audio[start:start + overlap] =  frame.squeeze()[:overlap]
        full_audio[start:end] = frame.squeeze()
        start = start + frame_length - overlap

    return full_audio

def compress(model: ConvAutoEncoder, audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compress the audio using the model's encoder.

    Args:
        model (ConvAutoEncoder): The ConvAutoEncoder model.
        audio_tensor (torch.Tensor): The audio tensor to compress.

    Returns:
        torch.Tensor: Compressed audio tensor.
    """
    model.set_mode(RunningMode.encoder)
    with torch.no_grad():
        return model(audio_tensor)

def decompress(model: ConvAutoEncoder, audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Decompress the audio using the model's decoder.

    Args:
        model (ConvAutoEncoder): The ConvAutoEncoder model.
        audio_tensor (torch.Tensor): The compressed audio tensor to decompress.

    Returns:
        torch.Tensor: Decompressed audio tensor.
    """
    model.set_mode(RunningMode.decoder)
    with torch.no_grad():
        return model(audio_tensor)

def main():
    # Load the model
    model = ConvAutoEncoder()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.display()

    # Prepare the dataset and dataloader
    test_dataset = WAVDataset(clean_path=TEST_CLEAN_PATH, noisy_path=TEST_NOISY_PATH, noise=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Process the audio frames
    decompressed_frames = []
    for audio_data, _ in test_loader:
        compressed = compress(model, audio_data)
        decompressed = decompress(model, compressed)
        decompressed_frames.append(decompressed)

    # Reconstruct and save the full audio
    full_audio = reconstruct_full_audio(decompressed_frames, FRAME_LENGTH, OVERLAP)
    torchaudio.save(f"./audio/results/full_reconstructed_audio.wav", full_audio.unsqueeze(0), SAMPLE_RATE)

if __name__ == "__main__":
    main()
