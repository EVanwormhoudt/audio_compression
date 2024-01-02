import torch
import torch.nn as nn
from own.neuralcomp.audio_2_mel import Audio2Mel

class BalancedSpeechLoss(nn.Module):
    def __init__(self, n_mel_channels=64, sampling_rate=16000):
        """
        Initialize the BalancedSpeechLoss module.

        Args:
            n_mel_channels (int): Number of mel channels.
            sampling_rate (int): Sampling rate of the audio.
        """
        super(BalancedSpeechLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate

    def _frequency_loss(self, predicted_audio, target_audio):
        """
        Calculate frequency loss.

        Args:
            predicted_audio (Tensor): Predicted audio tensor.
            target_audio (Tensor): Target audio tensor.

        Returns:
            Tensor: Computed frequency loss.
        """
        frequency_loss = 0.0
        for i in range(5, 12): # FFT ranges from 2^5 to 2^11 according to Encodec paper
            fft = Audio2Mel(n_fft=2 ** i, win_length=2 ** i, hop_length=(2 ** i) // 4,
                            n_mel_channels=self.n_mel_channels, sampling_rate=self.sampling_rate)
            frequency_loss += self.l1_loss(fft(target_audio), fft(predicted_audio)) \
                              + self.l2_loss(fft(target_audio), fft(predicted_audio))
        return frequency_loss

    def forward(self, predicted_audio, target_audio):
        """
        Forward pass for calculating the loss.

        Args:
            predicted_audio (Tensor): Predicted audio tensor.
            target_audio (Tensor): Target audio tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Total loss, frequency loss, L1 loss.
        """

        # Move tensors to CPU because it doesn't support MPS
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predicted_audio, target_audio = predicted_audio.to(device), target_audio.to(device)

        # Frequency loss computation
        frequency_loss = self._frequency_loss(predicted_audio, target_audio)

        # L1 loss computation, multiplied by 50 to balance the frequency loss
        l1_loss = 50 * self.l1_loss(predicted_audio, target_audio)

        # Total loss
        total_loss = frequency_loss + l1_loss

        return total_loss, frequency_loss, l1_loss
