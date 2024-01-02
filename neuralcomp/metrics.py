import numpy as np
from pesq import pesq
from scipy.io import wavfile


from auraloss.time import  SNRLoss,SISDRLoss
import torch

# Read the WAV files
rate, ref = wavfile.read('../audio/metrics/clnsp1.wav')
rate2, deg = wavfile.read('../audio/metrics/full_reconstructed_audio.wav')

ref = ref / np.max(np.abs(ref))
deg = deg / np.max(np.abs(deg))

# Compute the PESQ score
score = pesq(rate, ref, deg, 'wb')  # Use 'wb' for wideband, 'nb' for narrowband
print('PESQ score:', score)

# Make sure the sampling rates are the same
assert rate == rate2

# make them the same length
if len(ref) > len(deg):
    ref = ref[:len(deg)]

else:
    deg = deg[:len(ref)]
# Convert to tensors
ref = torch.from_numpy(ref).float()
deg = torch.from_numpy(deg).float()



l1 = torch.nn.L1Loss()
loss = l1(ref, deg)
print('L1 loss:', loss.item() )

# Compute the SNR loss
snr_loss = SNRLoss()
loss = snr_loss(ref, deg)
print('SNR loss:', loss.item())

# Compute the MelSTFT loss
sisdr_loss = SISDRLoss()
loss = sisdr_loss(ref, deg)
print('SISDR loss:', loss.item())



