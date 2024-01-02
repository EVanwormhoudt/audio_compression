import time
import torch
from torch.utils.data import DataLoader
from model import ConvAutoEncoder
from own.neuralcomp.loss import BalancedSpeechLoss
from wav_dataset import WAVDataset
from tqdm import tqdm



LEARNING_RATE = 0.001
N_EPOCHS = 100
CLEAN_TRAIN_PATH = ['./audio/train/clean']
NOISY_TRAIN_PATH = ['./audio/train/noisy']
CLEAN_EVAL_PATH = ['./audio/eval/clean']
NOISY_EVAL_PATH = ['./audio/eval/noisy']
NOISY = False



def train(model: ConvAutoEncoder, data_loader: DataLoader, optimizer: torch.optim.Optimizer,
          loss_fn: BalancedSpeechLoss, device: torch.device, is_noisy: bool = False) -> float:
    """
    Train the model on noisy or clean data.

    Args:
        model: The model to be trained.
        data_loader: DataLoader for training data.
        optimizer: Optimizer for model training.
        loss_fn: Loss function used in training.
        device: Device on which the model is trained.
        is_noisy: Flag to indicate if training is on noisy data.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        inputs, target = batch if is_noisy else (batch[0], batch[0])
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss, *rest = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def evaluate(model: ConvAutoEncoder, data_loader: DataLoader, loss_fn: BalancedSpeechLoss,
             device: torch.device, is_noisy: bool = False) -> float:
    """
    Evaluate the model on noisy or clean data.

    Args:
        model: The model to be evaluated.
        data_loader: DataLoader for evaluation data.
        loss_fn: Loss function used in evaluation.
        device: Device on which the model is evaluated.
        is_noisy: Flag to indicate if evaluation is on noisy data.

    Returns:
        Average loss for the epoch.
    """
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, target = batch if is_noisy else (batch[0], batch[0])
            inputs, target = inputs.to(device), target.to(device)

            predictions = model(inputs)
            loss, *_ = loss_fn(predictions, target)
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def epoch_time(start_time: float, end_time: float) -> tuple:
    """
    Calculate elapsed time for an epoch.

    Args:
        start_time: Start time of the epoch.
        end_time: End time of the epoch.

    Returns:
        Tuple of elapsed minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Data preparation
    print("Loading data...")
    train_dataset = WAVDataset(noisy_path=['./audio/train/noisy'], clean_path=['./audio/train/clean'], noise=NOISY)
    eval_dataset = WAVDataset(noisy_path=['./audio/eval/noisy'], clean_path=['./audio/eval/clean'], noise=NOISY)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    # Model setup
    print("Creating Model...")
    model = ConvAutoEncoder().to(device)
    print(f'Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Training configuration

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = BalancedSpeechLoss().to(device)

    # Training loop

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device, is_noisy=NOISY)
        valid_loss = evaluate(model, eval_loader, loss_fn, device, is_noisy=NOISY)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

if __name__ == "__main__":
    main()
