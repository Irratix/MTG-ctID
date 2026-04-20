import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from src.models.modelconfig import ModelConfig
import torch

def plot_losses(losses):
    train_losses = [l["training"] for l in losses]
    val_losses = [l["valuation"] for l in losses]
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def train(config: ModelConfig, device, train_loader, val_loader, criterion, types):
    print("training model", config.name)

    losses = []
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    epoch_bar = tqdm(range(1, config.num_epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        # train
        config.model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Train", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            config.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = config.model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(config.optimizer)
            scaler.update()

            train_loss += loss.item()

        # validate
        config.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                for images, labels in tqdm(val_loader, desc="Val", leave=False):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = config.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        epoch_bar.write(f"{epoch} — train loss: {avg_train_loss:.4f} — val loss: {avg_val_loss:.4f}")
        losses.append({"training": avg_train_loss, "valuation": avg_val_loss})

        # unfreeze layers at threshold
        if epoch in config.unfreeze_schedule:
            print("unfreezing layer")
            for param in config.unfreeze_schedule[epoch].parameters():
                param.requires_grad = True

        # save model after epoch
        path = Path(f"{config.checkpoint_folder}/e_{epoch}_tl_{avg_train_loss:.3f}_vl_{avg_val_loss:.3f}.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': config.model,
            'all_types': types,
            'epoch': epoch,
            'val_loss': avg_val_loss,
        }, path)
    
    plot_losses(losses)