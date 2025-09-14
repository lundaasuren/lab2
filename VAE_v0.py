import os
import random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.manifold import TSNE
import umap

# ----------------------------
# Configuration
# ----------------------------
IMG_SIZE = 128
LATENT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
BETA = 0.001  # KL weighting (same as original)
DATA_DIR = "/home/groups/comp3710/OASIS"
TRAIN_DIR = os.path.join(DATA_DIR, "keras_png_slices_train")
VAL_DIR = os.path.join(DATA_DIR, "keras_png_slices_validate")
TEST_DIR = os.path.join(DATA_DIR, "keras_png_slices_test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# Dataset
# ----------------------------
class MRIDataset(Dataset):
    def __init__(self, folder: str, img_size: int = IMG_SIZE, max_samples: int = None):
        self.paths = list(Path(folder).glob("**/*.png"))
        if max_samples:
            self.paths = random.sample(self.paths, min(max_samples, len(self.paths)))
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # returns [C,H,W] floats in [0,1]
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("L")
        img = self.transform(img)  # tensor [1, H, W]
        return img, str(p)

# ----------------------------
# VAE model (Encoder / Decoder)
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim: int, img_size=IMG_SIZE):
        super().__init__()
        # Input: (B,1,IMG_SIZE,IMG_SIZE)
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)   # -> (32, IMG/2, IMG/2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # -> (64, IMG/4, IMG/4)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # -> (128, IMG/8, IMG/8)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # -> (256, IMG/16, IMG/16)
        self.bn4 = nn.BatchNorm2d(256)

        final_spatial = img_size // (2**4)  # 128/(2^4)=8 for IMG_SIZE=128
        self.flatten_dim = 256 * final_spatial * final_spatial

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, img_size=IMG_SIZE):
        super().__init__()
        final_spatial = img_size // (2**4)  # should be 8
        self.fc = nn.Linear(latent_dim, 256 * final_spatial * final_spatial)

        self.deconv1 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1) # 8->16
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1) # 16->32
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 32->64
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)   # 64->128
        self.bn4 = nn.BatchNorm2d(32)

        self.final_conv = nn.Conv2d(32, 1, 3, padding=1)  # keep size same
        # Use sigmoid in forward for output in [0,1]

    def forward(self, z):
        final_spatial = IMG_SIZE // (2**4)
        x = self.fc(z)
        x = x.view(z.size(0), 256, final_spatial, final_spatial)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.sigmoid(self.final_conv(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_size=IMG_SIZE):
        super().__init__()
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ----------------------------
# Utilities: loss, early stopping, save/load
# ----------------------------
def loss_function(recon_x, x, mu, logvar, beta=BETA):
    # Reconstruction loss: binary cross entropy summed across pixels then averaged across batch
    # We keep sum over pixels to match TF implementation's reduce_sum axis=(1,2)
    # PyTorch BCE with reduction='sum' sums across all elements in a batch,
    # so we'll compute batch-wise and then divide by batch_size for logging if needed.
    batch_size = x.size(0)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')  # sum over batch
    # KL divergence per element then sum over latent dims and batch
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kld
    # return values as scalars (not normalized by batch) to match summed TF behaviour
    return total, recon_loss, kld

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_score = None
        self.num_bad = 0
        self.best_state = None
        self.best_epoch = -1

    def step(self, score, model, epoch):
        # score is the metric to *minimize* (e.g., val_loss)
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.num_bad = 0
            if self.restore_best:
                # deepcopy the model state dict
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                self.best_epoch = epoch
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.best_state.items()})
            return True
        return False

# ----------------------------
# Training loop
# ----------------------------
def train_vae(
    vae: VAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    device: torch.device = DEVICE,
    checkpoint_path: str = "best_vae.pt"
) -> Tuple[VAE, Dict[str, List[float]]]:
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    early_stopper = EarlyStopping(patience=10, restore_best=True)

    history = {"train_loss": [], "train_recon": [], "train_kld": [], "val_loss": [], "val_recon": [], "val_kld": []}

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        vae.train()
        running_total = 0.0
        running_recon = 0.0
        running_kld = 0.0
        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(imgs)
            total, recon_loss, kld = loss_function(recon, imgs, mu, logvar)
            total.backward()
            optimizer.step()

            running_total += total.item()
            running_recon += recon_loss.item()
            running_kld += kld.item()

        # Average by number of samples (note: losses above are summed over batch)
        n_train = len(train_loader.dataset)
        avg_total = running_total / n_train
        avg_recon = running_recon / n_train
        avg_kld = running_kld / n_train

        # Validation
        vae.eval()
        val_total = 0.0
        val_recon = 0.0
        val_kld_sum = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                recon, mu, logvar = vae(imgs)
                total_v, recon_v, kld_v = loss_function(recon, imgs, mu, logvar)
                val_total += total_v.item()
                val_recon += recon_v.item()
                val_kld_sum += kld_v.item()

        n_val = len(val_loader.dataset)
        avg_val_total = val_total / n_val
        avg_val_recon = val_recon / n_val
        avg_val_kld = val_kld_sum / n_val

        # Scheduler step uses validation loss (per-sample average)
        scheduler.step(avg_val_total)

        history["train_loss"].append(avg_total)
        history["train_recon"].append(avg_recon)
        history["train_kld"].append(avg_kld)
        history["val_loss"].append(avg_val_total)
        history["val_recon"].append(avg_val_recon)
        history["val_kld"].append(avg_val_kld)

        print(f"Epoch {epoch}/{epochs}  train_loss: {avg_total:.6f}  val_loss: {avg_val_total:.6f}  lr: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best
        if avg_val_total < best_val:
            best_val = avg_val_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_total
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path} (val_loss {best_val:.6f})")

        # Early stopping
        stop = early_stopper.step(avg_val_total, vae, epoch)
        if stop:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best model (epoch {early_stopper.best_epoch})")
            early_stopper.restore(vae)
            break

    # load best checkpoint in case early stopper didn't restore to file
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        vae.load_state_dict(ckpt["model_state_dict"])
    return vae, history

# ----------------------------
# Visualization utilities
# ----------------------------
def visualize_reconstructions(vae: VAE, dataset: Dataset, device=DEVICE, n_samples=8, out_path="vae_reconstructions.png"):
    vae.eval()
    imgs = []
    # sample random indices
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    for idx in indices:
        x, _ = dataset[idx]
        imgs.append(x)
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        recon, _, _ = vae(batch)
    batch_np = batch.cpu().numpy()
    recon_np = recon.cpu().numpy()

    fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))
    for i in range(n_samples):
        axes[0, i].imshow(batch_np[i, 0], cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original")
        axes[1, i].imshow(recon_np[i, 0], cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed")
    plt.suptitle("VAE Reconstruction Results")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

def visualize_latent_space(vae: VAE, dataset: Dataset, method="umap", n_samples=2000, device=DEVICE, out_prefix="latent_space"):
    vae.eval()
    # sample indices
    if len(dataset) > n_samples:
        indices = np.random.choice(len(dataset), n_samples, replace=False)
    else:
        indices = np.arange(len(dataset))
    # encode
    zs = []
    with torch.no_grad():
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            imgs = [dataset[j][0] for j in batch_idx]
            imgs = torch.stack(imgs).to(device)
            _, mu, _ = vae(imgs)
            zs.append(mu.cpu().numpy())
    zs = np.vstack(zs)
    print(f"Latent space shape: {zs.shape}")

    if method.lower() == "umap":
        print("Applying UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED)
        emb = reducer.fit_transform(zs)
    else:
        print("Applying t-SNE...")
        reducer = TSNE(n_components=2, perplexity=30, random_state=SEED)
        emb = reducer.fit_transform(zs)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=np.arange(len(emb)), cmap="viridis", s=5, alpha=0.6)
    plt.colorbar(sc, label="Sample Index")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"VAE Latent Space ({method.upper()})")
    out_path = f"{out_prefix}_{method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

def generate_latent_grid(vae: VAE, n_grid=10, out_path="latent_grid_sampling.png", device=DEVICE):
    vae.eval()
    # grid over first 2 latent dims
    grid_x = np.linspace(-3, 3, n_grid)
    grid_y = np.linspace(-3, 3, n_grid)
    figure = np.zeros((IMG_SIZE * n_grid, IMG_SIZE * n_grid))
    with torch.no_grad():
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z = np.zeros((1, LATENT_DIM), dtype=np.float32)
                z[0, 0] = xi
                z[0, 1] = yi
                z_t = torch.from_numpy(z).to(device)
                x_dec = vae.decoder(z_t)
                img = x_dec.cpu().numpy()[0, 0]
                figure[i * IMG_SIZE: (i + 1) * IMG_SIZE, j * IMG_SIZE: (j + 1) * IMG_SIZE] = img
    plt.figure(figsize=(12, 12))
    plt.imshow(figure, cmap="gray")
    plt.axis("off")
    plt.title("VAE Latent Space Grid Sampling")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_training_history(history: Dict[str, List[float]], out_path="training_history.png"):
    epochs = len(history["train_loss"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Training")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[1].plot(history["train_recon"], label="Training")
    axes[1].plot(history["val_recon"], label="Validation")
    axes[1].set_title("Reconstruction Loss")
    axes[1].legend()
    axes[2].plot(history["train_kld"], label="Training")
    axes[2].plot(history["val_kld"], label="Validation")
    axes[2].set_title("KL Loss")
    axes[2].legend()
    plt.suptitle("VAE Training History")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    print("="*50)
    print("Brain MRI VAE (PyTorch) Training Pipeline")
    print("="*50)
    train_ds = MRIDataset(TRAIN_DIR, img_size=IMG_SIZE, max_samples=5000)
    val_ds = MRIDataset(VAL_DIR, img_size=IMG_SIZE, max_samples=1000)
    test_ds = MRIDataset(TEST_DIR, img_size=IMG_SIZE, max_samples=1000)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    vae = VAE(latent_dim=LATENT_DIM, img_size=IMG_SIZE)

    # train
    vae, history = train_vae(vae, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, checkpoint_path="best_vae.pt")

    # save final
    torch.save({"model_state_dict": vae.state_dict()}, "final_vae.pt")

    # Plot history
    plot_training_history(history)

    # Evaluate on test set (compute avg loss)
    vae.eval()
    total = 0.0
    recon_sum = 0.0
    kld_sum = 0.0
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(DEVICE)
            recon, mu, logvar = vae(imgs)
            t, r, k = loss_function(recon, imgs, mu, logvar)
            total += t.item()
            recon_sum += r.item()
            kld_sum += k.item()
    n_test = len(test_loader.dataset)
    print(f"Test avg total: {total / n_test:.6f}, recon: {recon_sum / n_test:.6f}, kld: {kld_sum / n_test:.6f}")

    # Reconstructions
    visualize_reconstructions(vae, test_ds, device=DEVICE)

    # Latent space
    visualize_latent_space(vae, test_ds, method="umap", device=DEVICE)
    visualize_latent_space(vae, test_ds, method="tsne", device=DEVICE)

    # Latent grid
    generate_latent_grid(vae, n_grid=10, device=DEVICE)

    print("Saved artifacts:")
    print(" - best_vae.pt")
    print(" - final_vae.pt")
    print(" - vae_reconstructions.png")
    print(" - latent_space_umap.png")
    print(" - latent_space_tsne.png")
    print(" - latent_grid_sampling.png")
    print(" - training_history.png")

    return vae, history

if __name__ == "__main__":
    main()
