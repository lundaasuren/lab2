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
from torchvision import transforms

import umap
from sklearn.manifold import TSNE

# ----------------------------
# Configuration
# ----------------------------
IMG_SIZE = 256
LATENT_DIM = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
TARGET_BETA = 0.02         # final beta after annealing (try 0.01 - 0.1)
ANNEAL_EPOCHS = 50         # linear anneal over this many epochs
BETA_MIN = 0.0            # starting beta
BATCH_CLIP_NORM = 1.0

DATA_DIR = "/home/groups/comp3710/OASIS"
TRAIN_DIR = os.path.join(DATA_DIR, "keras_png_slices_train")
VAL_DIR = os.path.join(DATA_DIR, "keras_png_slices_validate")
TEST_DIR = os.path.join(DATA_DIR, "keras_png_slices_test")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 1
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
    def __init__(self, folder: str, max_samples: int = None):
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        # recursive search in case slices are in subfolders
        self.paths = sorted(folder_path.glob("**/*.png"))
        if len(self.paths) == 0:
            raise ValueError(f"No PNG files found in {folder}")
        if max_samples is not None and max_samples < len(self.paths):
            self.paths = random.sample(self.paths, max_samples)
        # Keep values in [0,1] for BCE loss
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),   # [0,1]
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("L")
            img = self.transform(img)  # (1,H,W) float in [0,1]
            return img, str(p)
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}")
            return torch.zeros((1, IMG_SIZE, IMG_SIZE)), str(p)

def create_data_loaders(batch_size=BATCH_SIZE):
    # sanity checks
    for name, path in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} directory not found: {path}")
        n = len(list(Path(path).glob("**/*.png")))
        print(f"[INFO] {name} set: {n} PNG images")
    train_ds = MRIDataset(TRAIN_DIR)
    val_ds = MRIDataset(VAL_DIR)
    test_ds = MRIDataset(TEST_DIR)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

# ----------------------------
# VAE model
# (Encoder / Decoder tuned for 256x256)
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim: int, img_size=IMG_SIZE):
        super().__init__()
        # Input (B,1,256,256)
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)    # -> (32,128,128)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # -> (64,64,64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # -> (128,32,32)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # -> (256,16,16)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1) # -> (512,8,8)
        self.bn5 = nn.BatchNorm2d(512)

        final_spatial = img_size // (2**5)  # 8 for 256
        self.flatten_dim = 512 * final_spatial * final_spatial
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, img_size=IMG_SIZE):
        super().__init__()
        final_spatial = img_size // (2**5)  # 8
        self.final_spatial = final_spatial
        self.fc = nn.Linear(latent_dim, 512 * final_spatial * final_spatial)
        # transpose convs that double spatial dims exactly
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 8->16
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16->32
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 32->64
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 64->128
        self.bn4 = nn.BatchNorm2d(32)
        # final doubling step to 256 - use kernel_size=4,stride=2,padding=1 (no output_padding)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)     # 128->256

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(z.size(0), 512, self.final_spatial, self.final_spatial)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x))  # ensure [0,1] output
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_size=IMG_SIZE):
        super().__init__()
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)
        self.latent_dim = latent_dim

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
# Loss (dynamic beta)
# ----------------------------
def loss_function(recon_x, x, mu, logvar, beta: float):
    # basic validation
    if recon_x.shape != x.shape:
        # crop to minimum spatial dims (last-resort safeguard)
        min_h = min(recon_x.shape[2], x.shape[2])
        min_w = min(recon_x.shape[3], x.shape[3])
        recon_x = recon_x[:, :, :min_h, :min_w]
        x = x[:, :, :min_h, :min_w]
    # clamp to avoid log(0)
    recon_x = torch.clamp(recon_x, 1e-7, 1.0 - 1e-7)
    x = torch.clamp(x, 0.0, 1.0)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')  # summed over batch
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kld
    return total, recon_loss, kld

# ----------------------------
# Utilities: denormalize and visualizations
# ----------------------------
def denormalize(tensor):
    # inputs are in [0,1], so identity - keep for clarity
    return tensor.clamp(0.0, 1.0)

def visualize_reconstructions(vae: VAE, dataset: Dataset, n_samples=8, out_path="vae_reconstructions.png", device=DEVICE):
    vae.eval()
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    imgs = []
    for i in indices:
        x, _ = dataset[i]
        imgs.append(x)
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        recon, _, _ = vae(batch)
    originals = denormalize(batch).cpu().numpy()
    reconstructions = denormalize(recon).cpu().numpy()
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*1.5, 4))
    for i in range(n_samples):
        axes[0, i].imshow(originals[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original")
        axes[1, i].imshow(reconstructions[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed")
    plt.suptitle("VAE Reconstructions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

def generate_latent_grid(vae: VAE, n_grid=12, out_path="latent_grid.png", device=DEVICE):
    vae.eval()
    # grid for first two latent dims
    grid_x = np.linspace(-3, 3, n_grid)
    grid_y = np.linspace(-3, 3, n_grid)
    figure = np.zeros((IMG_SIZE * n_grid, IMG_SIZE * n_grid))
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = np.zeros((1, vae.latent_dim), dtype=np.float32)
                z[0, 0] = xi
                z[0, 1] = yi
                z_t = torch.from_numpy(z).to(device)
                x_dec = vae.decoder(z_t)
                img = x_dec.cpu().numpy()[0, 0]
                figure[i * IMG_SIZE: (i + 1) * IMG_SIZE, j * IMG_SIZE: (j + 1) * IMG_SIZE] = img
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Latent Grid (dims 0 & 1)")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

def visualize_umap(vae: VAE, dataset: Dataset, n_samples=2000, method="umap", out_path="latent_umap.png", device=DEVICE):
    vae.eval()
    # sample indices
    if len(dataset) > n_samples:
        idxs = np.random.choice(len(dataset), n_samples, replace=False)
    else:
        idxs = np.arange(len(dataset))
    zs = []
    with torch.no_grad():
        for i in range(0, len(idxs), BATCH_SIZE):
            batch_idx = idxs[i:i+BATCH_SIZE]
            imgs = [dataset[j][0] for j in batch_idx]
            imgs = torch.stack(imgs).to(device)
            _, mu, _ = vae(imgs)
            zs.append(mu.cpu().numpy())
    zs = np.vstack(zs)
    print(f"[INFO] Latent means shape for UMAP: {zs.shape}")
    if method.lower() == "umap":
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED)
        emb = reducer.fit_transform(zs)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=SEED)
        emb = reducer.fit_transform(zs)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=np.arange(len(emb)), cmap="viridis", s=6, alpha=0.6)
    plt.colorbar(sc, label="Sample index")
    plt.title(f"Latent Projection ({method.upper()})")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

# ----------------------------
# Training function with KL annealing
# ----------------------------
def train_vae(vae: VAE, train_loader: DataLoader, val_loader: DataLoader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, checkpoint_path="best_vae.pt"):
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        vae.train()
        running_total = 0.0
        running_recon = 0.0
        running_kld = 0.0
        seen = 0
        # compute beta for annealing
        beta = TARGET_BETA * min(1.0, epoch / float(ANNEAL_EPOCHS))
        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(imgs)
            total, recon_loss, kld = loss_function(recon, imgs, mu, logvar, beta=beta)
            if torch.isnan(total) or torch.isinf(total):
                print(f"[WARN] NaN/Inf loss at epoch {epoch}, batch {batch_idx} -- skipping batch")
                continue
            total.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), BATCH_CLIP_NORM)
            optimizer.step()
            running_total += total.item()
            running_recon += recon_loss.item()
            running_kld += kld.item()
            seen += imgs.size(0)
        # compute averages per image
        avg_train_total = running_total / seen if seen>0 else float('inf')
        avg_train_recon = running_recon / seen if seen>0 else float('inf')
        avg_train_kld = running_kld / seen if seen>0 else float('inf')

        # validation
        vae.eval()
        val_total = 0.0
        val_recon = 0.0
        val_kld = 0.0
        seen_val = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                recon, mu, logvar = vae(imgs)
                # for validation, use full KL (beta does not matter for logging), but we keep beta for consistency
                total_v, recon_v, kld_v = loss_function(recon, imgs, mu, logvar, beta=beta)
                val_total += total_v.item()
                val_recon += recon_v.item()
                val_kld += kld_v.item()
                seen_val += imgs.size(0)
        avg_val_total = val_total / seen_val if seen_val>0 else float('inf')
        avg_val_recon = val_recon / seen_val if seen_val>0 else float('inf')
        avg_val_kld = val_kld / seen_val if seen_val>0 else float('inf')

        scheduler.step(avg_val_total)
        history["train_loss"].append(avg_train_total)
        history["val_loss"].append(avg_val_total)

        print(f"Epoch {epoch}/{epochs}  train_loss: {avg_train_total:.4f}  val_loss: {avg_val_total:.4f}  beta:{beta:.4f}  lr:{optimizer.param_groups[0]['lr']:.2e}")

        # save best
        if avg_val_total < best_val:
            best_val = avg_val_total
            torch.save({"epoch": epoch, "model_state_dict": vae.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": best_val}, checkpoint_path)
            print(f"[INFO] Saved best model to {checkpoint_path} (val_loss {best_val:.4f})")

    # final load best
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        vae.load_state_dict(ckpt["model_state_dict"])
    return vae, history

# ----------------------------
# Main script
# ----------------------------
def main():
    print("="*60)
    print("Brain MRI VAE (PyTorch) - OASIS")
    print("="*60)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_data_loaders()
    print("[INFO] Creating VAE model...")
    vae = VAE(latent_dim=LATENT_DIM, img_size=IMG_SIZE).to(DEVICE)
    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"[INFO] VAE parameters: {total_params:,}")
    # quick forward pass check
    sample_batch, _ = next(iter(train_loader))
    print(f"[INFO] Sample batch shape: {sample_batch.shape}, range: [{sample_batch.min():.3f},{sample_batch.max():.3f}]")
    with torch.no_grad():
        recon, mu, logvar = vae(sample_batch.to(DEVICE)[:4])
        print(f"[INFO] Forward shapes - recon: {recon.shape}, mu: {mu.shape}, logvar: {logvar.shape}")
    # Train
    vae, history = train_vae(vae, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, checkpoint_path="best_vae.pt")
    torch.save({"model_state_dict": vae.state_dict()}, "final_vae.pt")
    print("[INFO] Saved final model to final_vae.pt")
    # Visualizations
    visualize_reconstructions(vae, test_ds, n_samples=8, out_path="vae_reconstructions.png", device=DEVICE)
    generate_latent_grid(vae, n_grid=10, out_path="latent_grid.png", device=DEVICE)
    visualize_umap(vae, test_ds, n_samples=2000, method="umap", out_path="latent_umap.png", device=DEVICE)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
