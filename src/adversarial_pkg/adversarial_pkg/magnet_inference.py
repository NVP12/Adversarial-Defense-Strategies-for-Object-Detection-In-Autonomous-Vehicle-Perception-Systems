import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, Normalize
from functools import lru_cache

# ---------- 1. model definition (must match training) ----------
class MagNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))


# ---------- 2. load-once checkpoint (cached) ----------
@lru_cache(maxsize=1)
def _load_model(path, dev):
    ckpt = torch.load(path, map_location=dev)
    net = MagNetDetector().to(dev)
    net.load_state_dict(ckpt["model_state"], strict=True)
    net.eval()
    net._input_size = ckpt["input_size"]
    net._norm_mean = ckpt["normalization"]["mean"]
    net._norm_std = ckpt["normalization"]["std"]
    return net


# ---------- 3. main detection entry point ----------
def magnet_detect(img,
                  weights_path="magnet_detector.pt",
                  device="cpu",
                  threshold=0.5) -> int:
    """
    Parameters
    ----------
    img : torch.Tensor
        Single image, **CHW** format.
        dtype uint8 0-255 or float32 0-1

    Returns
    -------
    int
        0 = clean
        1 = adversarial
    """

    model = _load_model(weights_path, device)

    # ---------- 4. preprocess identical to training ----------
    if img.dtype == torch.uint8:
        img = img.float().div_(255.0)

    tfm = Compose([
        Resize(model._input_size[1:]),
        Normalize(mean=model._norm_mean, std=model._norm_std),
    ])
    x = tfm(img).unsqueeze(0).to(device)  # shape [1, 3, H, W]

    # ---------- 5. predict ----------
    with torch.no_grad():
        prob_adv = model(x).item()

    return int(prob_adv > threshold)


##### THE WAY TO USE THE ABOVE FUNCTION ####


# img = img.float().div_(255.0)
#     tfm = Compose([
#         Resize(model._input_size[1:]),
#         Normalize(mean=model._norm_mean, std=model._norm_std),
#     ])
#     x = tfm(img).unsqueeze(0).to(device)   # shape [1,3,H,W]

#     # ---------- 4. predict ----------
#     with torch.no_grad():
#         prob_adv = torch.sigmoid(model(x)).item()
#     return int(prob_adv > threshold)
