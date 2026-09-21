# ============================================================
# AUROC SIGNAL VALIDATION — ResNet-18, CIFAR-10-C.
#
# Produces the diagnostic behind Table S4 of the supplementary material:
# AUROC of each label-free signal (raw confidence, flip agreement, style
# invariance, content invariance, and their fusion) against TRUE correctness.
# Labels are used ONLY here, for offline evaluation of signal quality — never
# fed into the adaptation loop, the feature bank, or the isotonic calibrator
# in any other script in this repository.
#
# Set FUSION_FORMULA below to "correct" (style * content, used in the paper)
# or "buggy" (the original inverted (1-content)*style formula) to reproduce
# both rows of Table S4.
# ============================================================
import os, copy, random, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ---------------------- CONFIG ----------------------
CIFAR10C_DIR    = "./data/CIFAR-10-C"
CHECKPOINT_PATH = "./checkpoints/resnet18_best_seed_42.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FUSION_FORMULA = "correct"   # "correct" or "buggy" — see module docstring

SEED = 0
SEVERITIES = [1, 2, 3, 4, 5]
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
BATCH_SIZE = 128
LR = 1e-3
N_STYLE_VARIANTS = 8
STYLE_PERTURB_LAYER = "layer1"
SICL_EPS = 1e-6

AUROC_CSV = f"./results/auroc_signal_validation_{FUSION_FORMULA}.csv"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_cifar10c(corruption, severity):
    data = np.load(os.path.join(CIFAR10C_DIR, f"{corruption}.npy"))
    labels = np.load(os.path.join(CIFAR10C_DIR, "labels.npy"))
    lo, hi = (severity - 1) * 10000, severity * 10000
    x, y = data[lo:hi], labels[lo:hi]
    x = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = torch.from_numpy(y).long()
    return x, y

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion))
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        return self.fc(out)

def load_model():
    model = ResNet18CIFAR(num_classes=10)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    return model.to(DEVICE)

def get_feat_extractor(model):
    feats = {}
    def hook(_, __, output):
        feats["z"] = output.flatten(1)
    handle = model.avgpool.register_forward_hook(hook)
    return feats, handle

def configure_tent_bn(model):
    model.train()
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.requires_grad_(True)
            params += [m.weight, m.bias]
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    return params

def entropy_loss(logits):
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-8)).sum(1).mean()

def flip_agreement_pseudo_correct(model, x, pred):
    with torch.no_grad():
        flipped_logits = model(x.flip(dims=[3]))
        flipped_pred = flipped_logits.argmax(1)
    return (flipped_pred == pred).float()

class SICLPerturber:
    def __init__(self, model, layer_name=STYLE_PERTURB_LAYER):
        self.mode = None
        target = dict(model.named_modules())[layer_name]
        self.handle = target.register_forward_hook(self._hook)
    def _hook(self, module, inp, output):
        if self.mode is None:
            return output
        mu = output.mean(dim=[2, 3], keepdim=True)
        sigma = output.std(dim=[2, 3], keepdim=True)
        if self.mode == "style":
            delta = mu.std(dim=0, keepdim=True) + SICL_EPS
            eps_mu = torch.randn_like(mu)
            eps_sigma = torch.randn_like(sigma)
            mu_perturb = mu + delta * eps_mu
            sigma_perturb = sigma + delta * eps_sigma
            return sigma_perturb * (output - mu) / (sigma + SICL_EPS) + mu_perturb
        elif self.mode == "content":
            f_white = (output - mu) / (sigma + SICL_EPS)
            sigma_white = f_white.std(dim=[2, 3], keepdim=True)
            noise = torch.randn_like(f_white) * sigma_white
            return sigma * (f_white + noise) + mu
        return output
    def set_mode(self, mode):
        self.mode = mode
    def remove(self):
        self.handle.remove()

def sicl_confidence_full(model, perturber, x, pred, N=N_STYLE_VARIANTS):
    style_agree = torch.zeros_like(pred, dtype=torch.float32)
    content_agree = torch.zeros_like(pred, dtype=torch.float32)
    with torch.no_grad():
        perturber.set_mode("style")
        for _ in range(N):
            style_agree += (model(x).argmax(1) == pred).float()
        perturber.set_mode("content")
        for _ in range(N):
            content_agree += (model(x).argmax(1) == pred).float()
        perturber.set_mode(None)
    style_invariance = style_agree / N
    content_invariance = content_agree / N
    if FUSION_FORMULA == "correct":
        relaxed = style_invariance * content_invariance
    elif FUSION_FORMULA == "buggy":
        relaxed = (1 - content_invariance) * style_invariance
    else:
        raise ValueError(f"Unknown FUSION_FORMULA: {FUSION_FORMULA}")
    return relaxed, style_invariance, content_invariance

def run_auroc_validation(model_source):
    from sklearn.metrics import roc_auc_score

    os.makedirs(os.path.dirname(AUROC_CSV), exist_ok=True)
    fieldnames = ["severity", "corruption", "signal", "auroc", "n_samples"]
    write_header = not os.path.exists(AUROC_CSV)
    f_out = open(AUROC_CSV, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for sev in SEVERITIES:
        for corr in CORRUPTIONS:
            x_all, y_all = load_cifar10c(corr, sev)
            ds = TensorDataset(x_all, y_all)
            g = torch.Generator(); g.manual_seed(SEED)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)

            set_seed(SEED)
            model = copy.deepcopy(model_source).to(DEVICE)
            feats_hook, handle = get_feat_extractor(model)
            perturber = SICLPerturber(model)
            params = configure_tent_bn(model)
            opt = torch.optim.Adam(params, lr=LR)

            raw_conf_all, flip_sig_all, sicl_sig_all, style_all, content_all, true_correct_all = [], [], [], [], [], []

            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                model.train()
                logits = model(x)
                loss = entropy_loss(logits)
                opt.zero_grad(); loss.backward(); opt.step()

                with torch.no_grad():
                    model.eval()
                    ada_logits = model(x)
                    ada_probs = F.softmax(ada_logits, 1)
                    ada_conf, ada_pred = ada_probs.max(1)

                flip_sig = flip_agreement_pseudo_correct(model, x, ada_pred)
                sicl_relaxed, style_inv, content_inv = sicl_confidence_full(model, perturber, x, ada_pred)
                true_correct = (ada_pred == y).float()   # evaluation-only use of labels

                raw_conf_all.append(ada_conf.detach().cpu().numpy())
                flip_sig_all.append(flip_sig.detach().cpu().numpy())
                sicl_sig_all.append(sicl_relaxed.detach().cpu().numpy())
                style_all.append(style_inv.detach().cpu().numpy())
                content_all.append(content_inv.detach().cpu().numpy())
                true_correct_all.append(true_correct.detach().cpu().numpy())

            handle.remove(); perturber.remove()

            raw_conf_all = np.concatenate(raw_conf_all)
            flip_sig_all = np.concatenate(flip_sig_all)
            sicl_sig_all = np.concatenate(sicl_sig_all)
            style_all = np.concatenate(style_all)
            content_all = np.concatenate(content_all)
            true_correct_all = np.concatenate(true_correct_all)

            for signal_name, signal_vals in [("raw_confidence", raw_conf_all),
                                              ("flip_agreement", flip_sig_all),
                                              ("sicl_relaxed_score", sicl_sig_all),
                                              ("sicl_style_only", style_all),
                                              ("sicl_content_invariance", content_all)]:
                auroc = roc_auc_score(true_correct_all, signal_vals) if len(np.unique(true_correct_all)) > 1 else float("nan")
                writer.writerow({"severity": sev, "corruption": corr, "signal": signal_name,
                                  "auroc": auroc, "n_samples": len(true_correct_all)})
            f_out.flush()
            print(f"[AUROC sev={sev} {corr:18s}] "
                  f"raw={roc_auc_score(true_correct_all, raw_conf_all):.3f}  "
                  f"relaxed={roc_auc_score(true_correct_all, sicl_sig_all):.3f}  "
                  f"style={roc_auc_score(true_correct_all, style_all):.3f}  "
                  f"content={roc_auc_score(true_correct_all, content_all):.3f}")

    f_out.close()
    print(f"\n[AUROC] done — results in {AUROC_CSV}")

if __name__ == "__main__":
    print(f"[device] using {DEVICE}, FUSION_FORMULA={FUSION_FORMULA}\n")
    model_source = load_model()
    model_source.eval()
    run_auroc_validation(model_source)
