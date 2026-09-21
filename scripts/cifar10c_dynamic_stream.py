# ============================================================
# CIFAR-10-C DYNAMIC STREAM — ResNet-18, multi-seed.
#
# Produces the results behind Table 2 of the paper (mean windowed ECE over a
# non-stationary, switching corruption stream at severity 5). Uses the
# CONFIRMED-FIXED sicl_confidence formula (style_invariance * content_invariance)
# — see auroc_signal_diagnostic.py and the supplementary material for the
# diagnostic that motivated this fusion.
# ============================================================
import os, copy, random, csv, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ---------------------- CONFIG (edit me) ----------------------
CIFAR10C_DIR    = "./data/CIFAR-10-C"
CHECKPOINT_PATH = "./checkpoints/resnet18_best_seed_42.pt"
CLEAN_VAL_ROOT  = "./data/cifar10_clean"
DYNAMIC_CSV     = "./results/dynamic_stream_resnet18_multiseed.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [0, 1, 2]
BATCH_SIZE = 128
NUM_CLASSES = 10
LR = 1e-3
N_STYLE_VARIANTS = 8
STYLE_PERTURB_LAYER = "layer1"
SICL_EPS = 1e-6

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
DYNAMIC_SEVERITY = 5
DYNAMIC_STAY_PROB = 0.985
DYNAMIC_TOTAL_BATCHES = 400
DYNAMIC_WINDOW = 20
DYNAMIC_MODES = ["tent", "tent_temp", "sicl", "cgt_isocal", "sicl_isocal"]
# ----------------------------------------------------------------

os.makedirs(os.path.dirname(DYNAMIC_CSV), exist_ok=True)

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
    model = ResNet18CIFAR(num_classes=NUM_CLASSES)
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

def sicl_confidence(model, perturber, x, pred, N=N_STYLE_VARIANTS):
    """Confirmed-fixed formula: style_invariance * content_invariance."""
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
    return style_invariance * content_invariance

class FeatureBank:
    def __init__(self, size=2048, k=20):
        self.size, self.k = size, k
        self.feats, self.correct = [], []
    def add(self, feats, correct):
        feats = F.normalize(feats, dim=1).detach().cpu()
        for f, c in zip(feats, correct):
            self.feats.append(f); self.correct.append(float(c))
            if len(self.feats) > self.size:
                self.feats.pop(0); self.correct.pop(0)
    def local_acc(self, feats):
        if len(self.feats) < self.k:
            return torch.full((feats.size(0),), 0.5)
        bank_f = torch.stack(self.feats)
        bank_c = torch.tensor(self.correct)
        q = F.normalize(feats, dim=1).detach().cpu()
        sim = q @ bank_f.T
        topk = sim.topk(self.k, dim=1).indices
        return bank_c[topk].mean(dim=1)

class OnlineIsotonicCalibrator:
    """Fit strictly causally: update() with the current batch happens AFTER
    calibrate() has already produced this batch's output (see the calling
    code below and the supplementary material's Algorithm 1)."""
    def __init__(self, window=2048):
        from sklearn.isotonic import IsotonicRegression
        self.window = window
        self.confs, self.pseudo_accs = [], []
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False
    def update(self, confs, pseudo_accs):
        self.confs.extend(confs.tolist())
        self.pseudo_accs.extend(pseudo_accs.tolist())
        if len(self.confs) > self.window:
            self.confs = self.confs[-self.window:]
            self.pseudo_accs = self.pseudo_accs[-self.window:]
    def refit(self):
        if len(self.confs) < 50:
            return
        self.iso.fit(self.confs, self.pseudo_accs)
        self._fitted = True
    def calibrate(self, confs):
        if not self._fitted:
            return confs
        out = self.iso.predict(confs.detach().cpu().numpy())
        return torch.from_numpy(out).float().to(confs.device)

def compute_ece(confidences, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(confidences)
    if n == 0:
        return float("nan")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0: continue
        acc_bin = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return ece * 100

def fit_temperature(model, val_loader):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_list.append(model(x))
            labels_list.append(y)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    temperature = nn.Parameter(torch.ones(1, device=DEVICE) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)
    def _step():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(_step)
    return temperature.item()

def build_clean_val_loader(size=2000, shuffle=False):
    import torchvision, torchvision.transforms as transforms
    mean = [0.4914, 0.4822, 0.4465]; std = [0.2470, 0.2435, 0.2616]
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    os.makedirs(CLEAN_VAL_ROOT, exist_ok=True)
    ds = torchvision.datasets.CIFAR10(root=CLEAN_VAL_ROOT, train=False, download=True, transform=tfm)
    idx = torch.randperm(len(ds))[:size]
    return DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=BATCH_SIZE, shuffle=shuffle)

def load_completed_keys(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                done.add((row["mode"], int(row["seed"]), int(row["batch_idx"])))
    return done

def build_dynamic_stream_indices(n_corruptions, n_batches, stay_prob, seed):
    rng = random.Random(seed)
    seq = [rng.randrange(n_corruptions)]
    for _ in range(n_batches - 1):
        if rng.random() < stay_prob:
            seq.append(seq[-1])
        else:
            choices = [c for c in range(n_corruptions) if c != seq[-1]]
            seq.append(rng.choice(choices))
    return seq

def run_dynamic_stream_one_seed(model_source, temperature, seed, all_data, done, writer, f_out):
    corruption_sequence = build_dynamic_stream_indices(
        len(CORRUPTIONS), DYNAMIC_TOTAL_BATCHES, DYNAMIC_STAY_PROB, seed)

    for mode in DYNAMIC_MODES:
        if (mode, seed, DYNAMIC_TOTAL_BATCHES - 1) in done:
            print(f"[dynamic seed={seed} mode={mode}] already complete — skipping")
            continue

        print(f"\n[dynamic seed={seed}] running mode={mode} ...")
        t0 = time.time()
        set_seed(seed)
        model = copy.deepcopy(model_source).to(DEVICE)
        feats_hook, handle = get_feat_extractor(model)
        params = configure_tent_bn(model)
        opt = torch.optim.Adam(params, lr=LR)

        needs_bank = mode in ("cgt_isocal", "sicl_isocal")
        needs_sicl = mode in ("sicl", "sicl_isocal")
        bank = FeatureBank() if needs_bank else None
        isocal = OnlineIsotonicCalibrator() if needs_bank else None
        perturber = SICLPerturber(model) if needs_sicl else None

        conf_window, correct_window = [], []

        for batch_idx, corr_idx in enumerate(corruption_sequence):
            x_all, y_all = all_data[corr_idx]
            idx = torch.randint(0, x_all.size(0), (BATCH_SIZE,),
                                 generator=torch.Generator().manual_seed(seed * 100000 + batch_idx))
            x, y = x_all[idx].to(DEVICE), y_all[idx].to(DEVICE)

            with torch.no_grad():
                model.eval()
                src_logits = model(x)
                src_feats = feats_hook["z"].clone()

            model.train()
            logits = model(x)
            loss = entropy_loss(logits)
            opt.zero_grad(); loss.backward(); opt.step()

            with torch.no_grad():
                model.eval()
                ada_logits = model(x)
                if mode == "tent_temp":
                    ada_logits = ada_logits / temperature
                ada_probs = F.softmax(ada_logits, 1)
                ada_conf, ada_pred = ada_probs.max(1)

            # NOTE ON CAUSALITY: for cgt_isocal/sicl_isocal, `isocal.calibrate()` is
            # always called BEFORE `isocal.update()`/`refit()` with this batch's own
            # pair — the map applied to this batch was fit only on previous batches.
            if mode == "sicl":
                conf = sicl_confidence(model, perturber, x, ada_pred)
                pred = ada_pred
            elif mode == "cgt_isocal":
                local_acc = bank.local_acc(src_feats).to(DEVICE)
                conf = isocal.calibrate(ada_conf)
                pseudo_correct = flip_agreement_pseudo_correct(model, x, ada_pred)
                bank.add(src_feats, pseudo_correct)
                isocal.update(ada_conf.detach().cpu().numpy(), local_acc.detach().cpu().numpy())
                isocal.refit()
                pred = ada_pred
            elif mode == "sicl_isocal":
                conf = isocal.calibrate(ada_conf)
                style_score = sicl_confidence(model, perturber, x, ada_pred)
                pseudo_correct = flip_agreement_pseudo_correct(model, x, ada_pred)
                bank.add(src_feats, pseudo_correct)
                isocal.update(ada_conf.detach().cpu().numpy(), style_score.detach().cpu().numpy())
                isocal.refit()
                pred = ada_pred
            else:  # tent, tent_temp
                conf, pred = ada_conf, ada_pred

            correct = (pred == y).float()
            conf_window.append(conf.detach().cpu().numpy())
            correct_window.append(correct.detach().cpu().numpy())
            if len(conf_window) > DYNAMIC_WINDOW:
                conf_window.pop(0); correct_window.pop(0)

            if (batch_idx + 1) % DYNAMIC_WINDOW == 0:
                c = np.concatenate(conf_window)
                r = np.concatenate(correct_window)
                window_acc = r.mean() * 100
                window_ece = compute_ece(c, r)
                writer.writerow({"mode": mode, "seed": seed, "batch_idx": batch_idx,
                                  "corruption": CORRUPTIONS[corr_idx],
                                  "window_acc": window_acc, "window_ece": window_ece})
                f_out.flush()
                print(f"[dynamic seed={seed} {mode:12s} batch {batch_idx+1:4d}/{DYNAMIC_TOTAL_BATCHES}] "
                      f"corr={CORRUPTIONS[corr_idx]:18s} acc={window_acc:5.2f}  ece={window_ece:5.2f}")

        handle.remove()
        if perturber is not None:
            perturber.remove()
        print(f"[dynamic seed={seed} mode={mode}] done in {time.time()-t0:.1f}s")

def summarize_multiseed(csv_path):
    from collections import defaultdict
    by_mode = defaultdict(list)
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            by_mode[row["mode"]].append(float(row["window_ece"]))
    print("\n" + "=" * 50)
    print(f"{'Mode':14s} {'Mean ECE':10s} {'Std ECE':10s} {'n windows'}")
    print("=" * 50)
    for mode in DYNAMIC_MODES:
        vals = np.array(by_mode.get(mode, []))
        if len(vals) == 0:
            continue
        print(f"{mode:14s} {vals.mean():8.2f}   {vals.std():8.2f}   {len(vals)}")

if __name__ == "__main__":
    print(f"[device] using {DEVICE}\n")
    model_source = load_model()
    model_source.eval()

    print("[temp scaling] fitting temperature on clean CIFAR-10 val split...")
    val_loader = build_clean_val_loader()
    temperature = fit_temperature(model_source, val_loader)
    print(f"[temp scaling] fitted T={temperature:.3f}\n")

    print(f"[dynamic] loading all {len(CORRUPTIONS)} corruptions at severity {DYNAMIC_SEVERITY}...")
    all_data = [load_cifar10c(c, DYNAMIC_SEVERITY) for c in CORRUPTIONS]

    done = load_completed_keys(DYNAMIC_CSV)
    write_header = not os.path.exists(DYNAMIC_CSV)
    f_out = open(DYNAMIC_CSV, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=["mode", "seed", "batch_idx", "corruption", "window_acc", "window_ece"])
    if write_header:
        writer.writeheader()

    try:
        for seed in SEEDS:
            run_dynamic_stream_one_seed(model_source, temperature, seed, all_data, done, writer, f_out)
    finally:
        f_out.close()
        summarize_multiseed(DYNAMIC_CSV)
