# ============================================================
# CGT-TTA / SICL-ISOCAL vs TENT vs EATA vs SAR — CIFAR-10-C,
# all severities/corruptions, multi-seed. Local / VS Code version.
#
# Adapted from the Kaggle sweep script:
#   - /kaggle/... paths replaced with local relative paths (edit CONFIG below)
#   - DEVICE now falls back cuda -> mps -> cpu, with a runtime warning on cpu
#   - EATA and SAR added as new label-free baselines (see notes below)
#
# LABEL-FREE CORRECTNESS FIX (carried over): the feature bank's pseudo-correctness
# signal uses horizontal-flip test-time-augmentation agreement, NOT ground-truth
# labels y. Ground truth y is only ever used for final accuracy/ECE evaluation,
# never fed into any gate, feature bank, calibrator, or adaptation loss.
#
# EATA (Niu et al., ICML 2022, "Efficient Test-Time Model Adaptation without
# Forgetting"): implements (1) entropy-based sample filtering — only backprop
# samples below an entropy margin E0 — and (2) a diversity weight that down-weights
# samples whose softmax output is too similar to a moving average of recent outputs
# (redundant samples contribute little new adaptation signal). The paper's third
# component, Fisher-regularized anti-forgetting, requires labeled SOURCE-domain data
# (not test-domain labels — this is the standard, legitimate TTA assumption that you
# have access to the clean training distribution you trained the source model on).
# SIMPLIFICATION, flagged honestly: Fisher information here is estimated from only
# EATA_FISHER_BATCHES batches of the clean CIFAR-10 val split (fast, single-cell
# friendly) rather than the full source train set the paper uses — treat this
# baseline as a reasonable first-pass reproduction, not a tuned one.
#
# SAR (Niu et al., ICLR 2023, "Towards Stable Test-Time Adaptation in Dynamic
# Wild World"): implements (1) the same entropy-margin filtering as EATA, (2) a
# Sharpness-Aware Minimization (SAM) style two-step update that seeks flat minima
# (more robust under a noisy, label-free loss), and (3) a model-reset mechanism —
# if the exponential moving average of batch entropy exceeds a threshold (sign of
# adaptation collapse), the model's BN-affine parameters are reset to their
# adaptation-start snapshot. SIMPLIFICATION: uses a single SAM ascent/descent step
# per batch (paper default), no additional stability tricks beyond the reset.
# ============================================================
import os, copy, random, csv, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

# ---------------------- CONFIG (edit me) ----------------------
CIFAR10C_DIR    = "./data/CIFAR-10-C"                          # dir with <corruption>.npy + labels.npy
CHECKPOINT_PATH = "./checkpoints/resnet18_best_seed_42.pt"     # your trained CIFAR-10 checkpoint
CLEAN_VAL_ROOT  = "./data/cifar10_clean"                       # torchvision downloads here if missing

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE == "cpu":
    print("[WARNING] No GPU detected — running on CPU. This sweep is 5 severities x 15 corruptions x "
          "3 seeds x 8 modes = 1800 conditions; sicl/sicl_isocal/eata/sar all add extra forward/backward "
          "cost per batch. Consider trimming SEVERITIES/CORRUPTIONS/SEEDS for a first smoke-test pass.")

SEVERITIES   = [1, 2, 3, 4, 5]
SEEDS        = [0, 1, 2]
CORRUPTIONS  = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
BATCH_SIZE   = 128
N_BINS       = 15
TAU_HARD     = 0.4
TEMP_SOFT    = 0.15
K_NEIGHBORS  = 20
BANK_SIZE    = 2048
ISOCAL_WINDOW = 2048
ISOCAL_REFIT_EVERY = 1
LR           = 1e-3
WARMSTART_BATCHES = 32
RESULTS_CSV  = "./results/cgt_tta_results_cifar10.csv"

RESUME_FROM_INPUT = None   # set to a prior CSV path to resume from it

CLEAN_VAL_SIZE = 2000

# ---------------------- EATA config ----------------------
EATA_E0_COEF     = 0.4     # entropy margin E0 = coef * ln(num_classes), paper default 0.4
EATA_D_MARGIN    = 0.05    # cosine-similarity margin for the diversity weight, paper default 0.05
EATA_FISHER_BATCHES = 10   # batches of clean source data used to estimate Fisher info (see note above)
EATA_FISHER_ALPHA   = 2000.0  # regularization strength on the Fisher anti-forgetting term (paper ~2000)

# ---------------------- SAR config ----------------------
SAR_E0_COEF      = 0.4     # same entropy-margin filtering as EATA
SAR_RHO          = 0.05    # SAM neighborhood radius, paper default 0.05
SAR_RESET_EMA_DECAY   = 0.9   # decay for the exponential moving average of batch entropy
SAR_RESET_THRESHOLD_COEF = 0.2  # reset if ema entropy < coef * ln(num_classes) (near-collapse to a
                                  # trivial low-entropy/degenerate prediction), paper-inspired heuristic
# ----------------------------------------------------------------

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

def load_completed_keys(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                done.add((row["mode"], int(row["severity"]), row["corruption"], int(row["seed"])))
    return done

def append_result_row(csv_path, mode, sev, corr, seed, acc, ece):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["mode", "severity", "corruption", "seed", "acc", "ece"])
        w.writerow([mode, sev, corr, seed, acc, ece])

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------------------- Data ----------------------
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
    return TensorDataset(x, y)

# ---------------------- Model / feature hook ----------------------
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

NUM_CLASSES = 10

def load_model():
    model = ResNet18CIFAR(num_classes=NUM_CLASSES)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and not any(k.startswith("conv1") or k.startswith("layer") for k in state):
        state = state["model"]
    cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[load_model] missing keys: {missing}")
        print(f"[load_model] unexpected keys: {unexpected}")
    return model.to(DEVICE)

def get_feat_extractor(model):
    feats = {}
    def hook(_, __, output):
        feats["z"] = output.flatten(1)
    handle = model.avgpool.register_forward_hook(hook)
    return feats, handle

def configure_tent_bn(model):
    """TENT / EATA / SAR all use this: BN layers -> train mode, batch stats, only affine params trainable."""
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

def per_sample_entropy(logits):
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-8)).sum(1)   # [B], per-sample, not reduced

def entropy_loss(logits):
    return per_sample_entropy(logits).mean()

def flip_agreement_pseudo_correct(model, x, pred):
    with torch.no_grad():
        flipped_logits = model(x.flip(dims=[3]))
        flipped_pred = flipped_logits.argmax(1)
    return (flipped_pred == pred).float()

# ---------------------- SICL (unchanged, confirmed-fixed formula) ----------------------
STYLE_PERTURB_LAYER = "layer1"
N_STYLE_VARIANTS = 8
SICL_EPS = 1e-6

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
    """CONFIRMED-FIXED formula: style_invariance * content_invariance."""
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


# ---------------------- Feature bank / CGS (unchanged) ----------------------
class FeatureBank:
    def __init__(self, size=BANK_SIZE, k=K_NEIGHBORS):
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
    def __init__(self, window=ISOCAL_WINDOW):
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

def compute_ece(confidences, correct, n_bins=N_BINS):
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

# ---------------------- Temperature scaling (unchanged) ----------------------
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
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)
    def _step():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(_step)
    T = temperature.item()
    with torch.no_grad():
        pre_conf, pre_pred = F.softmax(logits, 1).max(1)
        pre_ece = compute_ece(pre_conf.cpu().numpy(), (pre_pred == labels).float().cpu().numpy())
        post_conf, post_pred = F.softmax(logits / T, 1).max(1)
        post_ece = compute_ece(post_conf.cpu().numpy(), (post_pred == labels).float().cpu().numpy())
    print(f"[temp scaling] fitted T={T:.3f}  |  clean-val ECE before={pre_ece:.2f}  after={post_ece:.2f}")
    return T

def build_clean_val_loader(size=CLEAN_VAL_SIZE, shuffle=False):
    import torchvision, torchvision.transforms as transforms
    mean = [0.4914, 0.4822, 0.4465]; std = [0.2470, 0.2435, 0.2616]
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    os.makedirs(CLEAN_VAL_ROOT, exist_ok=True)
    try:
        ds = torchvision.datasets.CIFAR10(root=CLEAN_VAL_ROOT, train=False, download=True, transform=tfm)
    except Exception as e:
        raise RuntimeError(
            "Could not download clean CIFAR-10 test split (no internet access?). "
            "Point CLEAN_VAL_ROOT at a local copy if you already have one."
        ) from e
    idx = torch.randperm(len(ds))[:size]
    subset = torch.utils.data.Subset(ds, idx.tolist())
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=shuffle)

# ============================================================
# EATA — entropy filtering + diversity weighting + Fisher anti-forgetting
# ============================================================
def eata_entropy_margin():
    return EATA_E0_COEF * float(np.log(NUM_CLASSES))

def compute_fisher(model_source, n_batches=EATA_FISHER_BATCHES):
    """
    Diagonal Fisher-information estimate on clean SOURCE-domain data (not test-domain
    labels — standard TTA assumption of source-data access). Returns:
      fisher:  dict param_name -> Fisher diagonal (same shape as param)
      anchor:  dict param_name -> a detached copy of the source model's own parameter
               values, used as the regularization target (pull adapted params back
               toward their source values, weighted by how important each was for the
               source task).
    SIMPLIFICATION: estimated from EATA_FISHER_BATCHES batches only (fast), not the
    full source train set as in the original paper.
    """
    model = copy.deepcopy(model_source).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    loader = build_clean_val_loader(size=BATCH_SIZE * n_batches, shuffle=True)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    anchor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    n_seen = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)   # labeled SOURCE loss — legitimate, not test-domain labels
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2 * x.size(0)
        n_seen += x.size(0)
        if n_seen >= BATCH_SIZE * n_batches:
            break

    for n in fisher:
        fisher[n] /= max(n_seen, 1)
    return fisher, anchor

def eata_fisher_penalty(model, fisher, anchor):
    """sum_p fisher[p] * (p - anchor[p])^2, restricted to the BN-affine params actually being adapted."""
    penalty = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and n in fisher:
            penalty = penalty + (fisher[n] * (p - anchor[n]) ** 2).sum()
    return penalty

class EATAState:
    """Tracks the moving average of softmax outputs used for the diversity weight."""
    def __init__(self):
        self.ema_prob = None
    def diversity_weight(self, probs):
        # probs: [B, C]. Weight down samples whose output is too similar to the
        # running average (redundant, adds little new adaptation signal).
        if self.ema_prob is None:
            self.ema_prob = probs.mean(0, keepdim=True).detach()
            return torch.ones(probs.size(0), device=probs.device)
        cos_sim = F.cosine_similarity(probs, self.ema_prob.expand_as(probs), dim=1)
        weight = (cos_sim < (1 - EATA_D_MARGIN)).float()   # 1 if sufficiently different, else 0
        self.ema_prob = 0.9 * self.ema_prob + 0.1 * probs.mean(0, keepdim=True).detach()
        return weight

def run_eata_step(model, opt, x, fisher, anchor, eata_state):
    model.train()
    logits = model(x)
    ent = per_sample_entropy(logits)
    e0 = eata_entropy_margin()
    reliable_mask = (ent < e0).float()   # entropy-margin filter, label-free

    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
    div_weight = eata_state.diversity_weight(probs)

    sample_weight = reliable_mask * div_weight
    if sample_weight.sum() > 0:
        weighted_entropy = (ent * sample_weight).sum() / sample_weight.sum()
    else:
        weighted_entropy = ent.mean() * 0.0   # nothing reliable this batch, skip adaptation contribution

    fisher_term = eata_fisher_penalty(model, fisher, anchor) if fisher is not None else 0.0
    loss = weighted_entropy + EATA_FISHER_ALPHA * fisher_term

    opt.zero_grad()
    if sample_weight.sum() > 0 or (fisher is not None):
        loss.backward()
        opt.step()

    with torch.no_grad():
        model.eval()
        final_logits = model(x)
        final_probs = F.softmax(final_logits, 1)
        conf, pred = final_probs.max(1)
    return conf, pred


# ============================================================
# SAR — entropy filtering + Sharpness-Aware Minimization + model reset
# ============================================================
def sar_entropy_margin():
    return SAR_E0_COEF * float(np.log(NUM_CLASSES))

class SARState:
    def __init__(self, model, params):
        self.ema_entropy = None
        self.reset_threshold = SAR_RESET_THRESHOLD_COEF * float(np.log(NUM_CLASSES))
        # snapshot of BN-affine params right after adaptation begins, used to reset on collapse
        self.snapshot = [p.detach().clone() for p in params]
        self.params = params

    def maybe_reset(self):
        if self.ema_entropy is not None and self.ema_entropy < self.reset_threshold:
            with torch.no_grad():
                for p, snap in zip(self.params, self.snapshot):
                    p.copy_(snap)
            self.ema_entropy = None   # clear so we don't reset every subsequent batch
            return True
        return False

    def update(self, batch_mean_entropy):
        if self.ema_entropy is None:
            self.ema_entropy = batch_mean_entropy
        else:
            self.ema_entropy = SAR_RESET_EMA_DECAY * self.ema_entropy + (1 - SAR_RESET_EMA_DECAY) * batch_mean_entropy

def run_sar_step(model, params, opt, x, sar_state):
    e0 = sar_entropy_margin()

    # --- SAM ascent step: find the local worst-case direction ---
    model.train()
    logits = model(x)
    ent = per_sample_entropy(logits)
    reliable_mask = (ent < e0)
    if reliable_mask.sum() == 0:
        # nothing reliable in this batch: skip adaptation, still check reset, report source-ish output
        with torch.no_grad():
            model.eval()
            final_logits = model(x)
            conf, pred = F.softmax(final_logits, 1).max(1)
        sar_state.update(ent.mean().item())
        sar_state.maybe_reset()
        return conf, pred

    loss1 = ent[reliable_mask].mean()
    opt.zero_grad()
    loss1.backward()

    # ascent: perturb params by rho * grad / ||grad|| (SAM's epsilon-hat)
    grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])) + 1e-12
    e_ws = []
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                e_ws.append(None); continue
            e_w = p.grad * (SAR_RHO / grad_norm)
            p.add_(e_w)
            e_ws.append(e_w)

    # --- descent step: compute loss at the perturbed point, step, then undo perturbation ---
    logits2 = model(x)
    ent2 = per_sample_entropy(logits2)
    reliable_mask2 = (ent2 < e0)
    if reliable_mask2.sum() == 0:
        reliable_mask2 = reliable_mask   # fall back to first-pass filter if none pass at perturbed point
    loss2 = ent2[reliable_mask2].mean()

    opt.zero_grad()
    loss2.backward()
    with torch.no_grad():
        for p, e_w in zip(params, e_ws):
            if e_w is not None:
                p.sub_(e_w)   # undo the ascent perturbation before applying the real step
    opt.step()

    with torch.no_grad():
        model.eval()
        final_logits = model(x)
        conf, pred = F.softmax(final_logits, 1).max(1)

    sar_state.update(ent.mean().item())
    sar_state.maybe_reset()
    return conf, pred


# ---------------------- One evaluation pass over a corruption/severity ----------------------
def run_condition(model_source, loader, mode, seed, temperature=None, fisher=None, fisher_anchor=None):
    """mode in {'source','tent','tent_temp','sicl','sicl_isocal','cgt_hard','cgt_soft','cgt_isocal','eata','sar'}"""
    set_seed(seed)
    model = copy.deepcopy(model_source).to(DEVICE)
    feats_hook, handle = get_feat_extractor(model)

    if mode in ("tent", "tent_temp", "sicl", "sicl_isocal", "cgt_hard", "cgt_soft", "cgt_isocal", "eata", "sar"):
        params = configure_tent_bn(model)
        opt = torch.optim.Adam(params, lr=LR)
    else:
        model.eval()
        opt = None

    bank = FeatureBank() if mode in ("cgt_hard", "cgt_soft", "cgt_isocal") else None
    isocal = OnlineIsotonicCalibrator() if mode in ("cgt_isocal", "sicl_isocal") else None
    style_perturber = SICLPerturber(model) if mode in ("sicl", "sicl_isocal") else None
    eata_state = EATAState() if mode == "eata" else None
    sar_state = SARState(model, params) if mode == "sar" else None

    if bank is not None:
        model.eval()
        with torch.no_grad():
            wcount = 0
            for x, y in loader:
                x = x.to(DEVICE)
                logits = model(x)
                pred = logits.argmax(1)
                pseudo_correct = flip_agreement_pseudo_correct(model, x, pred)
                bank.add(feats_hook["z"], pseudo_correct)
                wcount += 1
                if wcount >= WARMSTART_BATCHES:
                    break
        configure_tent_bn(model)

    all_conf, all_correct = [], []

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        if mode == "source":
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, 1)
                conf, pred = probs.max(1)
            all_conf.append(conf.cpu().numpy())
            all_correct.append((pred == y).float().cpu().numpy())
            continue

        with torch.no_grad():
            model.eval()
            src_logits = model(x)
            src_probs = F.softmax(src_logits, 1)
            src_conf, src_pred = src_probs.max(1)
            src_feats = feats_hook["z"].clone()

        if mode == "eata":
            conf, pred = run_eata_step(model, opt, x, fisher, fisher_anchor, eata_state)

        elif mode == "sar":
            conf, pred = run_sar_step(model, params, opt, x, sar_state)

        elif mode in ("tent", "tent_temp", "sicl", "sicl_isocal"):
            model.train()
            logits = model(x)
            loss = entropy_loss(logits)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                model.eval()
                ada_logits = model(x)
                if mode == "tent_temp":
                    assert temperature is not None, "tent_temp requires a fitted temperature"
                    ada_logits = ada_logits / temperature
                probs = F.softmax(ada_logits, 1)
                conf, pred = probs.max(1)
                if mode == "sicl":
                    conf = sicl_confidence(model, style_perturber, x, pred)
                elif mode == "sicl_isocal":
                    style_score = sicl_confidence(model, style_perturber, x, pred)
                    isocal.update(conf.detach().cpu().numpy(), style_score.detach().cpu().numpy())
                    if batch_idx % ISOCAL_REFIT_EVERY == 0:
                        isocal.refit()
                    conf = isocal.calibrate(conf)

        elif mode in ("cgt_hard", "cgt_soft", "cgt_isocal"):
            local_acc = bank.local_acc(src_feats).to(DEVICE)
            model.train()
            logits = model(x)
            loss = entropy_loss(logits)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                model.eval()
                ada_logits = model(x)
                ada_probs = F.softmax(ada_logits, 1)
                ada_conf, ada_pred = ada_probs.max(1)

            if mode == "cgt_hard":
                cgs = src_conf - local_acc
                gate = (cgs > TAU_HARD).float()
                pred = torch.where(gate.bool(), ada_pred, src_pred)
                conf = torch.where(gate.bool(), ada_conf, src_conf)
            elif mode == "cgt_soft":
                cgs = src_conf - local_acc
                gate = torch.sigmoid(cgs / TEMP_SOFT).unsqueeze(1)
                mixed_probs = (1 - gate) * src_probs + gate * ada_probs
                conf, pred = mixed_probs.max(1)
            else:
                pred = ada_pred
                isocal.update(ada_conf.detach().cpu().numpy(), local_acc.detach().cpu().numpy())
                if batch_idx % ISOCAL_REFIT_EVERY == 0:
                    isocal.refit()
                conf = isocal.calibrate(ada_conf)

            pseudo_correct = flip_agreement_pseudo_correct(model, x, pred)
            bank.add(src_feats, pseudo_correct)

        all_conf.append(conf.detach().cpu().numpy())
        all_correct.append((pred == y).float().detach().cpu().numpy())

    handle.remove()
    if style_perturber is not None:
        style_perturber.remove()
    conf = np.concatenate(all_conf)
    correct = np.concatenate(all_correct)
    acc = correct.mean() * 100
    ece = compute_ece(conf, correct)
    return acc, ece

def summarize(modes):
    results = defaultdict(list)
    by_key = defaultdict(dict)
    if not os.path.exists(RESULTS_CSV):
        print(f"[summarize] {RESULTS_CSV} does not exist yet — nothing to summarize.")
        return by_key
    with open(RESULTS_CSV, "r") as f:
        for row in csv.DictReader(f):
            mode, sev = row["mode"], int(row["severity"])
            corr, seed = row["corruption"], int(row["seed"])
            acc, ece = float(row["acc"]), float(row["ece"])
            results[(mode, sev)].append((acc, ece))
            by_key[(mode, sev)][(corr, seed)] = ece

    expected_per_cell = len(CORRUPTIONS) * len(SEEDS)
    print("\n" + "=" * 70)
    print(f"{'Mode':10s} {'Sev':4s} {'Acc mean±std':18s} {'ECE mean±std':18s} {'n (of ' + str(expected_per_cell) + ')'}")
    print("=" * 70)
    for sev in SEVERITIES:
        for mode in modes:
            arr = np.array(results.get((mode, sev), []))
            if len(arr) == 0:
                print(f"{mode:10s} {sev:<4d}  (no results logged yet)")
                continue
            acc_m, acc_s = arr[:, 0].mean(), arr[:, 0].std()
            ece_m, ece_s = arr[:, 1].mean(), arr[:, 1].std()
            flag = "" if len(arr) == expected_per_cell else "  <-- INCOMPLETE"
            print(f"{mode:10s} {sev:<4d} {acc_m:5.2f} ± {acc_s:4.2f}     {ece_m:5.2f} ± {ece_s:4.2f}     {len(arr)}{flag}")
    return by_key

# ---------------------- Main loop ----------------------
def main():
    if RESUME_FROM_INPUT and os.path.exists(RESUME_FROM_INPUT) and not os.path.exists(RESULTS_CSV):
        import shutil
        shutil.copy(RESUME_FROM_INPUT, RESULTS_CSV)
        print(f"[resume] copied prior results from {RESUME_FROM_INPUT} -> {RESULTS_CSV}")

    modes = ["source", "tent", "tent_temp", "sicl", "cgt_isocal", "sicl_isocal", "eata", "sar"]

    model_source = load_model()
    model_source.eval()

    print("[temp scaling] fitting temperature on clean CIFAR-10 val split...")
    val_loader = build_clean_val_loader()
    temperature = fit_temperature(model_source, val_loader)

    print(f"\n[eata] estimating Fisher information on {EATA_FISHER_BATCHES} clean source batches...")
    t0 = time.time()
    fisher, fisher_anchor = compute_fisher(model_source)
    print(f"[eata] Fisher estimate done in {time.time()-t0:.1f}s\n")

    done = load_completed_keys(RESULTS_CSV)
    print(f"[resume] {len(done)} (mode,sev,corr,seed) results already logged in {RESULTS_CSV}")

    try:
        for sev in SEVERITIES:
            for corr in CORRUPTIONS:
                ds = None
                for seed in SEEDS:
                    needed_modes = [m for m in modes if (m, sev, corr, seed) not in done]
                    if not needed_modes:
                        continue
                    if ds is None:
                        ds = load_cifar10c(corr, sev)
                    g = torch.Generator()
                    g.manual_seed(seed)
                    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
                    for mode in needed_modes:
                        acc, ece = run_condition(model_source, loader, mode, seed,
                                                  temperature=temperature,
                                                  fisher=fisher if mode == "eata" else None,
                                                  fisher_anchor=fisher_anchor if mode == "eata" else None)
                        append_result_row(RESULTS_CSV, mode, sev, corr, seed, acc, ece)
                        print(f"[sev={sev} corr={corr:18s} seed={seed} mode={mode:9s}] acc={acc:5.2f}  ece={ece:5.2f}")
    except Exception as e:
        print(f"\n[ERROR] Sweep stopped early: {e}")
        print("[ERROR] Falling back to summarizing whatever was logged before the error.\n")
    finally:
        by_key = summarize(modes)
    return by_key

# ---------------------- Statistical significance (unchanged) ----------------------
def paired_significance(by_key, modes_to_compare=("tent", "cgt_hard")):
    from scipy.stats import wilcoxon
    mode_a, mode_b = modes_to_compare
    print(f"\nPaired significance: ECE({mode_a}) vs ECE({mode_b})")
    print("=" * 60)
    for sev in SEVERITIES:
        keys_a, keys_b = by_key[(mode_a, sev)], by_key[(mode_b, sev)]
        common = sorted(set(keys_a) & set(keys_b))
        if not common:
            print(f"sev={sev}: no overlapping (corruption,seed) pairs found — check both modes ran.")
            continue
        a = np.array([keys_a[k] for k in common])
        b = np.array([keys_b[k] for k in common])
        diff = a - b
        n_wins = (diff > 0).sum()
        n_total = len(diff)
        try:
            stat, p = wilcoxon(a, b)
        except ValueError:
            p = float("nan")
        print(f"sev={sev}: {mode_b} better in {n_wins}/{n_total} runs, "
              f"mean ECE diff={diff.mean():+.3f}pp, Wilcoxon p={p:.4f}"
              f"{'  (significant at .05)' if p < 0.05 else ''}")

if __name__ == "__main__":
    assert "flip_agreement_pseudo_correct" in globals(), \
        "STOP: bank correctness signal is missing — do not run without the label-free fix."
    print("[sanity check] label-free pseudo-correctness signal present — OK to proceed\n")

    by_key = main()

    comparisons = [
        ("tent_temp", "sicl"),
        ("tent_temp", "cgt_isocal"),
        ("sicl", "cgt_isocal"),
        ("sicl", "sicl_isocal"),
        ("cgt_isocal", "sicl_isocal"),
        ("tent", "eata"),          # sanity: does EATA beat plain TENT?
        ("tent", "sar"),           # sanity: does SAR beat plain TENT?
        ("eata", "sicl_isocal"),   # THE headline comparison: proposed method vs. published label-free baseline
        ("sar", "sicl_isocal"),    # THE headline comparison: proposed method vs. published label-free baseline
    ]
    for pair in comparisons:
        try:
            paired_significance(by_key, pair)
        except Exception as e:
            print(f"\n[significance test skipped for {pair}: {e}]")
