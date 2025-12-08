# 0) Environment
```Python3
# new venv (optional)
python -m venv venv && source venv/bin/activate

# core libs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install anndata scanpy pandas numpy scikit-learn scipy tqdm matplotlib

# optional (DE tests / QC helpers)
pip install statsmodels
```
# 1) Put your data in a standard shape
You’ll need three things for each cell:
- Expression: log-normalized vector (genes × 1)
- Labels: perturbation (or control), context (cell line / donor / cell type), optional batch
- Split tags: train / val / test according to the task you run (Section 4.2 tasks in the paper)
Most VCC-style releases come as AnnData (.h5ad) or MTX/CSV matrices + metadata. If you have MTX/CSV, load to AnnData first. Example loaders:
```Python3
import scanpy as sc
import pandas as pd
import numpy as np

# OPTION 1: h5ad (easiest)
adata = sc.read_h5ad("vcc.h5ad")

# OPTION 2: MTX + barcodes + genes + metadata.csv
# adata = sc.read_mtx("matrix.mtx.gz").T  # cells x genes
# adata.var["gene_symbol"] = pd.read_csv("genes.tsv", sep="\t", header=None)[1].values
# adata.obs_names = pd.read_csv("barcodes.tsv", sep="\t", header=None)[0].values
# meta = pd.read_csv("metadata.csv")
# adata.obs = adata.obs.join(meta.set_index("cell_id").loc[adata.obs_names])

# Expect columns like:
# adata.obs["perturbation"]  # string label ('ctrl' for control)
# adata.obs["context"]       # e.g., cell line/donor
# adata.obs["batch"]         # optional

```

# 2) Basic QC → normalization → HVG selection (paper-consistent)
The paper log-normalizes counts after depth normalization and uses top 2,000 HVGs for the ST+HVG track.

```Python3
import scanpy as sc

# Basic filters (tune thresholds per dataset)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=50)

# Normalize + log1p (paper uses Scanpy normalize_total->log1p)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Optional: regress-out / scale (depends on data)
# sc.pp.regress_out(adata, ['total_counts'])
# sc.pp.scale(adata, max_value=10)

# Highly variable genes for Track A
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
hvg_mask = adata.var["highly_variable"].values
```
# 3) Create sets and pairs (the core of ST)
State trains on sets of S cells that are matched on context (and optionally batch), and pairs each perturbed set with a control set from the same context.

```Python3
import numpy as np
from collections import defaultdict

def build_sets(adata, S=128, use_hvg=True):
    """Return lists of matched (ctrl_set, target_set, perturbation_label, context, batch) arrays."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    if use_hvg:
        X = X[:, hvg_mask]

    obs = adata.obs
    # group cells by (context, perturbation, batch)
    groups = defaultdict(list)
    for idx, (ctx, pert, bch) in enumerate(zip(obs["context"], obs["perturbation"], obs.get("batch", pd.Series(["na"]*adata.n_obs)))):
        groups[(ctx, pert, bch)].append(idx)

    # split into S-sized sets per group (pad last with sampling w/ replacement)
    def chunkify(idxs, S):
        idxs = np.array(idxs)
        sets = []
        if len(idxs) == 0: return sets
        n_full = len(idxs)//S
        for k in range(n_full):
            sets.append(idxs[k*S:(k+1)*S])
        if len(idxs) % S != 0:
            rest = idxs[n_full*S:]
            pad = np.random.choice(rest, size=S-len(rest), replace=True)
            sets.append(np.concatenate([rest, pad]))
        return sets

    # build matched pairs (ctrl vs each pert)
    pairs = []
    for (ctx, pert, bch), idxs in groups.items():
        if pert == "ctrl": 
            continue
        target_sets = chunkify(idxs, S)

        ctrl_key = (ctx, "ctrl", bch) if ("ctrl" in {k[1] for k in groups.keys() if k[0]==ctx}) else (ctx, "ctrl", "na")
        if ctrl_key not in groups: 
            continue
        ctrl_sets = chunkify(groups[ctrl_key], S)
        if len(ctrl_sets) == 0 or len(target_sets) == 0:
            continue

        # pair by cycling through available control sets
        for i, tset in enumerate(target_sets):
            cset = ctrl_sets[i % len(ctrl_sets)]
            pairs.append((X[cset], X[tset], pert, ctx, bch))
    return pairs

```
# 4) Track A: ST+HVG (fastest start)
Train ST directly on HVG expression (input B×S×G, output B×S×G) with MMD loss.
- Set size S and batch size B matter; the paper observes performance gains up to an optimal S (e.g., 256 on a large dataset).
- Start smaller (S=64–128) if memory is tight; then scale.
  
```Python3
import torch
import numpy as np
from tqdm import trange
from state_model import StateTransitionModel, compute_mmd

pairs = build_sets(adata, S=128, use_hvg=True)
print("num training pairs:", len(pairs))

# build vocab of perturbations for one-hot
perts = sorted(list({p for (_,_,p,_,_) in pairs} | {"ctrl"}))
pert2i = {p:i for i,p in enumerate(perts)}
P = len(perts)
G = hvg_mask.sum()

# model
st = StateTransitionModel(
    num_genes=G,            # input/output dim
    hidden_dim=256,
    transformer_heads=8,
    transformer_layers=4,
    mlp_hidden_dim=512,
    perturbation_dim=P,     # one-hot length
).train()

opt = torch.optim.AdamW(st.parameters(), lr=3e-4, weight_decay=1e-2)

def one_hot(batch_perts):
    # perts are strings; we need shape (B,S,P)
    B, S = len(batch_perts), len(batch_perts[0])
    z = torch.zeros(B, S, P)
    for b in range(B):
        p = batch_perts[b][0]          # set-level label (same across S)
        z[b, :, pert2i[p]] = 1.0
    return z

BATCH = 8
RBF_SIGMA = 1.0  # tune; can use median heuristic per batch

for step in trange(2000):
    batch = [pairs[np.random.randint(len(pairs))] for _ in range(BATCH)]
    Xctrl = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)  # (B,S,G)
    Xtarg = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)  # (B,S,G)
    perts_batch = [[b[2]]*Xctrl.shape[1] for b in batch]                        # repeat within set
    Zpert = one_hot(perts_batch)

    Xhat = st(Xctrl, Zpert)
    # MMD between predicted & observed perturbed sets (eq. 19)
    loss = compute_mmd(Xhat.reshape(-1,G), Xtarg.reshape(-1,G), sigma=RBF_SIGMA)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print("step", step, "loss", float(loss))

```
# 5) Track B: ST+SE (embedding space)
Here you first learn cell embeddings (SE) then train ST on those embeddings (with a lightweight decoder back to genes if you need gene-space outputs).

## 5.1 Build gene features
The paper projects ESM-2 protein language model features for each gene into the model’s hidden space. Practically:
- Start simple: identity features (one-hot by gene) or a random projection as a placeholder.
- Upgrade later to real ESM-2 gene features (precompute per gene → np.load('gene_features.npy')).

```Python3
# Build gene features
import numpy as np

# simple placeholder: random features per gene
G_all = adata.n_vars
h = 256  # SE hidden dim
rng = np.random.default_rng(0)
gene_features = rng.normal(size=(G_all, 5120)).astype(np.float32)  # pretend ESM-2 size
```

## 5.2 Train SE on observational cells
You select top L genes per cell (by expression), add expression-aware bins, then encode via a Transformer to get a cell embedding.
(If your StateEmbeddingModel in state_model.py doesn’t yet include a full pretraining loss, you can: (i) add a small MLP “decoder” head that takes [z_cell || g_j] to predict x_ij for a random subset of genes per batch, and (ii) use MSE/Huber loss.)

```Python3
import torch
from torch.utils.data import DataLoader, Dataset
from state_model import StateEmbeddingModel

class SEDataset(Dataset):
    def __init__(self, adata, L=2048):
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        self.X = X.astype(np.float32)
        self.L = L
        self.gene_features = gene_features  # (G, 5120)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        idx = np.argsort(-x)[:self.L]       # top-L by expression
        vals = x[idx]
        gf = self.gene_features[idx]        # (L, 5120)
        return torch.tensor(gf), torch.tensor(vals), torch.tensor(idx)  # features, expr, gene idx

train_loader = DataLoader(SEDataset(adata), batch_size=16, shuffle=True)

se = StateEmbeddingModel(
    gene_feature_dim=5120,   # input feature size (ESM-2 dim in paper)
    hidden_dim=h,
    L=2048,                  # number of top genes per cell
).train()

optim = torch.optim.AdamW(se.parameters(), lr=2e-4)

for epoch in range(3):  # demo; use more
    for gf, vals, idx in train_loader:
        # se should implement expression-aware binning internally or accept vals
        z = se(gf, vals)  # (B, hidden_dim) cell embedding
        # self-supervised loss: predict a subset of genes’ expression (see Sec. 4.4.5)
        loss = se.loss(z, gf, vals, idx)    # your SE class should expose a training loss
        optim.zero_grad(); loss.backward(); optim.step()
```

## 5.3 Build sets in embedding space + train ST
Replace HVG matrices with SE cell embeddings: Xemb ∈ R^{B×S×E}. Train ST to predict perturbed embeddings and (optionally) a small fdecode back to gene space, using a weighted MMD in both spaces.
```Python3

# compute embeddings for all cells (control & pert)
# (Pseudo: you can batch it)
with torch.no_grad():
    # For each cell compute top-L and run se(...) to get z_cell
    pass

# Now build pairs as before, but using embeddings instead of HVGs:
pairs_emb = build_sets(adata, S=128, use_hvg=False)  # returns indices; replace X[...] with embeddings

E = h  # embedding dim
st_emb = StateTransitionModel(
    num_genes=E, hidden_dim=256, transformer_heads=8, transformer_layers=4,
    mlp_hidden_dim=512, perturbation_dim=P
).train()

# optional decoder from embedding back to gene space (simple MLP)
decode = torch.nn.Sequential(
    torch.nn.Linear(E, 512),
    torch.nn.GELU(),
    torch.nn.Linear(512, G_all)
).train()

opt = torch.optim.AdamW(list(st_emb.parameters())+list(decode.parameters()), lr=3e-4, weight_decay=1e-2)

ALPHA = 0.1  # weight for expression-space MMD (paper down-weights by 0.1):contentReference[oaicite:15]{index=15}

for step in trange(2000):
    batch = [pairs_emb[np.random.randint(len(pairs_emb))] for _ in range(BATCH)]
    Zctrl = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)  # (B,S,E)
    Ztarg = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    perts_batch = [[b[2]]*Zctrl.shape[1] for b in batch]
    Zpert = one_hot(perts_batch)

    Zhat = st_emb(Zctrl, Zpert)                    # (B,S,E)
    loss_emb = compute_mmd(Zhat.reshape(-1,E), Ztarg.reshape(-1,E), sigma=1.0)

    Xhat = decode(Zhat.reshape(-1,E)).reshape(Zhat.shape[0], Zhat.shape[1], -1)
    # Need target gene expression aligned to same cells:
    # Xtarg_gene = ...
    loss_expr = compute_mmd(Xhat.reshape(-1,G_all), Xtarg_gene.reshape(-1,G_all), sigma=1.0)

    loss = loss_emb + ALPHA*loss_expr
    opt.zero_grad(); loss.backward(); opt.step()

```
# 6) Splitting by task (for fair evaluation)
The paper uses two key generalization tasks (Methods 4.2) you can reproduce with your metadata:
- Underrepresented context: hold out most perturbations in a target context (keep ~30% in train), test on the rest.
- -shot context: fine-tune on other contexts, test on a completely held-out context (no perturbations from that context in training).

A simple splitter:
```Python3
def split_underrepresented(adata, target_context, alpha=0.30, seed=0):
    df = adata.obs
    rng = np.random.default_rng(seed)
    in_ctx = df["context"] == target_context
    perts = sorted(df.loc[in_ctx, "perturbation"].unique())
    perts = [p for p in perts if p != "ctrl"]
    k = max(1, int(alpha*len(perts)))
    keep = set(rng.choice(perts, size=k, replace=False))
    train_mask = (~in_ctx) | (in_ctx & df["perturbation"].isin(keep))
    test_mask  =  (in_ctx) & (~df["perturbation"].isin(keep)) & (df["perturbation"]!="ctrl")
    return train_mask.values, test_mask.values

```
Use the masks to subset adata before building sets, then train on train split, evaluate on test.

# 7, Evaluation
The challenge metrics map well to:
- Counts correlation: Pearson/Spearman between predicted and observed Δexpression (pert − ctrl)
- Perturbation discrimination: rank each predicted profile against true profiles and compute an inverse normalized rank score (PDS) (see paper Section 2.2 narrative).
- DE metrics: Wilcoxon rank-sum on predicted vs observed to compare DE p-values, logFC, and DE-gene overlap (Cell-Eval suite in paper Section 2.2; Methods 4.7 in full paper).
  
```Python3
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr, spearmanr

def delta_expr(X_pred_ctrl, X_pred_pert, X_true_ctrl, X_true_pert):
    return (X_pred_pert - X_pred_ctrl), (X_true_pert - X_true_ctrl)

# 1) Counts correlation (per perturbation, then average)
# 2) DE genes (Wilcoxon in Scanpy)
# sc.tl.rank_genes_groups(adata_true, groupby="perturbation", reference="ctrl", method="wilcoxon")
# sc.tl.rank_genes_groups(adata_pred, groupby="perturbation", reference="ctrl", method="wilcoxon")
# compare DE gene overlap, logFC Spearman, AUPRC on p-values, etc.

```
# 8, Inference loop after training
Given control cells from a context and a target perturbation:
```Python3
# Build one control set S from the context (and batch), and the set-level one-hot Zpert
Xhat = st(Xctrl, Zpert)  # Track A
# or
Zhat = st_emb(Zctrl, Zpert); Xhat = decode(Zhat)  # Track B

# Xhat is the predicted *perturbed* single-cell transcriptomes at S cells.
# You can average, run DE, compute effect size, etc., exactly as you do with observed data.

```
