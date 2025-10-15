import argparse, math, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

class MatchDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

class Logistic(nn.Module):
    def __init__(self, inDim):
        super().__init__()
        self.lin = nn.Linear(inDim, 1)
    def forward(self, x): return self.lin(x)

class Mlp(nn.Module):
    def __init__(self, inDim, hidden=(64,32), p=0.2):
        super().__init__()
        layers, last = [], inDim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def selectFeatures(df):
    nonFeat = {
        "tourneyId","tourneyName","tourneyLevel","tourneyDate",
        "playerAName","playerAIoc","playerBName","playerBIoc","target"
    }
    cols = []
    for c in df.columns:
        if c in nonFeat: continue
        if c.startswith(("surf_","aHand_","bHand_")):
            cols.append(c)
        elif c in [
            "rankDiff","ageDiff","htDiff",
            "playerASeed","playerARank","playerAHt","playerAAge",
            "playerBSeed","playerBRank","playerBHt","playerBAge"
        ]:
            cols.append(c)
    return cols

def findContinuousIdx(cols):
    cont = {
        "rankDiff","ageDiff","htDiff",
        "playerASeed","playerARank","playerAHt","playerAAge",
        "playerBSeed","playerBRank","playerBHt","playerBAge"
    }
    return np.array([i for i,c in enumerate(cols) if c in cont], dtype=int)

def splitByTime(df, testFrom):
    dateCol = pd.to_numeric(df["tourneyDate"], errors="coerce").fillna(0).astype(int)
    trainMask = dateCol < testFrom
    testMask  = dateCol >= testFrom
    return trainMask, testMask

def applyStandardize(x, contIdx, mean, std):
    x = x.copy()
    if contIdx.size > 0:
        std = std.copy()
        std[std == 0] = 1.0
        x[:, contIdx] = (x[:, contIdx] - mean) / std
    return x

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    if len(loader.dataset) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    allLogits, allY, totalLoss = [], [], 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        totalLoss += loss.item() * xb.size(0)
        allLogits.append(logits.cpu().numpy())
        allY.append(yb.cpu().numpy())
    yTrue = np.vstack(allY).ravel()
    yLogits = np.vstack(allLogits).ravel()
    yProb = 1 / (1 + np.exp(-yLogits))
    auc = roc_auc_score(yTrue, yProb) if len(np.unique(yTrue)) > 1 else float("nan")
    ll  = log_loss(yTrue, yProb, labels=[0,1])
    acc = accuracy_score(yTrue, (yProb >= 0.5).astype(int))
    return totalLoss/len(loader.dataset), auc, ll, acc

def main():
    ap = argparse.ArgumentParser(description="Train PyTorch model for ATP match prediction.")
    ap.add_argument("csv", help="Path to mirrored/cleaned CSV (with target)")
    ap.add_argument("--model", choices=["logreg","mlp"], default="mlp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batchSize", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weightDecay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", type=str, default="64,32")
    ap.add_argument("--valSize", type=float, default=0.15)
    ap.add_argument("--testFrom", type=int, default=20250101, help="YYYYMMDD split point for test set")
    ap.add_argument("--out", default="model_artifacts.pth")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).fillna(0)

    featCols = selectFeatures(df)
    y = df["target"].astype(int).values
    x = df[featCols].astype(float).values

    trainMask, testMask = splitByTime(df, args.testFrom)
    xTrainFull, yTrainFull = x[trainMask], y[trainMask]
    xTest, yTest = x[testMask], y[testMask]

    if xTest.shape[0] == 0:
        print(f"[warn] no rows >= {args.testFrom}; using a random 15% of train as test")
        xTrainFull, xTest, yTrainFull, yTest = train_test_split(
            xTrainFull, yTrainFull, test_size=0.15, stratify=yTrainFull, random_state=42
        )

    xTr, xVal, yTr, yVal = train_test_split(
        xTrainFull, yTrainFull, test_size=args.valSize, stratify=yTrainFull, random_state=42
    )

    contIdx = findContinuousIdx(featCols)
    # compute mean/std on training only
    if contIdx.size > 0:
        mean = xTr[:, contIdx].mean(axis=0)
        std  = xTr[:, contIdx].std(axis=0)
    else:
        mean = np.array([])
        std  = np.array([])

    xTr  = applyStandardize(xTr,  contIdx, mean, std)
    xVal = applyStandardize(xVal, contIdx, mean, std)
    xTestStd = applyStandardize(xTest, contIdx, mean, std)

    trainDs = MatchDataset(xTr, yTr)
    valDs   = MatchDataset(xVal, yVal)
    testDs  = MatchDataset(xTestStd, yTest)

    trainLd = DataLoader(trainDs, batch_size=args.batchSize, shuffle=True,  num_workers=0)
    valLd   = DataLoader(valDs,   batch_size=args.batchSize, shuffle=False, num_workers=0)
    testLd  = DataLoader(testDs,  batch_size=args.batchSize, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inDim = x.shape[1]

    if args.model == "logreg":
        model = Logistic(inDim)
    else:
        hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())
        model = Mlp(inDim, hidden=hidden, p=args.dropout)

    model.to(device)
    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightDecay)

    bestVal = math.inf
    bestState = None
    patience, patienceLeft = 5, 5
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for xb, yb in trainLd:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        trLoss = total / len(trainLd.dataset)

        valLoss, valAuc, valLogLoss, valAcc = evaluate(model, valLd, crit, device)
        print(f"epoch {epoch:02d} | trainLoss {trLoss:.4f} | valLoss {valLoss:.4f} | valLogLoss {valLogLoss:.4f} | valAUC {valAuc:.3f} | valAcc {valAcc:.3f}")

        if not np.isnan(valLogLoss) and valLogLoss < bestVal - 1e-4:
            bestVal = valLogLoss
            bestState = {
                "model": model.state_dict(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "featCols": featCols,
                "inDim": inDim,
                "arch": args.model,
                "hidden": args.hidden,
                "dropout": args.dropout
            }
            patienceLeft = patience
        else:
            patienceLeft -= 1
            if patienceLeft == 0:
                print("early stop")
                break

    if bestState is None:
        bestState = {
            "model": model.state_dict(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "featCols": featCols,
            "inDim": inDim,
            "arch": args.model,
            "hidden": args.hidden,
            "dropout": args.dropout
        }

    model.load_state_dict(bestState["model"])
    tLoss, tAuc, tLogLoss, tAcc = evaluate(model, testLd, crit, device)
    print(f"TEST | loss {tLoss:.4f} | logLoss {tLogLoss:.4f} | AUC {tAuc:.3f} | acc {tAcc:.3f}")

    torch.save(bestState, args.out)
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()
