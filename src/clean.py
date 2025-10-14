import argparse
import pandas as pd

# Script used to pick the features 

parser = argparse.ArgumentParser(description="Clean ATP matches and create mirrored rows.")
parser.add_argument("input", help="Path to input CSV")
parser.add_argument("output", help="Path to output CSV")
args = parser.parse_args()

df = pd.read_csv(args.input)

keepCols = [
    "tourney_id","tourney_name","surface","tourney_level","tourney_date",
    "winner_name","winner_hand","winner_ht","winner_ioc","winner_age","winner_seed","winner_rank",
    "loser_name","loser_hand","loser_ht","loser_ioc","loser_age","loser_seed","loser_rank"
]
df = df[keepCols].copy()

numCols = ["winner_ht","winner_age","winner_seed","winner_rank","loser_ht","loser_age","loser_seed","loser_rank"]
for c in numCols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

lvlMap = {"A":"ATP250","M":"Masters","G":"GrandSlam","F":"Finals"}
df["tourney_level"] = df["tourney_level"].map(lvlMap).fillna(df["tourney_level"])

aWin = pd.DataFrame({
    "tourneyId": df["tourney_id"],
    "tourneyName": df["tourney_name"],
    "surface": df["surface"],
    "tourneyLevel": df["tourney_level"],
    "tourneyDate": df["tourney_date"],
    "playerAName": df["winner_name"],
    "playerAHand": df["winner_hand"],
    "playerAHt": df["winner_ht"],
    "playerAIoc": df["winner_ioc"],
    "playerAAge": df["winner_age"],
    "playerASeed": df["winner_seed"],
    "playerARank": df["winner_rank"],
    "playerBName": df["loser_name"],
    "playerBHand": df["loser_hand"],
    "playerBHt": df["loser_ht"],
    "playerBIoc": df["loser_ioc"],
    "playerBAge": df["loser_age"],
    "playerBSeed": df["loser_seed"],
    "playerBRank": df["loser_rank"],
    "target": 1
})

aLose = pd.DataFrame({
    "tourneyId": df["tourney_id"],
    "tourneyName": df["tourney_name"],
    "surface": df["surface"],
    "tourneyLevel": df["tourney_level"],
    "tourneyDate": df["tourney_date"],
    "playerAName": df["loser_name"],
    "playerAHand": df["loser_hand"],
    "playerAHt": df["loser_ht"],
    "playerAIoc": df["loser_ioc"],
    "playerAAge": df["loser_age"],
    "playerASeed": df["loser_seed"],
    "playerARank": df["loser_rank"],
    "playerBName": df["winner_name"],
    "playerBHand": df["winner_hand"],
    "playerBHt": df["winner_ht"],
    "playerBIoc": df["winner_ioc"],
    "playerBAge": df["winner_age"],
    "playerBSeed": df["winner_seed"],
    "playerBRank": df["winner_rank"],
    "target": 0
})

out = pd.concat([aWin, aLose], ignore_index=True)

for c in ["playerAHt","playerAAge","playerASeed","playerARank","playerBHt","playerBAge","playerBSeed","playerBRank"]:
    out[c] = pd.to_numeric(out[c], errors="coerce")

out["rankDiff"] = out["playerARank"] - out["playerBRank"]
out["ageDiff"] = round(out["playerAAge"] - out["playerBAge"],3)
out["htDiff"] = out["playerAHt"] - out["playerBHt"]
out["sameHand"] = (out["playerAHand"] == out["playerBHand"]).astype(int)

out = pd.get_dummies(out, columns=["surface"], prefix="surf")
out = pd.get_dummies(out, columns=["playerAHand","playerBHand"], prefix=["aHand","bHand"])

out.to_csv(args.output, index=False)
print(f"✅ Saved {args.output} with {len(out)} rows and {len(out.columns)} columns.")
