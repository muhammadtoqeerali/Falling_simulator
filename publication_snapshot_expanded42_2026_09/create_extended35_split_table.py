
import pandas as pd
from pathlib import Path


base = Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


rows = [

{
"split":"Full simulation campaign",
"scenarios":26,
"trials":572,
"purpose":"Complete generated dataset"
},

{
"split":"Validation subset",
"scenarios":21,
"trials":462,
"purpose":"Comparison against UniVRFall and KFall taxonomy"
},

{
"split":"Extension scenarios",
"scenarios":5,
"trials":110,
"purpose":"Additional simulator-only scenarios"
}

]


out = base / "Extended35_Dataset_Split_Definition.csv"

pd.DataFrame(rows).to_csv(
    out,
    index=False
)


print(pd.DataFrame(rows))

print("\nSaved:")
print(out)

