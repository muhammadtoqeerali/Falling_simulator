
import pandas as pd
from pathlib import Path


rows = [

["Forward fall when trying to sit down",
"F01",
"F01",
"Scenario 20"],

["Backward fall when trying to sit down",
"F02",
"F02",
"Scenario 21"],

["Lateral fall when trying to sit down",
"F03",
"F03",
"Scenario 22"],

["Forward fall when trying to get up",
"F04",
"F04",
"Scenario 23"],

["Lateral fall when trying to get up",
"F05",
"F05",
"Scenario 24"],

["Forward fall while sitting caused by fainting",
"F06",
"F06",
"Scenario 25"],

["Lateral fall while sitting caused by fainting",
"F07",
"F07",
"Scenario 26"],

["Backward fall while sitting caused by fainting",
"F08",
"F08",
"Scenario 27"],

["Walking fall caused by fainting",
"F09",
"F09",
"Scenario 28"],

["Protective fall using hands",
"F10",
"F10",
"Scenario 29"],

["Walking trip fall",
"F11",
"F11",
"Scenario 30"],

["Jogging trip fall",
"F12",
"F12",
"Scenario 31"],

["Walking slip forward",
"F13",
"F13",
"Scenario 32"],

["Walking slip lateral",
"F14",
"F14",
"Scenario 33"],

["Walking slip backward",
"F15",
"F15",
"Scenario 34"],

["Slow backward movement fall",
"-",
"F16",
"Scenario 37"],

["Fast backward movement fall",
"-",
"F17",
"Scenario 38"],

["Forward height fall",
"-",
"F18",
"Scenario 39"],

["Backward height fall",
"-",
"F19",
"Scenario 40"],

["Stair climbing up fall",
"-",
"F20",
"Scenario 41"],

["Stair climbing down fall",
"-",
"F21",
"Scenario 42"]

]


df=pd.DataFrame(
rows,
columns=[
"Fall_Category",
"KFall_Label",
"UniVRFall_Label",
"Extended35_Scenario"
]
)


out=Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35/"
"Extended35_KFall_UniVRFall_Taxonomy_Comparison.csv"
)


df.to_csv(out,index=False)


print(df)

print("\nSaved:")
print(out)

