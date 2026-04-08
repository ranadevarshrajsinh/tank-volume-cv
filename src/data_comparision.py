import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import re

# --- Select Files ---
Tk().withdraw()
feb_file = filedialog.askopenfilename(title="Select FEB Excel File", filetypes=[("Excel Files", "*.xlsx *.xls")])
nov_file = filedialog.askopenfilename(title="Select NOV Excel File", filetypes=[("Excel Files", "*.xlsx *.xls")])

# --- Load Data ---
feb_df = pd.read_excel(feb_file)
nov_df = pd.read_excel(nov_file)

# --- Detect tank column automatically ---
def find_tank_column(df):
    for col in df.columns:
        if "tank" in col.lower() or "name" in col.lower():
            return col
    return None

tank_col_feb = find_tank_column(feb_df)
tank_col_nov = find_tank_column(nov_df)

if not tank_col_feb or not tank_col_nov:
    raise KeyError("Could not find any column with 'Tank' or 'Name' in the Excel sheets.")

# --- Clean tank names to common format (remove _feb, _nov, .png etc.) ---
def clean_tank_name(name):
    name = str(name).lower()
    name = re.sub(r'(_feb|_nov|\.png|\.jpg|\.jpeg)$', '', name)
    return name.strip()

feb_df["Tank Name"] = feb_df[tank_col_feb].apply(clean_tank_name)
nov_df["Tank Name"] = nov_df[tank_col_nov].apply(clean_tank_name)

# --- Convert shadow percentage to numeric ---
feb_df["Shadow %"] = pd.to_numeric(feb_df["Shadow %"], errors="coerce")
nov_df["Shadow %"] = pd.to_numeric(nov_df["Shadow %"], errors="coerce")

# --- Formula: V = 100 * (1 - (s/100))^γ ---
gamma = 1.8
feb_df["Volume Calc"] = 100 * ((1 - (feb_df["Shadow %"] / 100)) ** gamma)
nov_df["Volume Calc"] = 100 * ((1 - (nov_df["Shadow %"] / 100)) ** gamma)

# --- Compute per-tank averages ---
feb_avg = feb_df.groupby("Tank Name")["Volume Calc"].mean().reset_index()
nov_avg = nov_df.groupby("Tank Name")["Volume Calc"].mean().reset_index()

# --- Multiply with total capacity (70 million barrels) ---
total_capacity = 70_000_000
feb_avg["Volume Calc"] = feb_avg["Volume Calc"] * (total_capacity / 100)
nov_avg["Volume Calc"] = nov_avg["Volume Calc"] * (total_capacity / 100)

# --- Merge and align Feb vs Nov ---
merged = pd.merge(feb_avg, nov_avg, on="Tank Name", how="outer", suffixes=("_Feb", "_Nov")).fillna(0)

# --- Melt for seaborn ---
melted = pd.melt(merged, id_vars="Tank Name", value_vars=["Volume Calc_Feb", "Volume Calc_Nov"],
                 var_name="Dataset", value_name="Average Volume")

# --- Beautify dataset labels ---
melted["Dataset"] = melted["Dataset"].replace({
    "Volume Calc_Feb": "February",
    "Volume Calc_Nov": "November"
})

# --- Seaborn styling ---
sns.set(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("crest", n_colors=2)

# --- Plot per tank ---
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=melted, x="Tank Name", y="Average Volume", hue="Dataset", palette=palette)
ax.set_title("Tank-wise Average Volume Comparison (Feb vs Nov)", fontsize=16, weight='bold')
ax.set_xlabel("Tank Name")
ax.set_ylabel("Average Volume (Barrels)")
plt.xticks(rotation=45, ha='right')

# --- Add value labels ---
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", label_type="edge", fontsize=10)

plt.tight_layout()
plt.show()

# --- Overall averages for both datasets ---
overall = pd.DataFrame({
    "Dataset": ["February", "November"],
    "Average Volume": [
        feb_avg["Volume Calc"].mean(),
        nov_avg["Volume Calc"].mean()
    ]
})

# --- Overall average bar plot ---
plt.figure(figsize=(6, 5))
ax2 = sns.barplot(data=overall, x="Dataset", y="Average Volume", palette=palette)
ax2.set_title("Overall Average Volume Comparison", fontsize=16, weight='bold')
ax2.set_ylabel("Average Volume (Barrels)")

for container in ax2.containers:
    ax2.bar_label(container, fmt="%.0f", label_type="edge", fontsize=11)

plt.tight_layout()
plt.show()
