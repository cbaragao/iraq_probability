import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

#st.title("Combat Incident Exposure Probability Viewer")

st.markdown("""
## 🧭 Combat Exposure Modeling Tool
This interactive app shows **weekly probabilities** of combat-related incidents during the Iraq War..

**Purpose:** Providing evidence-aligned probability estimates that could support VA’s *“at least as likely as not”* standard.

- 🔵 **Enemy Action** 
- 🟠 **Explosive Hazard** 
- ⚫ **Dashed Line @ 0.5** = VA's burden-of-proof threshold

---
""")


loaded = pd.read_parquet("data/probabilities.parquet")

# Select distinct localities
distinct_locs = duckdb.sql("SELECT DISTINCT locality FROM loaded ORDER BY locality").df()

# Extract as list
locality_list = distinct_locs['locality'].tolist()

# Streamlit dropdown
selected_locality = st.selectbox("Select a locality:", sorted(locality_list))

# Filter and show data
filtered = loaded[loaded['locality'] == selected_locality]

# Convert 'year_week' to datetime (for x-axis sorting)
filtered['week_start'] = pd.to_datetime(filtered['year_week'] + '-0', format='%Y-%U-%w')

min_date, max_date = filtered['week_start'].min(), filtered['week_start'].max()
start_date, end_date = st.slider(
    "Date Range:",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime())
)


filtered = filtered[(filtered['week_start'] >= start_date) & (filtered['week_start'] <= end_date)]

df_melted = filtered.melt(
    id_vars=['week_start', 'locality'],
    value_vars=[
        'enemy_action_adj',
        'explosive_hazard_adj'
    ],
    var_name='event_type',
    value_name='adjusted_level'
)






# Plotting with seaborn and matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
sns.set_theme(style='darkgrid')

sns.scatterplot(
    data=df_melted,
    x='week_start',
    y='adjusted_level',
    hue='event_type',
    ax=ax
)

ax.set_title(f'Probability of Incident Types Over Time (By Year-Week) – {filtered["locality"].iloc[0]}')
ax.set_xlabel('')
ax.set_ylabel('Adjusted Level')

# Clean up x-axis
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.setp(ax.get_xticklabels(), rotation=45)

# threshold line
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold = 0.5')

# Legend
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))




st.sidebar.markdown("""
### What is Adjusted Probability?

Each point shows the **probability that at least one event** (e.g., enemy action or explosive hazard) occurred during a week in the selected locality.

- **≥ 0.5:** Indicates it is _at least as likely as not_ that exposure occurred that week (meets VA threshold).
- **< 0.5:** Less likely, but may still support claims if corroborated by other evidence.

Probabilities are derived from weekly rolling counts of events, smoothed to account for reporting gaps.
""")

st.metric("Weeks with ≥ 50% Enemy Action", f"{(filtered['enemy_action_adj'] >= 0.5).sum()} weeks")
st.metric("Weeks with ≥ 50% Explosive Hazards", f"{(filtered['explosive_hazard_adj'] >= 0.5).sum()} weeks")

# Render inside Streamlit
st.pyplot(fig)
