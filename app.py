import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import fastparquet

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

df_melted = filtered.melt(
    id_vars=['week_start', 'locality'],
    value_vars=[
        'enemy_action_adj',
        'explosive_hazard_adj'
    ],
    var_name='event_type',
    value_name='adjusted_level'
)



st.write(filtered)


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

# Render inside Streamlit
st.pyplot(fig)
