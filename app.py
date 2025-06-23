import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

@st.cache_data
def load_data():
    return duckdb.sql("SELECT * FROM 'data/IQ_SIGACTs.csv'").df()

loaded_df = load_data()

# Query distinct values from DuckDB
# SQL query with FROM clause
base_query="""
SELECT
    CAST(date_time_occ AS DATE) AS attack_date, 
    ADM3NAME AS locality,
    type,
    DATEDIFF('day', CAST(strftime(date_time_occ, '%Y-%m') || '-01' AS DATE), last_day(CAST(date_time_occ AS DATE))) AS days_in_month,
    COUNT(1) AS attacks
FROM loaded_df
GROUP BY date_time_occ,ADM3NAME, type, DATEDIFF('day', CAST(strftime(date_time_occ, '%Y-%m') || '-01' AS DATE), last_day(CAST(date_time_occ AS DATE)))
ORDER BY date_time_occ, ADM3NAME, type
"""

# Read CSV into DuckDB relation and convert to DataFrame
data = duckdb.sql(base_query).df()

# Select distinct localities
distinct_locs = duckdb.sql("SELECT DISTINCT locality FROM data ORDER BY locality").df()

# Extract as list
locality_list = distinct_locs['locality'].tolist()

# Streamlit dropdown
selected_locality = st.selectbox("Select a locality:", sorted(locality_list))

awy_sql = '''
   SELECT
        attack_date,
        strftime(attack_date, '%Y-%U') AS awy,
        locality,
        type
    FROM data
'''

awy_df = duckdb.sql(awy_sql).df()

grouped_sql = ''' 
    SELECT
        attack_date,
        awy,
        locality,
        type,
        COUNT(1) AS attacked
    FROM awy_df
    GROUP BY attack_date, awy, locality, type
'''

grouped = duckdb.sql(grouped_sql).df()

logistic_sql = '''
SELECT
    attack_date,
    awy,
    locality,
    type,
    CASE WHEN attacked >0 THEN 1 else 0 END AS was_attacked
FROM grouped
'''

logistic= duckdb.sql(logistic_sql).df()

by_week_year = ''' 
SELECT
    awy,
    locality,
    type,
    SUM(was_attacked) / 7 AS probability
FROM logistic
GROUP BY awy, locality, type
ORDER BY awy, locality, type
'''

by_week_year_df = duckdb.sql(by_week_year).df()

pivot_sql = ''' 
    PIVOT by_week_year_df
    ON type
    USING SUM(probability)
    ORDER BY locality, awy;
'''
pivot_df = duckdb.sql(pivot_sql).to_df()
pivot_df = pivot_df.fillna(0)

rolling_avg_sql = ''' 
    SELECT
        awy,
        locality,
        "Enemy Action" AS enemy_action,
        AVG("Enemy Action") OVER (PARTITION BY  locality ORDER BY awy ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS enemy_action_ra,
        "Explosive Hazard" AS explosive_hazard,
        AVG("Explosive Hazard") OVER (PARTITION BY  locality ORDER BY awy ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS explosive_hazard_ra,
        "Friendly Fire" AS friendly_fire,
        AVG("Friendly Fire") OVER (PARTITION BY locality ORDER BY awy ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS friendly_fire_ra,
        "Host Nation Activity" AS host_nation_activity,
        AVG("Host_Nation_Activity") OVER (PARTITION BY awy,locality ORDER BY awy ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS host_nation_activity_ra
    FROM pivot_df
    ORDER BY locality, awy
'''

rolling_avg_df = duckdb.sql(rolling_avg_sql).df()

dates_sql = ''' 
WITH base AS (SELECT * FROM UNNEST( generate_series(
    (SELECT MIN(attack_date) FROM data),
    (SELECT MAX(attack_date) FROM data),
    INTERVAL 1 WEEK
))),
week_years AS
(
SELECT 
    strftime(unnest, '%Y-%U') AS dates
FROM base
),
localities AS (
SELECT DISTINCT(locality) AS locality FROM data),
cross_join AS (

    SELECT 
        wy.dates,
        l.locality,

    FROM  localities l
    CROSS JOIN  week_years wy
    ORDER BY l.locality, wy.dates
),
imputed_dates AS (
SELECT
    cj.dates,
    cj.locality,
    COALESCE(ra.enemy_action,0) AS enemy_action,
    COALESCE(ra.enemy_action_ra,0) AS enemy_action_ra,
    COALESCE(ra.explosive_hazard,0) AS explosive_hazard,
    COALESCE(ra.explosive_hazard_ra,0) AS explosive_hazard_ra,
    COALESCE(friendly_fire,0) AS friendly_fire,
    COALESCE(friendly_fire_ra,0) AS friendly_fire_ra,
    COALESCE(host_nation_activity,0) AS host_nation_activity,
    COALESCE(host_nation_activity_ra,0) AS host_nation_activity_ra
FROM cross_join cj
LEFT JOIN rolling_avg_df ra
ON cj.dates = ra.awy AND cj.locality = ra.locality
ORDER BY cj.locality, cj.dates)

SELECT 
    idt.dates AS year_week,
    idt.locality,
    CASE WHEN idt.enemy_action = 0 THEN idt.enemy_action_ra ELSE idt.enemy_action END AS enemy_action_adj,
    CASE WHEN idt.explosive_hazard=0 THEN  idt.explosive_hazard_ra ELSE idt.explosive_hazard END AS explosive_hazard_adj,
    CASE WHEN idt.friendly_fire=0 THEN idt.friendly_fire_ra ELSE idt.friendly_fire END AS friendly_fire_adj,
    CASE WHEN idt.host_nation_activity = 0 THEN idt.host_nation_activity_ra ELSE idt.host_nation_activity END AS host_nation_activity_adj
FROM imputed_dates idt

'''

dates_df = duckdb.sql(dates_sql).df()

# Filter and show data
filtered = dates_df[dates_df['locality'] == selected_locality]

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
