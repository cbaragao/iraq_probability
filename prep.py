import duckdb
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict
import re


# Query distinct values from DuckDB
# SQL query with FROM clause
base_query="""
SELECT
    CAST(date_time_occ AS DATE) AS attack_date, 
    ADM3NAME AS locality,
    type,
    DATEDIFF('day', CAST(strftime(date_time_occ, '%Y-%m') || '-01' AS DATE), last_day(CAST(date_time_occ AS DATE))) AS days_in_month,
    COUNT(1) AS attacks
FROM 'data/IQ_SIGACTs.csv'
GROUP BY date_time_occ,ADM3NAME, type, DATEDIFF('day', CAST(strftime(date_time_occ, '%Y-%m') || '-01' AS DATE), last_day(CAST(date_time_occ AS DATE)))
ORDER BY date_time_occ, ADM3NAME, type
"""

# Read CSV into DuckDB relation and convert to DataFrame
data = duckdb.sql(base_query).df()

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
dates_df.to_parquet('data/probabilities.parquet')
dates_df.to_csv('data/probabilities.csv', index=False)

# Load and process GDELT data
print("Loading GDELT data...")
gdelt_df = pd.read_parquet('data/iraq_gdelt_2003_2011_ollama.parquet')

def extract_localities_from_gdelt(paragraph):
    """Extract locality mentions from GDELT paragraph"""
    localities = set()
    
    # Get all localities from our probability data
    all_localities = dates_df['locality'].unique()
    
    # Search for locality mentions in the paragraph
    for locality in all_localities:
        if locality.lower() in paragraph.lower():
            localities.add(locality)
    
    return list(localities)

def analyze_event_trends(gdelt_df, dates_df):
    """Analyze trends in GDELT events and correlate with probability data"""
    
    # Create expanded GDELT data with locality mentions
    gdelt_expanded = []
    
    for _, row in gdelt_df.iterrows():
        localities = extract_localities_from_gdelt(row['paragraph'])
        for locality in localities:
            gdelt_expanded.append({
                'year_week': row['year_week'],
                'locality': locality,
                'paragraph': row['paragraph']
            })
    
    gdelt_expanded_df = pd.DataFrame(gdelt_expanded)
    
    # Count events per locality per week
    event_counts = gdelt_expanded_df.groupby(['locality', 'year_week']).size().reset_index(name='event_count')
    
    # Merge with probability data
    merged_df = dates_df.merge(event_counts, on=['locality', 'year_week'], how='left')
    merged_df['event_count'] = merged_df['event_count'].fillna(0)
    
    return merged_df, gdelt_expanded_df

def generate_trend_insights(merged_df, locality):
    """Generate trend insights for a specific locality"""
    
    locality_data = merged_df[merged_df['locality'] == locality].copy()
    
    if locality_data.empty:
        return {}
    
    # Calculate trend metrics
    insights = {}
    
    # Peak periods analysis
    locality_data['total_probability'] = locality_data['enemy_action_adj'] + locality_data['explosive_hazard_adj']
    
    # Find top 5 highest activity periods
    top_periods = locality_data.nlargest(5, 'total_probability')[['year_week', 'total_probability', 'event_count']]
    
    # Calculate yearly averages
    locality_data['year'] = locality_data['year_week'].str[:4]
    yearly_stats = locality_data.groupby('year').agg({
        'enemy_action_adj': 'mean',
        'explosive_hazard_adj': 'mean',
        'event_count': 'sum'
    }).reset_index()
    
    # Find peak year
    yearly_stats['total_avg'] = yearly_stats['enemy_action_adj'] + yearly_stats['explosive_hazard_adj']
    peak_year = yearly_stats.loc[yearly_stats['total_avg'].idxmax()]
    
    # Calculate trend direction (using simple correlation)
    locality_data['week_num'] = range(len(locality_data))
    if len(locality_data) > 1:
        correlation = locality_data['week_num'].corr(locality_data['total_probability'])
        trend_direction = "increasing" if correlation > 0.1 else "decreasing" if correlation < -0.1 else "stable"
    else:
        trend_direction = "insufficient data"
    
    insights = {
        'peak_periods': top_periods.to_dict('records'),
        'peak_year': {
            'year': peak_year['year'],
            'avg_enemy_prob': float(peak_year['enemy_action_adj']),
            'avg_explosive_prob': float(peak_year['explosive_hazard_adj']),
            'total_events': int(peak_year['event_count'])
        },
        'trend_direction': trend_direction,
        'yearly_stats': yearly_stats.to_dict('records'),
        'total_weeks_analyzed': len(locality_data),
        'correlation_coefficient': float(correlation) if len(locality_data) > 1 else None
    }
    
    return insights

def create_date_range_cache(merged_df):
    """Create cached analyses for common date ranges"""
    
    cache = {}
    
    # Define common date ranges
    date_ranges = [
        ('2003-2004', '2003-00', '2004-52'),
        ('2004-2005', '2004-00', '2005-52'),
        ('2005-2006', '2005-00', '2006-52'),
        ('2006-2007', '2006-00', '2007-52'),
        ('2007-2008', '2007-00', '2008-52'),
        ('2008-2009', '2008-00', '2009-52'),
        ('2009-2010', '2009-00', '2010-52'),
        ('2010-2011', '2010-00', '2011-52'),
        ('peak_violence', '2004-00', '2007-52'),  # Peak violence period
        ('surge_period', '2007-00', '2008-52'),   # Surge period
        ('full_period', '2003-00', '2011-52')     # Full period
    ]
    
    localities = merged_df['locality'].unique()
    
    for period_name, start_week, end_week in date_ranges:
        cache[period_name] = {}
        
        # Filter data for this period
        period_data = merged_df[
            (merged_df['year_week'] >= start_week) & 
            (merged_df['year_week'] <= end_week)
        ]
        
        for locality in localities:
            locality_period_data = period_data[period_data['locality'] == locality]
            
            if not locality_period_data.empty:
                enemy_weeks_high = (locality_period_data['enemy_action_adj'] >= 0.5).sum()
                explosive_weeks_high = (locality_period_data['explosive_hazard_adj'] >= 0.5).sum()
                total_weeks = len(locality_period_data)
                total_events = locality_period_data['event_count'].sum()
                
                cache[period_name][locality] = {
                    'enemy_weeks_high': int(enemy_weeks_high),
                    'explosive_weeks_high': int(explosive_weeks_high),
                    'total_weeks': int(total_weeks),
                    'total_events': int(total_events),
                    'enemy_percentage': float(enemy_weeks_high / total_weeks * 100) if total_weeks > 0 else 0,
                    'explosive_percentage': float(explosive_weeks_high / total_weeks * 100) if total_weeks > 0 else 0,
                    'avg_enemy_prob': float(locality_period_data['enemy_action_adj'].mean()),
                    'avg_explosive_prob': float(locality_period_data['explosive_hazard_adj'].mean())
                }
    
    return cache

print("Analyzing event trends and correlations...")
merged_df, gdelt_expanded_df = analyze_event_trends(gdelt_df, dates_df)

print("Generating trend insights...")
trend_insights = {}
localities = dates_df['locality'].unique()

for locality in localities:
    trend_insights[locality] = generate_trend_insights(merged_df, locality)

print("Creating date range cache...")
date_range_cache = create_date_range_cache(merged_df)

# Save enhanced data
print("Saving enhanced analysis data...")
enhanced_data = {
    'date_range_cache': date_range_cache,
    'trend_insights': trend_insights,
    'gdelt_summary': {
        'total_events': len(gdelt_expanded_df),
        'localities_mentioned': len(gdelt_expanded_df['locality'].unique()),
        'weeks_covered': len(gdelt_df)
    }
}

with open('data/enhanced_analysis.json', 'w') as f:
    json.dump(enhanced_data, f, indent=2)

# Save merged dataset for potential future analysis
merged_df.to_parquet('data/merged_combat_gdelt.parquet')
gdelt_expanded_df.to_parquet('data/gdelt_by_locality.parquet')

print("Enhanced data architecture created successfully!")
print(f"- Date range cache created for {len(date_range_cache)} periods")
print(f"- Trend insights generated for {len(trend_insights)} localities")
print(f"- GDELT events processed: {len(gdelt_expanded_df)} locality-event pairs")
print(f"- Merged dataset saved with {len(merged_df)} records")

def generate_locality_analysis(locality_data, gdelt_data=None):
    """Generate pre-computed analysis for a specific locality"""
    
    # Basic statistics
    total_weeks = len(locality_data)
    enemy_weeks_high = (locality_data['enemy_action_adj'] >= 0.5).sum()
    explosive_weeks_high = (locality_data['explosive_hazard_adj'] >= 0.5).sum()
    
    # Calculate percentages
    enemy_pct = (enemy_weeks_high / total_weeks * 100) if total_weeks > 0 else 0
    explosive_pct = (explosive_weeks_high / total_weeks * 100) if total_weeks > 0 else 0
    
    # Peak activity periods
    enemy_peak = locality_data.loc[locality_data['enemy_action_adj'].idxmax()] if not locality_data.empty else None
    explosive_peak = locality_data.loc[locality_data['explosive_hazard_adj'].idxmax()] if not locality_data.empty else None
    
    # Time period analysis
    start_date = locality_data['year_week'].min()
    end_date = locality_data['year_week'].max()
    
    # Generate summary text
    summary = f"""**Combat Exposure Analysis for {locality_data['locality'].iloc[0] if not locality_data.empty else 'Unknown'}**

**Overall Assessment:**
During the period from {start_date} to {end_date}, this locality experienced varying levels of combat activity across {total_weeks} weeks of data.

**Key Findings:**
• **Enemy Action**: {enemy_weeks_high} weeks ({enemy_pct:.1f}%) met or exceeded the 50% probability threshold
• **Explosive Hazards**: {explosive_weeks_high} weeks ({explosive_pct:.1f}%) met or exceeded the 50% probability threshold"""

    if enemy_peak is not None:
        summary += f"\n• **Peak Enemy Activity**: Week {enemy_peak['year_week']} (probability: {enemy_peak['enemy_action_adj']:.2f})"
    
    if explosive_peak is not None:
        summary += f"\n• **Peak Explosive Activity**: Week {explosive_peak['year_week']} (probability: {explosive_peak['explosive_hazard_adj']:.2f})"

    # VA claim relevance
    if enemy_weeks_high > 0 or explosive_weeks_high > 0:
        summary += f"\n\n**VA Claim Relevance:**\nThis data shows {enemy_weeks_high + explosive_weeks_high} total weeks where combat exposure probabilities met the VA's 'at least as likely as not' standard (≥50% probability). This evidence could support claims of combat exposure during the specified periods."
    else:
        summary += f"\n\n**VA Claim Relevance:**\nWhile no weeks exceeded the 50% threshold, the data shows measurable combat activity. Lower probability periods may still support claims when combined with other evidence."

    return {
        'locality': locality_data['locality'].iloc[0] if not locality_data.empty else 'Unknown',
        'total_weeks': int(total_weeks),
        'enemy_weeks_high': int(enemy_weeks_high),
        'explosive_weeks_high': int(explosive_weeks_high),
        'enemy_percentage': float(enemy_pct),
        'explosive_percentage': float(explosive_pct),
        'date_range': f"{start_date} to {end_date}",
        'peak_enemy_week': str(enemy_peak['year_week']) if enemy_peak is not None else None,
        'peak_enemy_prob': float(enemy_peak['enemy_action_adj']) if enemy_peak is not None else None,
        'peak_explosive_week': str(explosive_peak['year_week']) if explosive_peak is not None else None,
        'peak_explosive_prob': float(explosive_peak['explosive_hazard_adj']) if explosive_peak is not None else None,
        'summary_text': summary
    }

# Generate analysis for all localities
print("Generating pre-computed analysis for all localities...")
localities = dates_df['locality'].unique()
analysis_data = {}

for locality in localities:
    locality_data = dates_df[dates_df['locality'] == locality].copy()
    analysis = generate_locality_analysis(locality_data)
    analysis_data[locality] = analysis

# Save analysis data
with open('data/locality_analysis.json', 'w') as f:
    json.dump(analysis_data, f, indent=2)

print(f"Analysis generated for {len(localities)} localities and saved to data/locality_analysis.json")
