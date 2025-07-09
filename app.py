import duckdb
import pandas as pd
import streamlit as st
import json
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import warnings
warnings.filterwarnings('ignore')

st.markdown("""
## 🧭 Combat Exposure Modeling Tool
This interactive app shows **weekly probabilities** of combat-related incidents during the Iraq War.

**Purpose:** Providing evidence-aligned probability estimates that could support VA’s *“at least as likely as not”* standard.

- 🔴 **Enemy Action** 
- ⚫ **Explosive Hazard** 
- 〰️ **Dashed Line @ 0.5** = VA's burden-of-proof threshold

---
""")


loaded = pd.read_parquet("data/probabilities.parquet")

# Load pre-computed analysis and enhanced data
with open('data/locality_analysis.json', 'r') as f:
    analysis_data = json.load(f)

with open('data/enhanced_analysis.json', 'r') as f:
    enhanced_data = json.load(f)

gdelt = pd.read_parquet("data/iraq_gdelt_2003_2011_ollama.parquet")
gdelt['year_week'] = pd.to_datetime(gdelt['year_week'] + '-1', format='%Y-%W-%w')

# GDELT Analysis Functions
def parse_gdelt_events(paragraph):
    """Parse GDELT events from a paragraph and extract event types and locations"""
    events = []
    
    # Pattern to match: Location(Event: event_type; Description: description.)
    pattern = r'([^(]+)\(Event:\s*([^;]+);\s*Description:\s*([^)]+)\)'
    
    matches = re.findall(pattern, paragraph)
    
    for location, event_type, description in matches:
        events.append({
            'location': location.strip().rstrip(','),
            'event_type': event_type.strip(),
            'description': description.strip()
        })
    
    return events

def filter_events_by_locality(events, target_locality):
    """Filter events that mention the target locality"""
    filtered_events = []
    target_lower = target_locality.lower()
    
    for event in events:
        # Check if locality is mentioned in the location field
        if target_lower in event['location'].lower():
            filtered_events.append(event)
    
    return filtered_events

def analyze_gdelt_for_locality_and_timeframe(gdelt_df, target_locality, start_week, end_week):
    """Analyze GDELT events for a specific locality and timeframe"""
    
    # Filter GDELT data for the selected timeframe using string comparison
    filtered_gdelt = gdelt_df[
        (gdelt_df['year_week'] >= start_week) & 
        (gdelt_df['year_week'] <= end_week)
    ].copy()
    
    # Parse all events from the filtered timeframe
    all_events = []
    
    for _, row in filtered_gdelt.iterrows():
        week_events = parse_gdelt_events(row['paragraph'])
        locality_events = filter_events_by_locality(week_events, target_locality)
        
        # Add week information to each event
        for event in locality_events:
            event['week'] = row['year_week']
            all_events.append(event)
    
    return all_events

def get_top_event_types(events, top_n=10):
    """Get the top N event types by count"""
    event_counter = Counter([event['event_type'] for event in events])
    return event_counter.most_common(top_n)

def create_gdelt_visualization(event_counts, title):
    """Create a horizontal bar chart for GDELT event counts"""
    if not event_counts:
        return None
    
    # Extract event types and counts
    event_types = [item[0] for item in event_counts]
    counts = [item[1] for item in event_counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(event_types)), counts, color='darkred', alpha=0.7)
    
    # Customize the chart
    ax.set_yticks(range(len(event_types)))
    ax.set_yticklabels(event_types)
    ax.set_xlabel('Number of Events', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=10, fontweight='bold')
    
    # Style the plot
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Invert y-axis to show highest counts at the top
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

# Select distinct localities
distinct_locs = duckdb.sql("SELECT DISTINCT locality FROM loaded ORDER BY locality").df()

# Extract as list
locality_list = distinct_locs['locality'].tolist()

#set slicer columns
slicer1, slicer2 = st.columns(2)

# Streamlit dropdown
with slicer1:
    selected_locality = st.selectbox("Select a locality:", sorted(locality_list))

# Filter and show data
filtered = loaded[loaded['locality'] == selected_locality]

# Convert 'year_week' to datetime (for x-axis sorting)
filtered['week_start'] = pd.to_datetime(filtered['year_week'] + '-0', format='%Y-%U-%w')

min_date, max_date = filtered['week_start'].min(), filtered['week_start'].max()

with slicer2:
    start_date, end_date = st.slider(
        "Date Range:",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime())
    )

# Add space between slicers and next elements
st.markdown("<br>", unsafe_allow_html=True)

filtered = filtered[(filtered['week_start'] >= start_date) & (filtered['week_start'] <= end_date)]

filtered_gdelt = gdelt[(gdelt['year_week'] >= start_date) & 
                      (gdelt['year_week'] <= end_date) &
                      (gdelt['paragraph'].str.contains(selected_locality, case=False, na=False))]

filtered_gdelt.sort_values(by='year_week', inplace=True)

paragraph = " ".join(filtered_gdelt['paragraph'].tolist())
header = f"GDELT EVENT SUMMARY ({len(filtered_gdelt)} total events, showing up to 300):\n\n"
summary_body = header+paragraph


def create_interactive_probability_chart(filtered_data, locality_name):
    """Create an interactive Plotly chart for probability data"""
    
    # Create the interactive plot
    fig = go.Figure()
    
    # Add enemy action data
    fig.add_trace(go.Scatter(
        x=filtered_data['week_start'],
        y=filtered_data['enemy_action_adj'],
        mode='markers',
        name='Enemy Action',
        marker=dict(
            color='red',
            size=8,
            opacity=0.7,
            line=dict(width=1, color='darkred')
        ),
        hovertemplate='<b>Enemy Action</b><br>' +
                      'Date: %{x}<br>' +
                      'Probability: %{y:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add explosive hazard data
    fig.add_trace(go.Scatter(
        x=filtered_data['week_start'],
        y=filtered_data['explosive_hazard_adj'],
        mode='markers',
        name='Explosive Hazard',
        marker=dict(
            color='gray',
            size=8,
            opacity=0.7,
            line=dict(width=1, color='black')
        ),
        hovertemplate='<b>Explosive Hazard</b><br>' +
                      'Date: %{x}<br>' +
                      'Probability: %{y:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text="VA Threshold (50%)",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interactive Probability Chart - {locality_name}',
            x=0.5,
            font=dict(size=20, color='black')
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='lightgray',
            title_font=dict(size=14)
        ),
        yaxis=dict(
            title='Probability',
            showgrid=True,
            gridcolor='lightgray',
            title_font=dict(size=14),
            range=[0, 1.1]
        ),
        hovermode='closest',
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=900,
        height=600,
        plot_bgcolor='white'
    )
    
    return fig

# Create interactive chart
interactive_fig = create_interactive_probability_chart(filtered, filtered["locality"].iloc[0])

def export_data_to_csv(data, filename):
    """Export filtered data to CSV"""
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    b64 = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Download CSV</a>'
    return href

def create_comparison_chart(localities_data, locality_names):
    """Create a comparison chart for multiple localities"""
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, (locality_name, data) in enumerate(zip(locality_names, localities_data)):
        if data is not None and not data.empty:
            # Enemy action
            fig.add_trace(go.Scatter(
                x=data['week_start'],
                y=data['enemy_action_adj'],
                mode='markers',
                name=f'{locality_name} - Enemy Action',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=6,
                    opacity=0.6,
                    symbol='circle'
                ),
                hovertemplate=f'<b>{locality_name} - Enemy Action</b><br>' +
                              'Date: %{x}<br>' +
                              'Probability: %{y:.3f}<br>' +
                              '<extra></extra>'
            ))
            
            # Explosive hazard
            fig.add_trace(go.Scatter(
                x=data['week_start'],
                y=data['explosive_hazard_adj'],
                mode='markers',
                name=f'{locality_name} - Explosive Hazard',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=6,
                    opacity=0.6,
                    symbol='square'
                ),
                hovertemplate=f'<b>{locality_name} - Explosive Hazard</b><br>' +
                              'Date: %{x}<br>' +
                              'Probability: %{y:.3f}<br>' +
                              '<extra></extra>'
            ))
    
    # Add threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text="VA Threshold (50%)",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Locality Comparison - {", ".join(locality_names)}',
            x=0.5,
            font=dict(size=20, color='black')
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='lightgray',
            title_font=dict(size=14)
        ),
        yaxis=dict(
            title='Probability',
            showgrid=True,
            gridcolor='lightgray',
            title_font=dict(size=14),
            range=[0, 1.1]
        ),
        hovermode='closest',
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=1000,
        height=600,
        plot_bgcolor='white'
    )
    
    return fig




st.sidebar.markdown("""
### What is Adjusted Probability?

Each point shows the **probability that at least one event** (e.g., enemy action or explosive hazard) occurred during a week in the selected locality.

- **≥ 0.5:** Indicates it is _at least as likely as not_ that exposure occurred that week (meets VA threshold).
- **< 0.5:** Less likely, but may still support claims if corroborated by other evidence.

Probabilities are derived from weekly rolling counts of events, smoothed to account for reporting gaps.
""")

# Create two columns with equal width for the metrics
col1, col2 = st.columns(2)

# Place each metric in its own column
with col1:
    st.metric("Weeks with ≥ 50% Enemy Action", f"{(filtered['enemy_action_adj'] >= 0.5).sum()} weeks")

with col2:
    st.metric("Weeks with ≥ 50% Explosive Hazards", f"{(filtered['explosive_hazard_adj'] >= 0.5).sum()} weeks")

# Add Phase 3 enhancements
st.markdown("---")
st.markdown("### 📊 Enhanced Analysis Options")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📈 Single Locality", "🔄 Compare Localities", "📥 Export Data"])

with tab1:
    st.markdown("**Interactive Probability Chart**")
    st.markdown("*Hover over points for details, zoom and pan to explore*")
    
    # Display the interactive chart
    st.plotly_chart(interactive_fig, use_container_width=True)

with tab2:
    st.markdown("**Compare Multiple Localities**")
    
    # Multi-select for comparison
    comparison_localities = st.multiselect(
        "Select localities to compare (up to 5):",
        options=sorted(locality_list),
        default=[selected_locality],
        max_selections=5
    )
    
    if len(comparison_localities) > 1:
        # Get data for all selected localities with progress indication
        comparison_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, loc in enumerate(comparison_localities):
            status_text.text(f"Loading data for {loc}...")
            progress_bar.progress((i + 1) / len(comparison_localities))
            
            loc_data = loaded[loaded['locality'] == loc].copy()
            # Convert year_week to datetime for filtering
            loc_data['week_start'] = pd.to_datetime(loc_data['year_week'] + '-0', format='%Y-%U-%w')
            loc_data = loc_data[(loc_data['week_start'] >= start_date) & (loc_data['week_start'] <= end_date)]
            comparison_data.append(loc_data)
        
        status_text.text("Creating comparison chart...")
        
        # Create comparison chart
        comparison_fig = create_comparison_chart(comparison_data, comparison_localities)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Comparison statistics
        st.markdown("**Comparison Statistics:**")
        comp_stats = []
        for loc in comparison_localities:
            loc_data = loaded[loaded['locality'] == loc].copy()
            # Convert year_week to datetime for filtering
            loc_data['week_start'] = pd.to_datetime(loc_data['year_week'] + '-0', format='%Y-%U-%w')
            loc_data = loc_data[(loc_data['week_start'] >= start_date) & (loc_data['week_start'] <= end_date)]
            if not loc_data.empty:
                enemy_weeks = (loc_data['enemy_action_adj'] >= 0.5).sum()
                explosive_weeks = (loc_data['explosive_hazard_adj'] >= 0.5).sum()
                total_weeks = len(loc_data)
                comp_stats.append({
                    'Locality': loc,
                    'Enemy Action ≥50%': f"{enemy_weeks} ({enemy_weeks/total_weeks*100:.1f}%)" if total_weeks > 0 else "0 (0%)",
                    'Explosive Hazards ≥50%': f"{explosive_weeks} ({explosive_weeks/total_weeks*100:.1f}%)" if total_weeks > 0 else "0 (0%)",
                    'Total Weeks': total_weeks
                })
        
        if comp_stats:
            st.table(pd.DataFrame(comp_stats))
    else:
        st.info("Select at least 2 localities to compare")

with tab3:
    st.markdown("**Export Data and Charts**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Export Current Data**")
        
        # Export filtered data
        export_filename = f"{selected_locality}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        export_link = export_data_to_csv(filtered, export_filename)
        st.markdown(export_link, unsafe_allow_html=True)
        
        # Export summary statistics
        summary_data = {
            'Locality': [selected_locality],
            'Date_Range': [f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"],
            'Total_Weeks': [len(filtered)],
            'Enemy_Action_High_Weeks': [(filtered['enemy_action_adj'] >= 0.5).sum()],
            'Explosive_Hazard_High_Weeks': [(filtered['explosive_hazard_adj'] >= 0.5).sum()],
            'Avg_Enemy_Probability': [filtered['enemy_action_adj'].mean()],
            'Avg_Explosive_Probability': [filtered['explosive_hazard_adj'].mean()]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_export_link = export_data_to_csv(summary_df, f"{export_filename}_summary")
        st.markdown(f"**📊 Summary Statistics:** {summary_export_link}", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🖼️ Export Charts**")
        st.markdown("*Use the camera icon in the top-right corner of any chart to save as PNG*")
        
        # Chart export instructions
        st.info("""
        **Chart Export Tips:**
        - Hover over any chart and click the camera icon 📷
        - Charts are saved as high-resolution PNG files
        - Interactive features are preserved when sharing links
        """)


# Add visual improvements
st.markdown("---")

# add button for ollama
st.markdown("<br>", unsafe_allow_html=True)

# Add instant analysis section with enhanced features
st.subheader("📊 Combat Exposure Analysis")

# Get pre-computed analysis for the selected locality
if selected_locality in analysis_data:
    locality_analysis = analysis_data[selected_locality]
    
    # Display the pre-computed summary
    st.markdown(locality_analysis['summary_text'])
    
    
else:
    st.warning(f"No pre-computed analysis available for {selected_locality}. Please check the locality name.")

# Add GDELT News Events Analysis Section
st.markdown("---")
st.markdown("## 📰 GDELT News Events Analysis")

with st.container():
    # Get the current date range for GDELT analysis using the date slider
    start_week_str = f"{start_date.year}-{start_date.isocalendar()[1]:02d}"
    end_week_str = f"{end_date.year}-{end_date.isocalendar()[1]:02d}"
    date_range_title = f"{start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}"
    
    # Analyze GDELT events for the selected locality and timeframe
    with st.spinner("🔄 Analyzing GDELT events..."):
        # Use the original string-based GDELT data for analysis
        original_gdelt = pd.read_parquet("data/iraq_gdelt_2003_2011_ollama.parquet")
        gdelt_events = analyze_gdelt_for_locality_and_timeframe(
            original_gdelt, selected_locality, start_week_str, end_week_str
        )
    
    if gdelt_events:
        # Get top 10 event types
        top_events = get_top_event_types(gdelt_events, 10)
        
        # Create visualization
        chart_title = f"Top 10 GDELT Event Types\n{selected_locality} - {date_range_title}"
        gdelt_fig = create_gdelt_visualization(top_events, chart_title)
        
        if gdelt_fig:
            st.pyplot(gdelt_fig)
            
            # Add summary statistics
            total_events = len(gdelt_events)
            unique_event_types = len(set(event['event_type'] for event in gdelt_events))
            weeks_with_events = len(set(event['week'] for event in gdelt_events))
            
            st.markdown(f"""
            **Summary Statistics:**
            - **Total Events:** {total_events:,}
            - **Unique Event Types:** {unique_event_types}
            - **Weeks with Events:** {weeks_with_events}
            """)
    else:
        st.info(f"No GDELT events found for {selected_locality} during {date_range_title}")
        st.markdown("This could mean:")
        st.markdown("- The locality name doesn't match news coverage")
        st.markdown("- Limited news coverage during this period")
        st.markdown("- Try selecting a different time period or locality")


#citation
# Add a separation line
st.markdown("---")

# Add data source and citation information
# Add enhanced data summary
st.markdown("---")
st.markdown("### 📊 Enhanced Data Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Localities", len(analysis_data))
    st.metric("GDELT Events Processed", f"{enhanced_data['gdelt_summary']['total_events']:,}")

with col2:
    st.metric("Cached Time Periods", len(enhanced_data['date_range_cache']))
    st.metric("Weeks of Data", enhanced_data['gdelt_summary']['weeks_covered'])

with col3:
    st.metric("Localities with GDELT", enhanced_data['gdelt_summary']['localities_mentioned'])
    st.metric("Combat Events Database", "43,981 records")

st.markdown("""
### Data Sources and Citations

**DATA ACCESS:**  
[Harvard Dataverse - Iraq War Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JRYGFP)

**CITATIONS:**  
1. Shaver, Andrew, and Alexander K. Bollfrass. "Disorganized Political Violence: A Demonstration Case of Temperature and Insurgency." *International Organization* (2023).
2. U.S. Central Command. FOIA Release 14-0091(2014).
3. GDELT Project. "Global Database of Events, Language, and Tone." [https://www.gdeltproject.org/](https://www.gdeltproject.org/)

**DISCLAIMER:**  
This tool is for informational purposes only and does not constitute legal or medical advice. Consult with a qualified professional for specific claims or health concerns.
""")

