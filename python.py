import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# PART 1: DATA INPUT
# ============================================================================

def create_sample_data():
    """
    Creates sample dataset for the four states.
    Replace this with actual census data from NPC/NBS/UN sources.

    Data structure based on:
    - Nigerian Census and NBS projections
    - UN World Urbanization Prospects
    - State statistical yearbooks

    Years: 2000, 2005, 2010, 2015, 2020
    """

    # Years of analysis
    years = [2000, 2005, 2010, 2015, 2020]

    # LAGOS STATE DATA
    lagos_data = {
        'Year': years,
        'Total_Population': [8029200, 8947000, 10788000, 12550598, 14368332],
        'Urban_Population': [6422560, 7426150, 9170880, 11038980, 12966892],
        'Rural_Population': [1606640, 1520850, 1617120, 1511618, 1401440],
        'Major_City': 'Lagos Metropolis',
        'State': 'Lagos'
    }

    # OYO STATE DATA (Ibadan as major city)
    oyo_data = {
        'Year': years,
        'Total_Population': [4472200, 5135000, 6311200, 7069479, 7840864],
        'Urban_Population': [2010990, 2464650, 3029376, 3534740, 4154449],
        'Rural_Population': [2461210, 2670350, 3281824, 3534739, 3686415],
        'Major_City': 'Ibadan',
        'State': 'Oyo'
    }

    # RIVERS STATE DATA (Port Harcourt as major city)
    rivers_data = {
        'Year': years,
        'Total_Population': [4100500, 4835000, 5839500, 6559906, 7303700],
        'Urban_Population': [2337365, 2998850, 3736700, 4460633, 5186664],
        'Rural_Population': [1763135, 1836150, 2102800, 2099273, 2117036],
        'Major_City': 'Port Harcourt',
        'State': 'Rivers'
    }

    # FCT ABUJA DATA
    fct_data = {
        'Year': years,
        'Total_Population': [830000, 1215000, 2245000, 3277740, 4370000],
        'Urban_Population': [747000, 1094550, 2130950, 3113853, 4197300],
        'Rural_Population': [83000, 120450, 114050, 163887, 172700],
        'Major_City': 'Abuja',
        'State': 'FCT'
    }

    # Create DataFrames
    lagos_df = pd.DataFrame(lagos_data)
    oyo_df = pd.DataFrame(oyo_data)
    rivers_df = pd.DataFrame(rivers_data)
    fct_df = pd.DataFrame(fct_data)

    # Combine all states
    all_states = pd.concat([lagos_df, oyo_df, rivers_df, fct_df], ignore_index=True)

    return all_states, lagos_df, oyo_df, rivers_df, fct_df


def calculate_urbanization_metrics(df):
    """Calculate key urbanization indicators"""

    df = df.copy()

    # Urbanization Rate (%)
    df['Urbanization_Rate'] = (df['Urban_Population'] / df['Total_Population']) * 100

    # Rural Population Rate (%)
    df['Rural_Rate'] = (df['Rural_Population'] / df['Total_Population']) * 100

    # Population Growth Rate (%)
    df['Pop_Growth_Rate'] = df.groupby('State')['Total_Population'].pct_change() * 100

    # Urban Growth Rate (%)
    df['Urban_Growth_Rate'] = df.groupby('State')['Urban_Population'].pct_change() * 100

    # Rural Growth Rate (%)
    df['Rural_Growth_Rate'] = df.groupby('State')['Rural_Population'].pct_change() * 100

    return df


# ============================================================================
# PART 2: DESCRIPTIVE STATISTICS
# ============================================================================

def generate_summary_statistics(df):
    """Generate summary statistics table"""

    print("=" * 80)
    print("SUMMARY STATISTICS: URBANIZATION IN FOUR NIGERIAN STATES")
    print("=" * 80)
    print()

    # Latest year statistics
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]

    summary = latest_data.groupby('State').agg({
        'Total_Population': 'sum',
        'Urban_Population': 'sum',
        'Rural_Population': 'sum',
        'Urbanization_Rate': 'mean'
    }).round(2)

    summary['Urban_Population_Million'] = (summary['Urban_Population'] / 1_000_000).round(2)
    summary['Total_Population_Million'] = (summary['Total_Population'] / 1_000_000).round(2)

    print(f"\nData for Year: {latest_year}")
    print(summary[['Total_Population_Million', 'Urban_Population_Million', 'Urbanization_Rate']])
    print()

    # Calculate average annual growth rates
    print("\nAVERAGE ANNUAL GROWTH RATES (2000-2020)")
    print("-" * 80)

    for state in df['State'].unique():
        state_data = df[df['State'] == state].sort_values('Year')

        # Calculate CAGR (Compound Annual Growth Rate)
        first_pop = state_data.iloc[0]['Total_Population']
        last_pop = state_data.iloc[-1]['Total_Population']
        years_diff = state_data.iloc[-1]['Year'] - state_data.iloc[0]['Year']

        cagr = (((last_pop / first_pop) ** (1/years_diff)) - 1) * 100

        # Urban CAGR
        first_urban = state_data.iloc[0]['Urban_Population']
        last_urban = state_data.iloc[-1]['Urban_Population']
        urban_cagr = (((last_urban / first_urban) ** (1/years_diff)) - 1) * 100

        print(f"{state:15} - Total Pop CAGR: {cagr:5.2f}% | Urban Pop CAGR: {urban_cagr:5.2f}%")

    print()
    print("=" * 80)
    print()

    return summary


# ============================================================================
# PART 3: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_1_total_vs_urban_population(df):
    """Chart 1: Total vs Urban Population Over Time"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    states = df['State'].unique()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for idx, state in enumerate(states):
        state_data = df[df['State'] == state].sort_values('Year')

        ax = axes[idx]

        # Plot total and urban population
        ax.plot(state_data['Year'], state_data['Total_Population']/1_000_000,
                marker='o', linewidth=2.5, markersize=8, label='Total Population',
                color=colors[idx], alpha=0.7)

        ax.plot(state_data['Year'], state_data['Urban_Population']/1_000_000,
                marker='s', linewidth=2.5, markersize=8, label='Urban Population',
                color=colors[idx], linestyle='--')

        ax.fill_between(state_data['Year'],
                        state_data['Urban_Population']/1_000_000,
                        alpha=0.2, color=colors[idx])

        ax.set_title(f'{state} State Population Trends (2000-2020)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Population (Millions)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020])

        # Add latest year annotation
        latest = state_data.iloc[-1]
        ax.annotate(f"{latest['Total_Population']/1_000_000:.2f}M",
                   xy=(latest['Year'], latest['Total_Population']/1_000_000),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color=colors[idx], fontweight='bold')

    plt.tight_layout()
    plt.savefig('1_total_vs_urban_population.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 1 saved: 1_total_vs_urban_population.png")
    plt.show()


def plot_2_urbanization_rate_trends(df):
    """Chart 2: Urbanization Rate Trends"""

    fig, ax = plt.subplots(figsize=(14, 8))

    states = df['State'].unique()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    markers = ['o', 's', '^', 'D']

    for idx, state in enumerate(states):
        state_data = df[df['State'] == state].sort_values('Year')

        ax.plot(state_data['Year'], state_data['Urbanization_Rate'],
                marker=markers[idx], linewidth=3, markersize=10,
                label=state, color=colors[idx], alpha=0.8)

    ax.set_title('Urbanization Rate Trends Across Four Nigerian States (2000-2020)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Urbanization Rate (%)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020])

    # Add horizontal reference line at 50%
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.text(2001, 52, '50% Urbanization Threshold', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig('2_urbanization_rate_trends.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 2 saved: 2_urbanization_rate_trends.png")
    plt.show()


def plot_3_urban_rural_comparison(df):
    """Chart 3: Urban vs Rural Population Comparison (Latest Year)"""

    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].sort_values('Total_Population', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(latest_data['State']))
    width = 0.35

    bars1 = ax.bar(x - width/2, latest_data['Urban_Population']/1_000_000,
                   width, label='Urban Population', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, latest_data['Rural_Population']/1_000_000,
                   width, label='Rural Population', color='#2ecc71', alpha=0.8)

    ax.set_title(f'Urban vs Rural Population Distribution ({latest_year})',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('State', fontsize=14, fontweight='bold')
    ax.set_ylabel('Population (Millions)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(latest_data['State'], fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}M',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('3_urban_rural_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 3 saved: 3_urban_rural_comparison.png")
    plt.show()


def plot_4_urbanization_rate_comparison(df):
    """Chart 4: Urbanization Rate Comparison (Bar Chart - Latest Year)"""

    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].sort_values('Urbanization_Rate', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors_map = {'Lagos': '#e74c3c', 'Oyo': '#3498db',
                  'Rivers': '#2ecc71', 'FCT': '#f39c12'}
    colors = [colors_map[state] for state in latest_data['State']]

    bars = ax.barh(latest_data['State'], latest_data['Urbanization_Rate'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_title(f'Urbanization Rate Comparison ({latest_year})',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Urbanization Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('State', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for idx, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
               f'{width:.1f}%',
               ha='left', va='center', fontsize=12, fontweight='bold')

    # Add reference line
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=2)

    plt.tight_layout()
    plt.savefig('4_urbanization_rate_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 4 saved: 4_urbanization_rate_comparison.png")
    plt.show()


def plot_5_population_growth_rates(df):
    """Chart 5: Annual Population Growth Rates"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    states = df['State'].unique()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # Plot 1: Total Population Growth Rate
    ax1 = axes[0]
    for idx, state in enumerate(states):
        state_data = df[df['State'] == state].sort_values('Year')
        # Remove NaN values
        plot_data = state_data[state_data['Pop_Growth_Rate'].notna()]

        ax1.plot(plot_data['Year'], plot_data['Pop_Growth_Rate'],
                marker='o', linewidth=2, markersize=6,
                label=state, color=colors[idx], alpha=0.8)

    ax1.set_title('Total Population Growth Rate', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Annual Growth Rate (%)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.set_xticks([2000, 2005, 2010, 2015, 2020])

    # Plot 2: Urban Population Growth Rate
    ax2 = axes[1]
    for idx, state in enumerate(states):
        state_data = df[df['State'] == state].sort_values('Year')
        plot_data = state_data[state_data['Urban_Growth_Rate'].notna()]

        ax2.plot(plot_data['Year'], plot_data['Urban_Growth_Rate'],
                marker='s', linewidth=2, markersize=6,
                label=state, color=colors[idx], alpha=0.8)

    ax2.set_title('Urban Population Growth Rate', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Growth Rate (%)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_xticks([2000, 2005, 2010, 2015, 2020])

    plt.tight_layout()
    plt.savefig('5_population_growth_rates.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 5 saved: 5_population_growth_rates.png")
    plt.show()


def plot_6_stacked_area_chart(df):
    """Chart 6: Stacked Area Chart - Urban and Rural Composition"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    states = df['State'].unique()

    for idx, state in enumerate(states):
        state_data = df[df['State'] == state].sort_values('Year')

        ax = axes[idx]

        ax.fill_between(state_data['Year'], 0,
                       state_data['Urban_Population']/1_000_000,
                       label='Urban', color='#3498db', alpha=0.7)

        ax.fill_between(state_data['Year'],
                       state_data['Urban_Population']/1_000_000,
                       state_data['Total_Population']/1_000_000,
                       label='Rural', color='#2ecc71', alpha=0.7)

        ax.set_title(f'{state} State: Urban-Rural Composition (2000-2020)',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Population (Millions)', fontsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020])

    plt.tight_layout()
    plt.savefig('6_stacked_area_composition.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 6 saved: 6_stacked_area_composition.png")
    plt.show()


def plot_7_comparative_heatmap(df):
    """Chart 7: Heatmap of Urbanization Rates Over Time"""

    # Pivot data for heatmap
    pivot_data = df.pivot(index='State', columns='Year', values='Urbanization_Rate')

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Urbanization Rate (%)'},
                linewidths=0.5, linecolor='white', ax=ax)

    ax.set_title('Urbanization Rate Heatmap: Four Nigerian States (2000-2020)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('State', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('7_urbanization_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 7 saved: 7_urbanization_heatmap.png")
    plt.show()


def plot_8_population_pyramid_comparison(df):
    """Chart 8: Population Distribution Comparison (Latest Year)"""

    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].sort_values('Total_Population', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(latest_data['State']))

    # Create horizontal bars
    ax.barh(y_pos, latest_data['Urban_Population']/1_000_000,
           height=0.4, label='Urban', color='#3498db', alpha=0.8)
    ax.barh(y_pos, -latest_data['Rural_Population']/1_000_000,
           height=0.4, label='Rural', color='#2ecc71', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(latest_data['State'])
    ax.set_xlabel('Population (Millions)', fontsize=12, fontweight='bold')
    ax.set_title(f'Urban-Rural Population Distribution ({latest_year})',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=1.5)

    # Format x-axis to show absolute values
    xticks = ax.get_xticks()
    ax.set_xticklabels([f'{abs(x):.1f}' for x in xticks])

    plt.tight_layout()
    plt.savefig('8_population_pyramid_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 8 saved: 8_population_pyramid_comparison.png")
    plt.show()


def plot_9_urbanization_change(df):
    """Chart 9: Change in Urbanization Rate (2000 vs 2020)"""

    first_year = df['Year'].min()
    last_year = df['Year'].max()

    first_data = df[df['Year'] == first_year][['State', 'Urbanization_Rate']].rename(
        columns={'Urbanization_Rate': 'Rate_2000'})
    last_data = df[df['Year'] == last_year][['State', 'Urbanization_Rate']].rename(
        columns={'Urbanization_Rate': 'Rate_2020'})

    comparison = pd.merge(first_data, last_data, on='State')
    comparison['Change'] = comparison['Rate_2020'] - comparison['Rate_2000']
    comparison = comparison.sort_values('Change', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#e74c3c' if x > 0 else '#3498db' for x in comparison['Change']]

    bars = ax.barh(comparison['State'], comparison['Change'], color=colors, alpha=0.8)

    ax.set_title(f'Change in Urbanization Rate ({first_year} to {last_year})',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Change in Urbanization Rate (Percentage Points)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('State', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for idx, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + (0.5 if width > 0 else -0.5), bar.get_y() + bar.get_height()/2,
               f'{width:+.1f}pp',
               ha='left' if width > 0 else 'right', va='center',
               fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('9_urbanization_change.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 9 saved: 9_urbanization_change.png")
    plt.show()


def plot_10_ranking_over_time(df):
    """Chart 10: State Rankings by Urban Population Over Time"""

    fig, ax = plt.subplots(figsize=(14, 8))

    states = df['State'].unique()
    colors = {'Lagos': '#e74c3c', 'Oyo': '#3498db',
              'Rivers': '#2ecc71', 'FCT': '#f39c12'}

    for state in states:
        state_data = df[df['State'] == state].sort_values('Year')

        # Calculate ranking for each year
        rankings = []
        for year in state_data['Year']:
            year_data = df[df['Year'] == year].sort_values('Urban_Population', ascending=False)
            rank = list(year_data['State']).index(state) + 1
            rankings.append(rank)

        ax.plot(state_data['Year'], rankings, marker='o', linewidth=3,
               markersize=10, label=state, color=colors[state], alpha=0.8)

    ax.set_title('Urban Population Ranking Over Time',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank (1 = Highest Urban Population)', fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(4.5, 0.5)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020])
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('10_ranking_over_time.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 10 saved: 10_ranking_over_time.png")
    plt.show()


# ============================================================================
# PART 4: CORRELATION AND STATISTICAL ANALYSIS
# ============================================================================

def statistical_analysis(df):
    """Perform statistical tests and correlations"""

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Test if urbanization rates differ significantly across states (ANOVA)
    states_data = [df[df['State'] == state]['Urbanization_Rate'].dropna()
                   for state in df['State'].unique()]

    f_stat, p_value = stats.f_oneway(*states_data)

    print(f"\nANOVA Test: Do urbanization rates differ significantly across states?")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: YES - Significant differences exist (p < 0.05)")
    else:
        print("Result: NO - No significant differences (p >= 0.05)")

    # Correlation between total population and urbanization rate
    print(f"\n" + "-" * 80)
    print("CORRELATION ANALYSIS")
    print("-" * 80)

    corr_pop_urban = df[['Total_Population', 'Urbanization_Rate']].corr().iloc[0, 1]
    print(f"Correlation (Total Population vs Urbanization Rate): {corr_pop_urban:.4f}")

    # State-specific growth rates
    print(f"\n" + "-" * 80)
    print("AVERAGE URBANIZATION GROWTH (Percentage Points per Decade)")
    print("-" * 80)

    for state in df['State'].unique():
        state_data = df[df['State'] == state].sort_values('Year')
        first_rate = state_data.iloc[0]['Urbanization_Rate']
        last_rate = state_data.iloc[-1]['Urbanization_Rate']
        years_diff = (state_data.iloc[-1]['Year'] - state_data.iloc[0]['Year']) / 10

        decade_growth = (last_rate - first_rate) / years_diff

        print(f"{state:15} - {decade_growth:+.2f} percentage points per decade")

    print("\n" + "=" * 80 + "\n")


# ============================================================================
# PART 5: DATA EXPORT FUNCTIONS
# ============================================================================

def export_data_to_csv(df):
    """Export processed data to CSV files"""

    # Export complete dataset
    df.to_csv('urbanization_data_complete.csv', index=False)
    print("✓ Complete data exported: urbanization_data_complete.csv")

    # Export summary by state
    summary = df.groupby('State').agg({
        'Total_Population': ['min', 'max', 'mean'],
        'Urban_Population': ['min', 'max', 'mean'],
        'Urbanization_Rate': ['min', 'max', 'mean']
    }).round(2)

    summary.to_csv('urbanization_summary_by_state.csv')
    print("✓ Summary by state exported: urbanization_summary_by_state.csv")

    # Export latest year data
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    latest_data.to_csv(f'urbanization_data_{latest_year}.csv', index=False)
    print(f"✓ Latest year data exported: urbanization_data_{latest_year}.csv")


def create_data_dictionary():
    """Create data dictionary explaining all variables"""

    data_dict = {
        'Variable': ['Year', 'State', 'Major_City', 'Total_Population',
                     'Urban_Population', 'Rural_Population', 'Urbanization_Rate',
                     'Rural_Rate', 'Pop_Growth_Rate', 'Urban_Growth_Rate',
                     'Rural_Growth_Rate'],
        'Description': [
            'Year of observation (2000, 2005, 2010, 2015, 2020)',
            'Nigerian state (Lagos, Oyo, Rivers, FCT)',
            'Major city in the state',
            'Total population in the state',
            'Population living in urban areas',
            'Population living in rural areas',
            'Percentage of population living in urban areas',
            'Percentage of population living in rural areas',
            'Annual percentage change in total population',
            'Annual percentage change in urban population',
            'Annual percentage change in rural population'
        ],
        'Unit': ['Year', 'Text', 'Text', 'Number', 'Number', 'Number',
                 'Percentage', 'Percentage', 'Percentage', 'Percentage', 'Percentage'],
        'Source': ['Census/Projection', 'Administrative', 'Administrative',
                   'NPC/NBS/UN', 'NPC/NBS/UN', 'Calculated', 'Calculated',
                   'Calculated', 'Calculated', 'Calculated', 'Calculated']
    }

    dict_df = pd.DataFrame(data_dict)
    dict_df.to_csv('data_dictionary.csv', index=False)
    print("✓ Data dictionary exported: data_dictionary.csv")

    return dict_df


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function to run the complete analysis
    """

    print("\n" + "=" * 80)
    print("URBANIZATION TRAJECTORIES IN NIGERIA - ANALYSIS STARTING")
    print("=" * 80 + "\n")

    # Step 1: Create/Load Data
    print("Step 1: Loading data...")
    all_data, lagos, oyo, rivers, fct = create_sample_data()
    print(f"✓ Data loaded successfully - {len(all_data)} records")

    # Step 2: Calculate Metrics
    print("\nStep 2: Calculating urbanization metrics...")
    all_data = calculate_urbanization_metrics(all_data)
    print("✓ Metrics calculated")

    # Step 3: Generate Summary Statistics
    print("\nStep 3: Generating summary statistics...")
    summary = generate_summary_statistics(all_data)

    # Step 4: Export Data
    print("\nStep 4: Exporting data files...")
    export_data_to_csv(all_data)
    data_dict = create_data_dictionary()

    # Step 5: Generate Visualizations
    print("\nStep 5: Generating visualizations...")
    print("(This may take a minute...)")

    plot_1_total_vs_urban_population(all_data)
    plot_2_urbanization_rate_trends(all_data)
    plot_3_urban_rural_comparison(all_data)
    plot_4_urbanization_rate_comparison(all_data)
    plot_5_population_growth_rates(all_data)
    plot_6_stacked_area_chart(all_data)
    plot_7_comparative_heatmap(all_data)
    plot_8_population_pyramid_comparison(all_data)
    plot_9_urbanization_change(all_data)
    plot_10_ranking_over_time(all_data)

    # Step 6: Statistical Analysis
    print("\nStep 6: Performing statistical analysis...")
    statistical_analysis(all_data)

    # Final message
    print("\n" + "=" * 80)
    print("✓✓✓ ANALYSIS COMPLETE! ✓✓✓")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  - 10 PNG chart files (1_*.png through 10_*.png)")
    print("  - 3 CSV data files")
    print("  - 1 Data dictionary")
    print("\nAll files saved in current directory.")
    print("=" * 80 + "\n")

    return all_data, summary


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Execute the complete analysis
    results, summary = main()
