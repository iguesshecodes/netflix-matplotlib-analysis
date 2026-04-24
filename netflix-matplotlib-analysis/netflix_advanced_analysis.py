import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#141414',
    'axes.facecolor':   '#1f1f1f',
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#e5e5e5',
    'xtick.color':      '#a0a0a0',
    'ytick.color':      '#a0a0a0',
    'text.color':       '#e5e5e5',
    'grid.color':       '#2a2a2a',
    'grid.linestyle':   '--',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

NETFLIX_RED   = '#E50914'
ACCENT_GOLD   = '#F5C518'
ACCENT_BLUE   = '#00B4D8'
ACCENT_GREEN  = '#06D6A0'
ACCENT_PURPLE = '#9B5DE5'
LIGHT_GRAY    = '#a0a0a0'

# ── Load & Clean ──────────────────────────────────────────────────────────────
df_raw = pd.read_csv('/mnt/user-data/uploads/netflix_titles.csv')
df = df_raw.copy()

# Parse date_added
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), format='%B %d, %Y', errors='coerce')
df['month_added'] = df['date_added'].dt.month
df['year_added']  = df['date_added'].dt.year

# Duration
df['duration_clean'] = df['duration'].str.extract(r'(\d+)').astype(float)

# Fix bad ratings (e.g. "74 min")
valid_ratings = ['TV-MA','TV-14','TV-PG','R','PG-13','TV-Y7','TV-Y','PG','TV-G','NR','G','TV-Y7-FV','NC-17','UR']
df['rating'] = df['rating'].where(df['rating'].isin(valid_ratings), other=np.nan)

# Explode genres (comma-separated)
df_genres = df.copy()
df_genres['genre'] = df_genres['listed_in'].str.split(', ')
df_genres = df_genres.explode('genre')

# Explode countries
df_countries = df.copy()
df_countries['country_single'] = df_countries['country'].str.split(', ')
df_countries = df_countries.explode('country_single')
df_countries = df_countries.dropna(subset=['country_single'])

movies = df[(df['type']=='Movie') & df['duration_clean'].notna()].copy()
shows  = df[(df['type']=='TV Show') & df['duration_clean'].notna()].copy()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Content Addition Heatmap (Month × Year)
# ══════════════════════════════════════════════════════════════════════════════
df_heat = df.dropna(subset=['year_added','month_added'])
pivot = df_heat.groupby(['year_added','month_added']).size().unstack(fill_value=0)
pivot = pivot[(pivot.index >= 2015) & (pivot.index <= 2021)]
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot.columns)]

cmap = LinearSegmentedColormap.from_list('netflix_heat', ['#1f1f1f','#8B0000', NETFLIX_RED, ACCENT_GOLD])

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor('#141414')
ax.set_facecolor('#141414')

sns.heatmap(pivot, cmap=cmap, linewidths=0.4, linecolor='#141414',
            annot=True, fmt='d', annot_kws={'size':9, 'color':'white'},
            ax=ax, cbar_kws={'label':'Titles Added'})

ax.set_title('🗓  Content Added to Netflix — Monthly Heatmap (2015–2021)',
             fontsize=14, fontweight='bold', color='white', pad=15)
ax.set_xlabel('Month', fontsize=11, color=LIGHT_GRAY)
ax.set_ylabel('Year', fontsize=11, color=LIGHT_GRAY)
ax.tick_params(colors='white')
plt.setp(ax.get_xticklabels(), color='white', fontsize=10)
plt.setp(ax.get_yticklabels(), color='white', fontsize=10, rotation=0)
ax.collections[0].colorbar.ax.tick_params(colors='white')
ax.collections[0].colorbar.ax.yaxis.label.set_color('white')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/1_content_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#141414')
plt.close()
print("✅ Chart 1 saved")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Top Genres: Movies vs TV Shows (Grouped Bar)
# ══════════════════════════════════════════════════════════════════════════════
top_genres_movies = (df_genres[df_genres['type']=='Movie']['genre']
                     .value_counts().head(10).index)
top_genres_shows  = (df_genres[df_genres['type']=='TV Show']['genre']
                     .value_counts().head(10).index)
top_genres = list(set(top_genres_movies) | set(top_genres_shows))[:12]

genre_movie = df_genres[df_genres['type']=='Movie']['genre'].value_counts()
genre_show  = df_genres[df_genres['type']=='TV Show']['genre'].value_counts()

genre_df = pd.DataFrame({
    'Movies':   genre_movie,
    'TV Shows': genre_show
}).fillna(0).loc[top_genres].sort_values('Movies', ascending=False).head(12)

x = np.arange(len(genre_df))
w = 0.38

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor('#141414')
ax.set_facecolor('#1f1f1f')

bars1 = ax.bar(x - w/2, genre_df['Movies'],   width=w, color=NETFLIX_RED,  label='Movies',   alpha=0.9, zorder=3)
bars2 = ax.bar(x + w/2, genre_df['TV Shows'], width=w, color=ACCENT_BLUE,  label='TV Shows', alpha=0.9, zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8, color=NETFLIX_RED)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8, color=ACCENT_BLUE)

ax.set_xticks(x)
ax.set_xticklabels(genre_df.index, rotation=30, ha='right', fontsize=10)
ax.set_title('🎭  Top Genres — Movies vs TV Shows', fontsize=14, fontweight='bold', color='white', pad=15)
ax.set_ylabel('Number of Titles', color=LIGHT_GRAY)
ax.grid(axis='y', zorder=0)
ax.legend(fontsize=11)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/2_genres_grouped_bar.png', dpi=150, bbox_inches='tight', facecolor='#141414')
plt.close()
print("✅ Chart 2 saved")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Movie Duration Box Plot by Rating Group
# ══════════════════════════════════════════════════════════════════════════════
rating_order = ['G','PG','PG-13','R','NC-17','TV-Y','TV-Y7','TV-G','TV-PG','TV-14','TV-MA']
movies_rated = movies[movies['rating'].isin(rating_order)].copy()

rating_groups = {
    'Kids\n(G / TV-Y)':       ['G','TV-Y'],
    'Family\n(PG / TV-Y7)':   ['PG','TV-Y7'],
    'Teen\n(PG-13 / TV-PG)':  ['PG-13','TV-PG'],
    'Young Adult\n(TV-14)':   ['TV-14'],
    'Adult\n(R / TV-MA)':     ['R','TV-MA'],
}
movies_rated['rating_group'] = movies_rated['rating'].map(
    {r: g for g, rs in rating_groups.items() for r in rs}
)
movies_rated = movies_rated.dropna(subset=['rating_group'])
group_order = list(rating_groups.keys())

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#141414')
ax.set_facecolor('#1f1f1f')

colors = [ACCENT_GREEN, ACCENT_BLUE, ACCENT_GOLD, ACCENT_PURPLE, NETFLIX_RED]
box_data = [movies_rated[movies_rated['rating_group']==g]['duration_clean'].dropna().values
            for g in group_order]

bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(color=LIGHT_GRAY),
                capprops=dict(color=LIGHT_GRAY),
                flierprops=dict(marker='o', color=LIGHT_GRAY, markersize=3, alpha=0.4))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

medians = [np.median(d) for d in box_data if len(d) > 0]
for i, med in enumerate(medians):
    ax.text(i+1, med + 3, f'{int(med)} min', ha='center', va='bottom',
            fontsize=9, color='white', fontweight='bold')

ax.set_xticklabels(group_order, fontsize=10)
ax.set_title('🎬  Movie Duration Distribution by Audience Rating Group',
             fontsize=14, fontweight='bold', color='white', pad=15)
ax.set_ylabel('Duration (minutes)', color=LIGHT_GRAY)
ax.set_xlabel('Rating Group', color=LIGHT_GRAY)
ax.grid(axis='y', zorder=0)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/3_duration_boxplot_by_rating.png', dpi=150, bbox_inches='tight', facecolor='#141414')
plt.close()
print("✅ Chart 3 saved")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Country Content Mix: Stacked Horizontal Bar
# ══════════════════════════════════════════════════════════════════════════════
top10_countries = df_countries['country_single'].value_counts().head(10).index.tolist()
country_type = (df_countries[df_countries['country_single'].isin(top10_countries)]
                .groupby(['country_single','type']).size().unstack(fill_value=0))
country_type['Total'] = country_type.sum(axis=1)
country_type = country_type.sort_values('Total')
country_type = country_type.drop(columns='Total')

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#141414')
ax.set_facecolor('#1f1f1f')

bottoms = np.zeros(len(country_type))
bar_colors = {'Movie': NETFLIX_RED, 'TV Show': ACCENT_BLUE}
for col, color in bar_colors.items():
    if col in country_type.columns:
        vals = country_type[col].values
        bars = ax.barh(country_type.index, vals, left=bottoms,
                       color=color, label=col, alpha=0.9, height=0.6)
        for bar, val, b in zip(bars, vals, bottoms):
            if val > 30:
                ax.text(b + val/2, bar.get_y() + bar.get_height()/2,
                        f'{int(val)}', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
        bottoms += vals

totals = country_type.sum(axis=1).values
for i, (total, country) in enumerate(zip(totals, country_type.index)):
    ax.text(total + 15, i, f'n={int(total)}', va='center', fontsize=9, color=LIGHT_GRAY)

ax.set_title('🌍  Content Mix by Country — Movies vs TV Shows (Top 10)',
             fontsize=14, fontweight='bold', color='white', pad=15)
ax.set_xlabel('Number of Titles', color=LIGHT_GRAY)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='x', zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/4_country_content_mix.png', dpi=150, bbox_inches='tight', facecolor='#141414')
plt.close()
print("✅ Chart 4 saved")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Content Gap Analysis: Release Year vs Year Added
# ══════════════════════════════════════════════════════════════════════════════
df_gap = df.dropna(subset=['year_added','release_year']).copy()
df_gap['gap_years'] = df_gap['year_added'] - df_gap['release_year']
df_gap = df_gap[(df_gap['gap_years'] >= 0) & (df_gap['gap_years'] <= 40)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#141414')

for ax in axes:
    ax.set_facecolor('#1f1f1f')

for ax, content_type, color in zip(axes, ['Movie','TV Show'], [NETFLIX_RED, ACCENT_BLUE]):
    subset = df_gap[df_gap['type']==content_type]['gap_years']
    ax.hist(subset, bins=range(0, 42, 2), color=color, alpha=0.85,
            edgecolor='#141414', linewidth=0.5, zorder=3)
    median_gap = subset.median()
    ax.axvline(median_gap, color='white', linestyle='--', linewidth=1.5, zorder=4)
    ax.text(median_gap + 0.5, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 100,
            f'Median: {int(median_gap)}y', color='white', fontsize=10, fontweight='bold')
    ax.set_title(f'{content_type}s — Years Between Release & Netflix Addition',
                 fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Gap (years)', color=LIGHT_GRAY)
    ax.set_ylabel('Number of Titles', color=LIGHT_GRAY)
    ax.grid(axis='y', zorder=0)

# Recalculate ylim after plotting
for ax, content_type, color in zip(axes, ['Movie','TV Show'], [NETFLIX_RED, ACCENT_BLUE]):
    subset = df_gap[df_gap['type']==content_type]['gap_years']
    median_gap = subset.median()
    ymax = ax.get_ylim()[1]
    ax.texts[-1].set_position((median_gap + 0.5, ymax * 0.85))

fig.suptitle('⏳  Content Age Gap — How Old is Netflix Content When Added?',
             fontsize=14, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/5_content_age_gap.png', dpi=150, bbox_inches='tight', facecolor='#141414')
plt.close()
print("✅ Chart 5 saved")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Executive Dashboard (4-panel summary)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#141414')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel A: Content Growth Over Time (area chart) ───────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor('#1f1f1f')
yearly = df.groupby(['year_added','type']).size().unstack(fill_value=0)
yearly = yearly[(yearly.index >= 2008) & (yearly.index <= 2021)]
ax_a.fill_between(yearly.index, yearly.get('Movie',0), alpha=0.6, color=NETFLIX_RED, label='Movies')
ax_a.fill_between(yearly.index, yearly.get('TV Show',0), alpha=0.6, color=ACCENT_BLUE, label='TV Shows')
ax_a.set_title('Content Added Per Year', fontweight='bold', color='white')
ax_a.set_xlabel('Year', color=LIGHT_GRAY)
ax_a.set_ylabel('Titles Added', color=LIGHT_GRAY)
ax_a.legend(fontsize=9)
ax_a.grid()

# ── Panel B: Rating Distribution (horizontal bar) ────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor('#1f1f1f')
rating_counts = df['rating'].dropna().value_counts().head(10)
bar_colors_b = [NETFLIX_RED if i == 0 else '#555555' for i in range(len(rating_counts))]
bars_b = ax_b.barh(rating_counts.index[::-1], rating_counts.values[::-1],
                   color=bar_colors_b[::-1], alpha=0.9)
ax_b.set_title('Content Rating Distribution', fontweight='bold', color='white')
ax_b.set_xlabel('Number of Titles', color=LIGHT_GRAY)
ax_b.grid(axis='x')
for bar in bars_b:
    ax_b.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
              f'{int(bar.get_width())}', va='center', fontsize=8, color=LIGHT_GRAY)

# ── Panel C: Top 8 Genres (donut) ────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor('#141414')
top_g = df_genres['genre'].value_counts().head(8)
donut_colors = [NETFLIX_RED,'#c0392b','#922b21',ACCENT_BLUE,'#1a6b85',
                ACCENT_GOLD, ACCENT_GREEN, ACCENT_PURPLE]
wedges, texts, autotexts = ax_c.pie(
    top_g.values,
    labels=[g[:20] for g in top_g.index],
    colors=donut_colors,
    autopct='%1.0f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor='#141414', linewidth=2)
)
for text in texts:
    text.set_color('white')
    text.set_fontsize(8.5)
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(7.5)
ax_c.set_title('Top 8 Genres (All Content)', fontweight='bold', color='white')

# ── Panel D: KPI Summary Cards ───────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor('#141414')
ax_d.axis('off')

kpis = [
    ('Total Titles',      f"{len(df):,}",               NETFLIX_RED),
    ('Movies',            f"{len(df[df['type']=='Movie']):,}",   ACCENT_BLUE),
    ('TV Shows',          f"{len(df[df['type']=='TV Show']):,}", ACCENT_GREEN),
    ('Avg Movie Length',  f"{movies['duration_clean'].mean():.0f} min", ACCENT_GOLD),
    ('Countries',         f"{df_countries['country_single'].nunique():,}", ACCENT_PURPLE),
    ('Years of Content',  f"{int(df['release_year'].max() - df['release_year'].min())} yrs", LIGHT_GRAY),
]

ax_d.text(0.5, 1.05, 'Key Metrics at a Glance', ha='center', va='top',
          fontsize=13, fontweight='bold', color='white', transform=ax_d.transAxes)

for i, (label, value, color) in enumerate(kpis):
    row, col = divmod(i, 2)
    x, y = col * 0.52 + 0.04, 0.78 - row * 0.32
    ax_d.add_patch(mpatches.FancyBboxPatch((x, y-0.12), 0.44, 0.26,
                   boxstyle='round,pad=0.02', facecolor='#1f1f1f',
                   edgecolor=color, linewidth=1.5, transform=ax_d.transAxes, zorder=2))
    ax_d.text(x+0.22, y+0.08, value, ha='center', va='center',
              fontsize=16, fontweight='bold', color=color, transform=ax_d.transAxes, zorder=3)
    ax_d.text(x+0.22, y-0.06, label, ha='center', va='center',
              fontsize=9, color=LIGHT_GRAY, transform=ax_d.transAxes, zorder=3)

fig.suptitle('Netflix Content Library — Executive Dashboard',
             fontsize=17, fontweight='bold', color='white', y=1.01)

plt.savefig('/mnt/user-data/outputs/6_executive_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#141414')
plt.close()
print("✅ Chart 6 saved")

print("\n🎉 All 6 advanced charts generated successfully!")
