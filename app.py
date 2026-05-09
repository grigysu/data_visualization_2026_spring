import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kagglehub

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

from visual_helper import COUNTRY_ALIASES

# ── 0.  Load & pre-process data ───────────────────────────────────────────────

print("Fetching dataset via kagglehub …")
path = kagglehub.dataset_download("ammaraahmad/immigration-to-canada")
filepath = os.path.join(path, "canadian_immegration_data.csv")
df = pd.read_csv(filepath)
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Identify year columns (avoids pandas 3 deprecation warning on select_dtypes)
year_cols     = sorted([c for c in df.columns if str(c).isdigit()], key=int)
year_ints     = [int(y) for y in year_cols]
categorical_cols = [c for c in df.columns if c not in year_cols and c != "Total"]

# Ensure Total column exists
if "Total" not in df.columns:
    df["Total"] = df[year_cols].sum(axis=1)

# Continent-level palette
cont_totals = df.groupby("Continent")["Total"].sum().reset_index()
continents  = cont_totals["Continent"].tolist()
palette     = px.colors.qualitative.Set2
cont_color  = {c: palette[i % len(palette)] for i, c in enumerate(continents)}


def label(country: str) -> str:
    """Return the short display alias for a country, or the original name."""
    return COUNTRY_ALIASES.get(country, country)

def label_series(s: "pd.Series") -> "pd.Series":
    """Apply aliases to a pandas Series of country names."""
    return s.map(lambda x: COUNTRY_ALIASES.get(x, x))


# ── 2.  Helper / chart builders ───────────────────────────────────────────────

def top_correlated(country_name: str, n: int = 5) -> pd.DataFrame:
    target = df[df["Country"] == country_name][year_cols].values.flatten().astype(float)
    others = df[df["Country"] != country_name].copy()
    corrs  = others[year_cols].apply(
        lambda row: np.corrcoef(target, row.values.astype(float))[0, 1], axis=1
    )
    others["corr"] = corrs.values
    return others.nlargest(n, "corr")[["Country", "Continent", "Total", "corr"]]


def make_treemap() -> go.Figure:
    fig = go.Figure(go.Treemap(
        labels=cont_totals["Continent"].tolist(),
        parents=[""] * len(cont_totals),
        values=cont_totals["Total"].tolist(),
        marker_colors=[cont_color[c] for c in cont_totals["Continent"]],
        texttemplate="<b>%{label}</b><br>%{value:,}",
        hovertemplate="<b>%{label}</b><br>Total immigrants: %{value:,}<extra></extra>",
        textfont_size=15,
    ))
    fig.update_layout(
        margin=dict(t=10, l=5, r=5, b=5),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_continent_bar(continent_name: str) -> go.Figure:
    sub = df[df["Continent"] == continent_name].sort_values("Total", ascending=True).copy()
    sub["Label"] = label_series(sub["Country"])
    fig = go.Figure(go.Bar(
        x=sub["Total"],
        y=sub["Label"],
        orientation="h",
        marker_color=cont_color[continent_name],
        customdata=sub["Country"],
        hovertemplate="<b>%{customdata}</b><br>Total: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Total immigrants 1980–2013",
        height=max(320, len(sub) * 22 + 80),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, l=10, r=20, b=50),
    )
    return fig


def make_country_detail(country_name: str, n_corr: int = 5) -> go.Figure:
    row       = df[df["Country"] == country_name].iloc[0]
    continent = row["Continent"]
    color     = cont_color[continent]
    y_vals    = row[year_cols].values.astype(float)
    corr_df   = top_correlated(country_name, n=n_corr)
    fillcolor = (
        color.replace("rgb", "rgba").replace(")", ", 0.15)")
        if color.startswith("rgb") else color
    )
    # 2 rows: line chart on top, correlation bar on bottom
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"{label(country_name)} — immigration by year",
            f"Top {n_corr} most-correlated countries",
        ],
        row_heights=[0.55, 0.45],
        vertical_spacing=0.18,
    )
    # Row 1 — line chart
    fig.add_trace(go.Scatter(
        x=year_ints, y=y_vals,
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor=fillcolor,
        hovertemplate="Year: %{x}<br>Immigrants: %{y:,}<extra></extra>",
        name=country_name,
    ), row=1, col=1)
    # Row 2 — horizontal correlation bar
    corr_labels = label_series(corr_df["Country"])
    bar_colors  = [cont_color.get(c, "#aaa") for c in corr_df["Continent"]]
    fig.add_trace(go.Bar(
        x=corr_df["corr"],
        y=corr_labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in corr_df["corr"]],
        textposition="outside",
        customdata=corr_df["Country"],
        hovertemplate="<b>%{customdata}</b><br>Correlation: %{x:.3f}<extra></extra>",
    ), row=2, col=1)
    fig.update_xaxes(title_text="Year",     row=1, col=1)
    fig.update_yaxes(title_text="Immigrants", row=1, col=1)
    fig.update_xaxes(title_text="Pearson r", range=[0, 1.15], row=2, col=1)
    fig.update_yaxes(categoryorder="total ascending", automargin=True, row=2, col=1)
    fig.update_layout(
        height=560, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, l=10, r=60, b=40),
    )
    return fig


def make_stacked_bar(category_col: str, year_range: list[int]) -> go.Figure:
    yr_cols_filtered = [y for y in year_cols if year_range[0] <= int(y) <= year_range[1]]
    grouped = df.groupby(category_col)[yr_cols_filtered].sum()
    top_groups = grouped.sum(axis=1).nlargest(8).index.tolist()
    other_sum  = grouped.drop(index=top_groups).sum()
    plot_df    = grouped.loc[top_groups].T
    if not other_sum.empty and other_sum.sum() > 0:
        plot_df["Other"] = other_sum.values
    plot_df.index = plot_df.index.astype(int)

    # Alias legend labels when grouping by Country
    is_country_col = (category_col == "Country")

    fig = go.Figure()
    for col in plot_df.columns:
        display_name = label(str(col)) if is_country_col else str(col)
        fig.add_trace(go.Bar(
            name=display_name,
            x=plot_df.index.tolist(),
            y=plot_df[col].tolist(),
            hovertemplate=f"<b>{display_name}</b><br>Year: %{{x}}<br>Immigrants: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        xaxis_title="Year",
        yaxis_title="Immigrants",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, l=20, r=20, b=50),
    )
    return fig


def make_outlier_box(year_range: list[int]) -> go.Figure:
    yr_cols_filtered = [y for y in year_cols if year_range[0] <= int(y) <= year_range[1]]
    totals = df[yr_cols_filtered].sum(axis=1)
    fig = go.Figure(go.Box(
        y=totals,
        name="Total immigration",
        boxpoints="outliers",
        marker_color="#4e79a7",
        hovertemplate="Country index %{pointNumber}<br>Total: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="Total immigrants (filtered period)",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, l=20, r=20, b=40),
    )
    return fig


def make_heatmap(top_n: int = 20) -> go.Figure:
    sub = df.nlargest(top_n, "Total")[["Country"] + year_cols].copy()
    sub["Country"] = label_series(sub["Country"])
    sub = sub.set_index("Country")
    fig = px.imshow(
        sub.astype(float),
        labels=dict(x="Year", y="Country", color="Immigrants"),
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(
        height=max(400, top_n * 22 + 80),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, l=20, r=20, b=50),
        coloraxis_colorbar=dict(title="Immigrants"),
    )
    return fig


# ── 2.  App layout helpers ─────────────────────────────────────────────────────

YEAR_MIN = int(year_cols[0])
YEAR_MAX = int(year_cols[-1])

def card(title: str, *children):
    return dbc.Card([
        dbc.CardHeader(html.H6(title, className="mb-0 fw-semibold")),
        dbc.CardBody(list(children)),
    ], className="shadow-sm mb-3")


def kpi_card(label: str, value: str, color: str = "primary"):
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="text-muted small mb-1"),
            html.H4(value, className=f"text-{color} fw-bold mb-0"),
        ]),
        className="shadow-sm text-center",
    )


# ── 3.  Pages ──────────────────────────────────────────────────────────────────

# ---- Page 1: Overview -------------------------------------------------------

total_immigrants = int(df["Total"].sum())
top_country      = df.loc[df["Total"].idxmax(), "Country"]
top_continent    = cont_totals.loc[cont_totals["Total"].idxmax(), "Continent"]
num_countries    = len(df)

page_overview = dbc.Container([
    html.H4("🌍 Immigration Overview", className="mt-3 mb-3 fw-bold"),

    # KPI row
    dbc.Row([
        dbc.Col(kpi_card("Total Immigrants (1980–2013)", f"{total_immigrants:,}", "primary"), md=3),
        dbc.Col(kpi_card("Countries Tracked", str(num_countries), "success"), md=3),
        dbc.Col(kpi_card("Top Source Country", label(top_country), "warning"), md=3),
        dbc.Col(kpi_card("Top Source Continent", top_continent, "info"), md=3),
    ], className="mb-3"),

    # Treemap
    dbc.Row([
        dbc.Col(card("Immigration Share by Continent",
            dcc.Graph(id="treemap", figure=make_treemap(), config={"displayModeBar": False}),
        ), md=6),

        dbc.Col(card("Top 15 Countries by Total Immigration",
            dcc.Graph(
                id="top15-bar",
                figure=(lambda d: px.bar(
                    d,
                    x="Total", y="Label", orientation="h",
                    color="Continent",
                    color_discrete_map=cont_color,
                    labels={"Total": "Total Immigrants", "Label": ""},
                    custom_data=["Country"],
                    template="none",
                ).update_traces(
                    hovertemplate="<b>%{customdata[0]}</b><br>Total: %{x:,}<extra></extra>",
                ).update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, l=5, r=20, b=40),
                    height=340,
                    showlegend=False,
                    yaxis=dict(automargin=True),
                ))(
                    df.nlargest(15, "Total")
                      .sort_values("Total")
                      .assign(Label=lambda d: label_series(d["Country"]))
                ),
                config={"displayModeBar": False},
            ),
        ), md=6),
    ]),

    # Annual trend
    dbc.Row([
        dbc.Col(card("Annual Immigration Trend — All Countries Combined",
            dcc.Graph(
                id="annual-trend",
                figure=go.Figure(go.Scatter(
                    x=year_ints,
                    y=df[year_cols].sum().tolist(),
                    mode="lines+markers",
                    line=dict(color="#4e79a7", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(78,121,167,0.15)",
                    hovertemplate="Year: %{x}<br>Total: %{y:,}<extra></extra>",
                )).update_layout(
                    xaxis_title="Year", yaxis_title="Total Immigrants",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, l=20, r=20, b=50),
                    height=280,
                ),
                config={"displayModeBar": False},
            ),
        )),
    ]),
], fluid=True)


# ---- Page 2: Drill-down Explorer --------------------------------------------

page_explorer = dbc.Container([
    html.H4("🔍 Country Explorer", className="mt-3 mb-3 fw-bold"),

    dbc.Row([
        dbc.Col(card("Step 1 — Select a Continent",
            dbc.Select(
                id="continent-dd",
                options=[{"label": c, "value": c} for c in continents],
                value=continents[0],
            ),
            html.Div(id="continent-summary", className="text-muted small mt-2"),
        ), md=4),

        dbc.Col(card("Step 2 — Select a Country",
            dbc.Select(id="country-dd", options=[], value=None),
            html.Div(id="country-summary", className="text-muted small mt-2"),
        ), md=4),

        dbc.Col(card("Correlation depth (top N)",
            dcc.Slider(
                id="corr-slider",
                min=3, max=10, step=1, value=5,
                marks={i: str(i) for i in range(3, 11)},
            ),
        ), md=4),
    ]),

    dbc.Row([
        dbc.Col(card("Countries in Selected Continent",
            dcc.Graph(id="continent-bar", config={"displayModeBar": False}),
        ), md=5),
        dbc.Col(card("Country Detail + Correlated Countries",
            dcc.Graph(id="country-detail", config={"displayModeBar": False}),
        ), md=7),
    ]),
], fluid=True)


# ---- Page 3: Category & Time Analysis ---------------------------------------

page_analysis = dbc.Container([
    html.H4("📊 Category & Time Analysis", className="mt-3 mb-3 fw-bold"),

    dbc.Row([
        dbc.Col(card("Group immigration by …",
            dbc.Select(
                id="cat-col-dd",
                options=[{"label": c, "value": c} for c in categorical_cols],
                value=categorical_cols[0] if categorical_cols else None,
            ),
        ), md=4),

        dbc.Col(card("Year Range",
            dcc.RangeSlider(
                id="year-range-slider",
                min=YEAR_MIN, max=YEAR_MAX, step=1,
                value=[YEAR_MIN, YEAR_MAX],
                marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ), md=8),
    ]),

    dbc.Row([
        dbc.Col(card("Stacked Immigration by Category & Year",
            dcc.Graph(id="stacked-bar", config={"displayModeBar": False}),
        )),
    ]),

    dbc.Row([
        dbc.Col(card("Distribution of Country Totals (Box Plot)",
            dcc.Graph(id="outlier-box", config={"displayModeBar": False}),
        ), md=6),

        dbc.Col(card("Heatmap — Top N Countries Over Time",
            dbc.InputGroup([
                dbc.InputGroupText("Top N countries"),
                dbc.Input(id="heatmap-n-input", type="number", value=20, min=5, max=50, step=1),
                dbc.Button("Update", id="heatmap-btn", color="primary", n_clicks=0),
            ], className="mb-2"),
            dcc.Graph(id="heatmap-fig", config={"displayModeBar": False}),
        ), md=6),
    ]),
], fluid=True)


# ── 4.  Full app layout ────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "Canadian Immigration Dashboard"

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "220px",
    "padding": "2rem 1rem",
    "backgroundColor": "#2c3e50",
    "color": "white",
    "zIndex": 1000,
}

CONTENT_STYLE = {
    "marginLeft": "230px",
    "padding": "1.5rem",
    "backgroundColor": "#f4f6f9",
    "minHeight": "100vh",
}

sidebar = html.Div([
    html.H5("🍁 Canada Immigration", className="fw-bold mb-1", style={"color": "#e74c3c"}),
    html.P("1980 – 2013", className="text-muted small mb-4", style={"color": "#bdc3c7"}),
    html.Hr(style={"borderColor": "#4a6278"}),
    dbc.Nav([
        dbc.NavLink("📊 Overview",    href="/",          active="exact", className="text-white mb-1"),
        dbc.NavLink("🔍 Explorer",    href="/explorer",  active="exact", className="text-white mb-1"),
        dbc.NavLink("📈 Analysis",    href="/analysis",  active="exact", className="text-white mb-1"),
    ], vertical=True, pills=True),
    html.Hr(style={"borderColor": "#4a6278", "marginTop": "auto"}),
    html.P("Data: Kaggle / ammaraahmad", className="small", style={"color": "#7f8c8d", "marginTop": "1rem"}),
], style=SIDEBAR_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    # Persist explorer selections across page navigations
    dcc.Store(id="store-continent", data=continents[0]),
    dcc.Store(id="store-country",   data=None),
    sidebar,
    html.Div(id="page-content", style=CONTENT_STYLE),
])


# ── 5.  Callbacks ──────────────────────────────────────────────────────────────

# ---- Router -----------------------------------------------------------------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/explorer":
        return page_explorer
    if pathname == "/analysis":
        return page_analysis
    return page_overview


# ---- Explorer: sync store when continent dropdown changes -------------------
@app.callback(
    Output("store-continent", "data"),
    Input("continent-dd", "value"),
    prevent_initial_call=True,
)
def sync_continent_store(value):
    return value


# ---- Explorer: sync store when country dropdown changes --------------------
@app.callback(
    Output("store-country", "data"),
    Input("country-dd", "value"),
    prevent_initial_call=True,
)
def sync_country_store(value):
    return value


# ---- Explorer: populate country dropdown (from store OR dropdown) -----------
# Fires on: continent dropdown change, OR page load (url change injects store value)
@app.callback(
    Output("country-dd",        "options"),
    Output("country-dd",        "value"),
    Output("continent-summary", "children"),
    Input("continent-dd",       "value"),
    State("store-country",      "data"),
    prevent_initial_call=True,
)
def update_country_dd(continent, stored_country):
    if not continent:
        raise dash.exceptions.PreventUpdate
    sub       = df[df["Continent"] == continent].sort_values("Total", ascending=False)
    countries = sub["Country"].tolist()
    total     = int(sub["Total"].sum())
    summary   = f"{len(countries)} countries · {total:,} total immigrants"
    # Restore previously selected country if it belongs to this continent
    value = stored_country if stored_country in countries else (countries[0] if countries else None)
    return [{"label": c, "value": c} for c in countries], value, summary


# ---- Explorer: pre-fill dropdowns when page first loads --------------------
@app.callback(
    Output("continent-dd", "value"),
    Input("url", "pathname"),
    State("store-continent", "data"),
)
def init_explorer_continent(pathname, stored):
    if pathname != "/explorer":
        raise dash.exceptions.PreventUpdate
    return stored or continents[0]


# ---- Explorer: continent bar chart -----------------------------------------
@app.callback(
    Output("continent-bar", "figure"),
    Input("continent-dd",   "value"),
    prevent_initial_call=True,
)
def update_continent_bar(continent):
    if not continent:
        raise dash.exceptions.PreventUpdate
    return make_continent_bar(continent)


# ---- Explorer: country detail ----------------------------------------------
@app.callback(
    Output("country-detail",  "figure"),
    Output("country-summary", "children"),
    Input("country-dd",       "value"),
    Input("corr-slider",      "value"),
    prevent_initial_call=True,
)
def update_country_detail(country, n_corr):
    if not country:
        raise dash.exceptions.PreventUpdate
    row     = df[df["Country"] == country].iloc[0]
    total   = int(row["Total"])
    peak_yr = year_cols[row[year_cols].values.astype(float).argmax()]
    summary = f"Total: {total:,} · Peak year: {peak_yr}"
    return make_country_detail(country, n_corr=n_corr), summary


# ---- Analysis: trigger all charts on page load -----------------------------
@app.callback(
    Output("cat-col-dd",        "value"),
    Output("year-range-slider", "value"),
    Input("url",                "pathname"),
    State("cat-col-dd",         "value"),
    State("year-range-slider",  "value"),
)
def init_analysis(pathname, cat_val, yr_val):
    if pathname != "/analysis":
        raise dash.exceptions.PreventUpdate
    return (
        cat_val or (categorical_cols[0] if categorical_cols else None),
        yr_val  or [YEAR_MIN, YEAR_MAX],
    )


# ---- Analysis: stacked bar -------------------------------------------------
@app.callback(
    Output("stacked-bar",      "figure"),
    Input("cat-col-dd",        "value"),
    Input("year-range-slider", "value"),
    prevent_initial_call=True,
)
def update_stacked(cat_col, year_range):
    if not cat_col or not year_range:
        raise dash.exceptions.PreventUpdate
    return make_stacked_bar(cat_col, year_range)


# ---- Analysis: box plot ----------------------------------------------------
@app.callback(
    Output("outlier-box",      "figure"),
    Input("year-range-slider", "value"),
    prevent_initial_call=True,
)
def update_box(year_range):
    if not year_range:
        raise dash.exceptions.PreventUpdate
    return make_outlier_box(year_range)


# ---- Analysis: heatmap (button-triggered, also fires on page load) ---------
@app.callback(
    Output("heatmap-fig",    "figure"),
    Input("heatmap-btn",     "n_clicks"),
    Input("url",             "pathname"),
    State("heatmap-n-input", "value"),
)
def update_heatmap(_, pathname, top_n):
    if ctx.triggered_id == "url" and pathname != "/analysis":
        raise dash.exceptions.PreventUpdate
    n = int(top_n) if top_n and 5 <= int(top_n) <= 50 else 20
    return make_heatmap(n)


# ── 6.  Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)