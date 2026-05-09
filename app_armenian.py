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

# ── 0.  Load data ─────────────────────────────────────────────────────────────
print("Fetching dataset via kagglehub …")
path     = kagglehub.dataset_download("ammaraahmad/immigration-to-canada")
filepath = os.path.join(path, "canadian_immegration_data.csv")
df       = pd.read_csv(filepath)
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

year_cols        = sorted([c for c in df.columns if str(c).isdigit()], key=int)
year_ints        = [int(y) for y in year_cols]
categorical_cols = [c for c in df.columns if c not in year_cols and c != "Total"]

if "Total" not in df.columns:
    df["Total"] = df[year_cols].sum(axis=1)

cont_totals = df.groupby("Continent")["Total"].sum().reset_index()
continents  = cont_totals["Continent"].tolist()
palette     = px.colors.qualitative.Set2
cont_color  = {c: palette[i % len(palette)] for i, c in enumerate(continents)}

YEAR_MIN = int(year_cols[0])
YEAR_MAX = int(year_cols[-1])

# ── 1.  Armenia highlight config ──────────────────────────────────────────────
HIGHLIGHT_COUNTRY   = "Armenia"
HIGHLIGHT_CONTINENT = df.loc[df["Country"] == HIGHLIGHT_COUNTRY, "Continent"].values[0]
HIGHLIGHT_COLOR     = "#e74c3c"
HIGHLIGHT_LABEL     = "🇦🇲 Armenia"

arm_row     = df[df["Country"] == HIGHLIGHT_COUNTRY].iloc[0]
arm_total   = int(arm_row["Total"])
arm_peak_yr = year_cols[arm_row[year_cols].values.astype(float).argmax()]
arm_rank    = int((df["Total"] > arm_total).sum()) + 1


def label(country: str) -> str:
    lbl = COUNTRY_ALIASES.get(country, country)
    return HIGHLIGHT_LABEL if country == HIGHLIGHT_COUNTRY else lbl

def label_series(s: "pd.Series") -> "pd.Series":
    return s.map(label)

def bar_colors_single(countries: "pd.Series", base_color: str) -> list:
    return [HIGHLIGHT_COLOR if c == HIGHLIGHT_COUNTRY else base_color for c in countries]

def bar_colors_map(countries: "pd.Series", color_map: dict) -> list:
    return [HIGHLIGHT_COLOR if c == HIGHLIGHT_COUNTRY else color_map.get(c, "#aaa")
            for c in countries]

# ── 3.  Chart builders ────────────────────────────────────────────────────────

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
    fig.update_layout(margin=dict(t=10, l=5, r=5, b=5), height=340,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def make_continent_bar(continent_name: str) -> go.Figure:
    sub = df[df["Continent"] == continent_name].sort_values("Total", ascending=True).copy()
    sub["Label"] = label_series(sub["Country"])
    colors     = bar_colors_single(sub["Country"], cont_color[continent_name])
    linewidths = [2 if c == HIGHLIGHT_COUNTRY else 0 for c in sub["Country"]]
    fig = go.Figure(go.Bar(
        x=sub["Total"], y=sub["Label"], orientation="h",
        marker=dict(color=colors, line=dict(color="#222", width=linewidths)),
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
    row        = df[df["Country"] == country_name].iloc[0]
    continent  = row["Continent"]
    is_armenia = (country_name == HIGHLIGHT_COUNTRY)
    color      = HIGHLIGHT_COLOR if is_armenia else cont_color[continent]
    y_vals     = row[year_cols].values.astype(float)
    corr_df    = top_correlated(country_name, n=n_corr)
    fillcolor  = ("rgba(231,76,60,0.12)" if is_armenia else
                  color.replace("rgb", "rgba").replace(")", ", 0.15)")
                  if color.startswith("rgb") else color + "26")
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"{label(country_name)} — immigration by year",
            f"Top {n_corr} most-correlated countries",
        ],
        row_heights=[0.55, 0.45], vertical_spacing=0.18,
    )
    fig.add_trace(go.Scatter(
        x=year_ints, y=y_vals, mode="lines+markers",
        line=dict(color=color, width=3 if is_armenia else 2.5),
        marker=dict(size=6 if is_armenia else 5,
                    line=dict(color="#222", width=1) if is_armenia else {}),
        fill="tozeroy", fillcolor=fillcolor,
        hovertemplate="Year: %{x}<br>Immigrants: %{y:,}<extra></extra>",
        name=label(country_name),
    ), row=1, col=1)
    corr_labels = label_series(corr_df["Country"])
    bar_clrs    = bar_colors_map(corr_df["Country"], cont_color)
    lw          = [2 if c == HIGHLIGHT_COUNTRY else 0 for c in corr_df["Country"]]
    fig.add_trace(go.Bar(
        x=corr_df["corr"], y=corr_labels, orientation="h",
        marker=dict(color=bar_clrs, line=dict(color="#222", width=lw)),
        text=[f"{v:.2f}" for v in corr_df["corr"]], textposition="outside",
        customdata=corr_df["Country"],
        hovertemplate="<b>%{customdata}</b><br>Correlation: %{x:.3f}<extra></extra>",
    ), row=2, col=1)
    fig.update_xaxes(title_text="Year",       row=1, col=1)
    fig.update_yaxes(title_text="Immigrants", row=1, col=1)
    fig.update_xaxes(title_text="Pearson r",  range=[0, 1.15], row=2, col=1)
    fig.update_yaxes(categoryorder="total ascending", automargin=True, row=2, col=1)
    fig.update_layout(height=560, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, l=10, r=60, b=40))
    return fig


def make_stacked_bar(category_col: str, year_range: list, armenia_mode: bool = False) -> go.Figure:
    yr_f    = [y for y in year_cols if year_range[0] <= int(y) <= year_range[1]]
    grouped = df.groupby(category_col)[yr_f].sum()

    if armenia_mode and category_col == "Country":
        # Armenia-mode: show ONLY Armenia
        if HIGHLIGHT_COUNTRY in grouped.index:
            plot_df = grouped.loc[[HIGHLIGHT_COUNTRY]].T
        else:
            plot_df = pd.DataFrame(index=yr_f)
        plot_df.index = plot_df.index.astype(int)
        fig = go.Figure()
        if not plot_df.empty and HIGHLIGHT_COUNTRY in plot_df.columns:
            fig.add_trace(go.Bar(
                name=HIGHLIGHT_LABEL,
                x=plot_df.index.tolist(),
                y=plot_df[HIGHLIGHT_COUNTRY].tolist(),
                marker_color=HIGHLIGHT_COLOR,
                marker_line=dict(color="#922b21", width=1),
                hovertemplate=f"<b>{HIGHLIGHT_LABEL}</b><br>Year: %{{x}}<br>Immigrants: %{{y:,}}<extra></extra>",
            ))
    else:
        top_groups = grouped.sum(axis=1).nlargest(8).index.tolist()
        other_sum  = grouped.drop(index=top_groups).sum()
        # Force Armenia in when grouping by Country
        if category_col == "Country" and HIGHLIGHT_COUNTRY not in top_groups and HIGHLIGHT_COUNTRY in grouped.index:
            top_groups = top_groups[:-1] + [HIGHLIGHT_COUNTRY]
            other_sum  = grouped.drop(index=top_groups).sum()
        plot_df = grouped.loc[top_groups].T
        if not other_sum.empty and other_sum.sum() > 0:
            plot_df["Other"] = other_sum.values
        plot_df.index = plot_df.index.astype(int)

        is_country_col = (category_col == "Country")
        fig = go.Figure()
        for col in plot_df.columns:
            raw_name     = str(col)
            display_name = label(raw_name) if is_country_col else raw_name
            is_arm       = is_country_col and raw_name == HIGHLIGHT_COUNTRY
            fig.add_trace(go.Bar(
                name=display_name,
                x=plot_df.index.tolist(),
                y=plot_df[col].tolist(),
                marker_color=HIGHLIGHT_COLOR if is_arm else None,
                marker_line=dict(color="#222", width=1) if is_arm else {},
                hovertemplate=f"<b>{display_name}</b><br>Year: %{{x}}<br>Immigrants: %{{y:,}}<extra></extra>",
            ))

    fig.update_layout(
        barmode="stack",
        xaxis_title="Year", yaxis_title="Immigrants",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, l=20, r=20, b=50),
    )
    return fig


def make_outlier_box(year_range: list) -> go.Figure:
    yr_f      = [y for y in year_cols if year_range[0] <= int(y) <= year_range[1]]
    totals    = df[yr_f].sum(axis=1)
    arm_total_filtered = int(df.loc[df["Country"] == HIGHLIGHT_COUNTRY, yr_f].sum(axis=1).iloc[0])
    point_colors = [HIGHLIGHT_COLOR if c == HIGHLIGHT_COUNTRY else "#4e79a7" for c in df["Country"]]
    point_sizes  = [10 if c == HIGHLIGHT_COUNTRY else 6 for c in df["Country"]]
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=totals, name="All countries", boxpoints=False,
        marker_color="#4e79a7", line_color="#4e79a7",
        fillcolor="rgba(78,121,167,0.25)",
    ))
    fig.add_trace(go.Scatter(
        x=["All countries"] * len(df), y=totals, mode="markers",
        marker=dict(color=point_colors, size=point_sizes,
                    line=dict(color="#222", width=[1 if c == HIGHLIGHT_COUNTRY else 0 for c in df["Country"]])),
        customdata=df["Country"],
        hovertemplate="<b>%{customdata}</b><br>Total: %{y:,}<extra></extra>",
        showlegend=False,
    ))
    fig.add_annotation(
        x="All countries", y=arm_total_filtered,
        text=f"  {HIGHLIGHT_LABEL}", showarrow=True, arrowhead=2,
        arrowcolor=HIGHLIGHT_COLOR,
        font=dict(color=HIGHLIGHT_COLOR, size=11, family="Arial Black"),
        ax=60, ay=-20,
    )
    fig.update_layout(yaxis_title="Total immigrants (filtered period)",
                      height=380, paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=20, l=20, r=20, b=40))
    return fig


def make_heatmap(top_n: int = 20) -> go.Figure:
    top_df = df.nlargest(top_n, "Total")
    if HIGHLIGHT_COUNTRY not in top_df["Country"].values:
        arm_df = df[df["Country"] == HIGHLIGHT_COUNTRY]
        top_df = pd.concat([top_df.iloc[:-1], arm_df])
    sub = top_df[["Country"] + year_cols].copy()
    sub["Country"] = label_series(sub["Country"])
    sub = sub.set_index("Country")
    fig = px.imshow(sub.astype(float),
                    labels=dict(x="Year", y="Country", color="Immigrants"),
                    color_continuous_scale="Blues", aspect="auto")
    arm_label = label(HIGHLIGHT_COUNTRY)
    fig.update_yaxes(
        tickfont=dict(size=11),
        tickvals=list(range(len(sub))),
        ticktext=[f"<b>{t}</b>" if t == arm_label else t for t in sub.index],
    )
    fig.update_layout(height=max(400, top_n * 22 + 80),
                      paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=20, l=20, r=20, b=50),
                      coloraxis_colorbar=dict(title="Immigrants"))
    return fig


def top_correlated(country_name: str, n: int = 5) -> pd.DataFrame:
    target = df[df["Country"] == country_name][year_cols].values.flatten().astype(float)
    others = df[df["Country"] != country_name].copy()
    corrs  = others[year_cols].apply(
        lambda row: np.corrcoef(target, row.values.astype(float))[0, 1], axis=1
    )
    others["corr"] = corrs.values
    return others.nlargest(n, "corr")[["Country", "Continent", "Total", "corr"]]


# ── 4.  Layout helpers ────────────────────────────────────────────────────────

def card(title: str, *children):
    return dbc.Card([
        dbc.CardHeader(html.H6(title, className="mb-0 fw-semibold")),
        dbc.CardBody(list(children)),
    ], className="shadow-sm mb-3")


def kpi_card(lbl: str, value: str, color: str = "primary"):
    return dbc.Card(
        dbc.CardBody([
            html.P(lbl, className="text-muted small mb-1"),
            html.H4(value, className=f"text-{color} fw-bold mb-0"),
        ]),
        className="shadow-sm text-center",
    )


# ── 5.  Pages ─────────────────────────────────────────────────────────────────

total_immigrants = int(df["Total"].sum())
top_country      = df.loc[df["Total"].idxmax(), "Country"]
top_continent    = cont_totals.loc[cont_totals["Total"].idxmax(), "Continent"]
num_countries    = len(df)

page_overview = dbc.Container([
    html.H4("🌍 Immigration Overview", className="mt-3 mb-3 fw-bold"),
    dbc.Row([
        dbc.Col(kpi_card("Total Immigrants (1980–2013)", f"{total_immigrants:,}", "primary"), md=3),
        dbc.Col(kpi_card("Countries Tracked", str(num_countries), "success"), md=3),
        dbc.Col(kpi_card("Top Source Country", label(top_country), "warning"), md=3),
        dbc.Col(kpi_card("Top Continent", top_continent, "info"), md=3),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(card("Immigration Share by Continent",
            dcc.Graph(id="treemap", figure=make_treemap(), config={"displayModeBar": False}),
        ), md=6),
        dbc.Col(card("Top 15 Countries by Total Immigration",
            dcc.Graph(
                id="top15-bar",
                figure=(lambda d: go.Figure(go.Bar(
                    x=d["Total"], y=d["Label"], orientation="h",
                    marker=dict(
                        color=[HIGHLIGHT_COLOR if c == HIGHLIGHT_COUNTRY
                               else cont_color.get(df.loc[df["Country"] == c, "Continent"].values[0], "#aaa")
                               for c in d["Country"]],
                        line=dict(
                            color=["#222" if c == HIGHLIGHT_COUNTRY else "rgba(0,0,0,0)" for c in d["Country"]],
                            width=[2 if c == HIGHLIGHT_COUNTRY else 0 for c in d["Country"]],
                        ),
                    ),
                    customdata=d["Country"],
                    hovertemplate="<b>%{customdata}</b><br>Total: %{x:,}<extra></extra>",
                )).update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, l=5, r=20, b=40),
                    height=340, xaxis_title="Total Immigrants",
                    yaxis=dict(automargin=True),
                ))(df.nlargest(15, "Total").sort_values("Total")
                     .assign(Label=lambda d: label_series(d["Country"]))),
                config={"displayModeBar": False},
            ),
        ), md=6),
    ]),
    dbc.Row([
        dbc.Col(card("Annual Immigration Trend — All Countries Combined",
            dcc.Graph(
                id="annual-trend",
                figure=go.Figure(go.Scatter(
                    x=year_ints, y=df[year_cols].sum().tolist(),
                    mode="lines+markers",
                    line=dict(color="#4e79a7", width=3),
                    fill="tozeroy", fillcolor="rgba(78,121,167,0.15)",
                    hovertemplate="Year: %{x}<br>Total: %{y:,}<extra></extra>",
                )).update_layout(
                    xaxis_title="Year", yaxis_title="Total Immigrants",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, l=20, r=20, b=50), height=280,
                ),
                config={"displayModeBar": False},
            ),
        )),
    ]),
], fluid=True)


page_explorer = dbc.Container([
    html.H4("🔍 Country Explorer", className="mt-3 mb-3 fw-bold"),
    dbc.Row([
        dbc.Col(card("Step 1 — Select a Continent",
            dbc.Select(id="continent-dd",
                       options=[{"label": c, "value": c} for c in continents],
                       value=continents[0]),
            html.Div(id="continent-summary", className="text-muted small mt-2"),
        ), md=4),
        dbc.Col(card("Step 2 — Select a Country",
            dbc.Select(id="country-dd", options=[], value=None),
            html.Div(id="country-summary", className="text-muted small mt-2"),
        ), md=4),
        dbc.Col(card("Correlation depth (top N)",
            dcc.Slider(id="corr-slider", min=3, max=10, step=1, value=5,
                       marks={i: str(i) for i in range(3, 11)}),
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


page_analysis = dbc.Container([
    html.H4("📊 Category & Time Analysis", className="mt-3 mb-3 fw-bold"),
    dbc.Row([
        dbc.Col(card("Group immigration by …",
            dbc.Select(id="cat-col-dd",
                       options=[{"label": c, "value": c} for c in categorical_cols],
                       value=categorical_cols[0] if categorical_cols else None),
        ), md=4),
        dbc.Col(card("Year Range",
            dcc.RangeSlider(id="year-range-slider",
                            min=YEAR_MIN, max=YEAR_MAX, step=1,
                            value=[YEAR_MIN, YEAR_MAX],
                            marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)},
                            tooltip={"placement": "bottom", "always_visible": True}),
        ), md=8),
    ]),
    dbc.Row([dbc.Col(card("Stacked Immigration by Category & Year",
        dcc.Graph(id="stacked-bar", config={"displayModeBar": False}),
    ))]),
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


# ── 6.  App shell ─────────────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
app.title = "Canadian Immigration Dashboard"

SIDEBAR_W = "240px"

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": SIDEBAR_W, "padding": "1.5rem 1rem",
    "backgroundColor": "#2c3e50", "color": "white", "zIndex": 1000,
    "overflowY": "auto",
}
CONTENT_STYLE = {
    "marginLeft": SIDEBAR_W, "padding": "1.5rem",
    "backgroundColor": "#f4f6f9", "minHeight": "100vh",
}

# Armenia KPI panel shown in sidebar
arm_sidebar_panel = html.Div(id="arm-sidebar-panel")

sidebar = html.Div([
    html.H5("🍁 Canada Immigration", className="fw-bold mb-0",
            style={"color": "#e74c3c", "fontSize": "1rem"}),
    html.P("1980 – 2013", className="mb-3",
           style={"color": "#bdc3c7", "fontSize": "0.78rem"}),

    # ── Armenian mode toggle ──────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("🇦🇲", style={"fontSize": "1.1rem", "marginRight": "6px"}),
            html.Span("Armenian mode", style={"fontWeight": "600",
                                              "fontSize": "0.85rem",
                                              "color": "white"}),
        ], className="d-flex align-items-center mb-1"),
        dbc.Switch(id="armenia-mode-toggle", value=False,
                   label="", className="mb-0",
                   style={"transform": "scale(1.2)", "transformOrigin": "left"}),
    ], id="armenia-toggle-wrapper",
       style={"backgroundColor": "#1a252f", "borderRadius": "8px",
              "padding": "10px 12px", "marginBottom": "12px",
              "border": "1px solid #4a6278"}),

    arm_sidebar_panel,

    html.Hr(style={"borderColor": "#4a6278"}),

    dbc.Nav([
        dbc.NavLink("📊 Overview",  href="/",         active="exact", className="text-white mb-1"),
        dbc.NavLink("🔍 Explorer",  href="/explorer", active="exact", className="text-white mb-1"),
        dbc.NavLink("📈 Analysis",  href="/analysis", active="exact", className="text-white mb-1"),
    ], vertical=True, pills=True),

    html.Hr(style={"borderColor": "#4a6278"}),
    html.P("Data: Kaggle / ammaraahmad",
           style={"color": "#7f8c8d", "fontSize": "0.72rem", "marginTop": "0.5rem"}),
], style=SIDEBAR_STYLE)


app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="store-armenia-mode", data=False),
    dcc.Store(id="store-continent",    data=HIGHLIGHT_CONTINENT),
    dcc.Store(id="store-country",      data=HIGHLIGHT_COUNTRY),
    sidebar,
    html.Div(id="page-content", style=CONTENT_STYLE),
])


# ── 7.  Callbacks ─────────────────────────────────────────────────────────────

# ---- Persist toggle to store ------------------------------------------------
@app.callback(
    Output("store-armenia-mode", "data"),
    Input("armenia-mode-toggle", "value"),
)
def sync_armenia_mode(val):
    return val


# ---- Armenia sidebar panel: change appearance based on mode -----------------
@app.callback(
    Output("arm-sidebar-panel",     "children"),
    Output("arm-sidebar-panel",     "style"),
    Output("armenia-toggle-wrapper","style"),
    Input("store-armenia-mode",     "data"),
)
def update_arm_sidebar(armenia_mode):
    active_style = {
        "backgroundColor": "#922b21", "borderRadius": "8px",
        "padding": "10px 12px", "marginBottom": "12px",
        "border": f"2px solid {HIGHLIGHT_COLOR}",
        "boxShadow": f"0 0 8px {HIGHLIGHT_COLOR}88",
    }
    inactive_style = {
        "backgroundColor": "#1a252f", "borderRadius": "8px",
        "padding": "10px 12px", "marginBottom": "12px",
        "border": "1px solid #4a6278",
    }
    wrapper_style = active_style if armenia_mode else inactive_style

    panel = html.Div([
        html.Div([
            html.Span(HIGHLIGHT_LABEL, style={
                "color": HIGHLIGHT_COLOR if not armenia_mode else "#fff",
                "fontWeight": "700", "fontSize": "0.85rem",
            }),
            html.Span(" · Total immigrants", style={"color": "#bdc3c7", "fontSize": "0.72rem"}),
        ]),
        html.Div(f"{arm_total:,}", style={
            "color": HIGHLIGHT_COLOR if not armenia_mode else "#fff",
            "fontWeight": "800", "fontSize": "1.4rem", "lineHeight": "1.2",
        }),
        html.Div(f"Rank #{arm_rank}  ·  Peak {arm_peak_yr}", style={
            "color": "#95a5a6", "fontSize": "0.72rem", "marginTop": "2px",
        }),
    ], style={
        "backgroundColor": "#922b21" if armenia_mode else "#1a252f",
        "borderRadius": "8px", "padding": "10px 12px", "marginBottom": "12px",
        "border": f"2px solid {HIGHLIGHT_COLOR}" if armenia_mode else "1px solid #4a6278",
    })

    panel_style = {"display": "block"} if armenia_mode else {"display": "none"}

    return panel, panel_style, wrapper_style


# ---- Router -----------------------------------------------------------------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/explorer":
        return page_explorer
    if pathname == "/analysis":
        return page_analysis
    return page_overview


# ---- Explorer: init continent on page load ----------------------------------
@app.callback(
    Output("continent-dd", "value"),
    Input("url", "pathname"),
    State("store-continent", "data"),
    State("store-armenia-mode", "data"),
)
def init_explorer_continent(pathname, stored_cont, armenia_mode):
    if pathname != "/explorer":
        raise dash.exceptions.PreventUpdate
    return HIGHLIGHT_CONTINENT if armenia_mode else (stored_cont or continents[0])


# ---- Explorer: handle armenia mode toggle on explorer page -------------------
@app.callback(
    Output("country-dd", "value", allow_duplicate=True),
    Output("continent-dd", "value", allow_duplicate=True),
    Input("store-armenia-mode", "data"),
    State("url", "pathname"),
    State("store-continent", "data"),
    prevent_initial_call=True,
)
def handle_armenia_toggle_explorer(armenia_mode, pathname, stored_cont):
    if pathname != "/explorer":
        raise dash.exceptions.PreventUpdate
    # When toggling Armenia mode on explorer page, update continent
    continent = HIGHLIGHT_CONTINENT if armenia_mode else (stored_cont or continents[0])
    # Armenia will be selected automatically via update_country_dd when continent changes
    return dash.no_update, continent


# ---- Explorer: populate country dropdown ------------------------------------
@app.callback(
    Output("country-dd",        "options"),
    Output("country-dd",        "value"),
    Output("continent-summary", "children"),
    Input("continent-dd",       "value"),
    State("store-country",      "data"),
    State("store-armenia-mode", "data"),
    prevent_initial_call=True,
)
def update_country_dd(continent, stored_country, armenia_mode):
    if not continent:
        raise dash.exceptions.PreventUpdate
    sub      = df[df["Continent"] == continent].sort_values("Total", ascending=False)
    countries = sub["Country"].tolist()
    total    = int(sub["Total"].sum())
    summary  = f"{len(countries)} countries · {total:,} total immigrants"
    
    # Select country based on mode
    if armenia_mode:
        # Armenia mode: only select Armenia if available in this continent
        value = HIGHLIGHT_COUNTRY if HIGHLIGHT_COUNTRY in countries else countries[0]
    else:
        # Normal mode: use stored country if available in this continent, else first country
        if stored_country and stored_country in countries:
            value = stored_country
        else:
            value = countries[0]
    
    return [{"label": label(c), "value": c} for c in countries], value, summary


# ---- Explorer: sync stores --------------------------------------------------
@app.callback(Output("store-continent", "data"),
              Input("continent-dd", "value"), prevent_initial_call=True)
def sync_continent_store(v): return v

@app.callback(Output("store-country", "data"),
              Input("country-dd", "value"), prevent_initial_call=True)
def sync_country_store(v): return v


# ---- Explorer: continent bar ------------------------------------------------
@app.callback(
    Output("continent-bar", "figure"),
    Input("continent-dd",   "value"),
    prevent_initial_call=True,
)
def update_continent_bar(continent):
    if not continent:
        raise dash.exceptions.PreventUpdate
    return make_continent_bar(continent)


# ---- Explorer: country detail -----------------------------------------------
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


# ---- Analysis: init on page load --------------------------------------------
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
    return (cat_val or (categorical_cols[0] if categorical_cols else None),
            yr_val  or [YEAR_MIN, YEAR_MAX])


# ---- Analysis: stacked bar — reads armenia-mode from store ------------------
@app.callback(
    Output("stacked-bar",        "figure"),
    Input("cat-col-dd",          "value"),
    Input("year-range-slider",   "value"),
    Input("store-armenia-mode",  "data"),
    prevent_initial_call=True,
)
def update_stacked(cat_col, year_range, armenia_mode):
    if not cat_col or not year_range:
        raise dash.exceptions.PreventUpdate
    return make_stacked_bar(cat_col, year_range, armenia_mode=bool(armenia_mode))


# ---- Analysis: box plot -----------------------------------------------------
@app.callback(
    Output("outlier-box",      "figure"),
    Input("year-range-slider", "value"),
    prevent_initial_call=True,
)
def update_box(year_range):
    if not year_range:
        raise dash.exceptions.PreventUpdate
    return make_outlier_box(year_range)


# ---- Analysis: heatmap ------------------------------------------------------
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


# ── 8.  Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8051)