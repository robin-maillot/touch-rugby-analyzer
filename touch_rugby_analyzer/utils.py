import pandas as pd
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rich
from collections import defaultdict
import json
from typing import Tuple, Dict
from touch_rugby_analyzer.constants import DATA_ROOT

output_data_root = DATA_ROOT / "output"
output_data_root.mkdir(parents=True, exist_ok=True)

AGAINST_LOCALS_COL = "Against Team1"
POSSESSION_COL = "Possession Owner"
FIG_WIDTH = 1000


def save_json(data, p: Path):
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)  # indent for pretty formatting


def time_to_n_seconds(time_obj):
    return 3600 * time_obj.hour + 60 * time_obj.minute + time_obj.second


def opponent(team_name: str, team_names: Tuple[str, str] = ("Team 1", "Team 2")) -> str:
    if team_name == team_names[0]:
        return team_names[1]
    else:
        return team_names[0]


def is_turnover(row: pd.Series):
    _is_turnover = False
    if row.Type == "Try":
        _is_turnover = True
    elif row.Type == "Penalty" and row.Name not in [
        "Offside",
        "Not Moving Forward",
        "Shoulder",
    ]:
        _is_turnover = True
    elif row.Type == "Turnover" and row.Name not in ["6 Again"]:
        _is_turnover = True
    return _is_turnover


def is_fully_analysable(data_df: pd.DataFrame):
    game_start_events = data_df[data_df["Name"] == "Game Start"]
    game_end_events = data_df[data_df["Name"] == "Game End"]
    return len(game_start_events) == len(game_end_events) and len(game_end_events) > 0


def infer_possession_owner(data_df: pd.DataFrame):
    data_df["Turnover"] = data_df.apply(is_turnover, axis=1)
    if not is_fully_analysable(data_df):
        rich.print("[yellow] Cannot infer possession owner [/yellow]")
        if POSSESSION_COL not in data_df:
            data_df[POSSESSION_COL] = data_df[AGAINST_LOCALS_COL].apply(
                lambda against_local: "Team 2" if AGAINST_LOCALS_COL else "Team 1"
            )
        return

    turnover = False
    possession_owner = None
    new_possession_owners = []
    for i, row in data_df.iterrows():
        if POSSESSION_COL in row:
            row_possession_owner = row[POSSESSION_COL]
        elif AGAINST_LOCALS_COL in row:
            if row[AGAINST_LOCALS_COL]:
                row_possession_owner = "Team 2"
            else:
                row_possession_owner = "Team 1"
        else:
            raise Exception(
                f"Cannot find {POSSESSION_COL} or {AGAINST_LOCALS_COL} in {row}"
            )

        if row.Type == "Game Event":
            turnover = data_df.iloc[i - 1].Turnover
            if row.Name == "Game End":
                if turnover:
                    possession_owner = opponent(new_possession_owners[-1])
                else:
                    possession_owner = new_possession_owners[-1]
            else:
                possession_owner = row_possession_owner
        elif row.Type == "To Review":
            new_possession_owners.append(row_possession_owner)
            continue
        else:
            if turnover:
                possession_owner = opponent(possession_owner)
            turnover = row.Turnover
        new_possession_owners.append(possession_owner)
    data_df[POSSESSION_COL] = new_possession_owners


def infer_action_owner(data_df: pd.DataFrame):
    action_owners = []
    for i, row in data_df.iterrows():
        row_possession_owner = row[POSSESSION_COL]
        row_turnover = row.Turnover
        row_action_owner = row_possession_owner
        if row.Type in ["Penalty", "Turnover"] and not row_turnover:
            row_action_owner = opponent(row_possession_owner)
        else:
            row_action_owner = row_possession_owner
        action_owners.append(row_action_owner)
    data_df["Action Owner"] = action_owners


def add_game_time(data_df: pd.DataFrame, game_stopage_time: int = 300):
    """Add a GameTime column (seconds) handling multiple halves/periods.

    - GameTime = 0 at the first Game Start (or first event if no Game Start exists).
    - Rows before the first Game Start get negative GameTime.
    - Each subsequent Game Start picks up at (previous Game End game time + game_stopage_time).
    """
    game_start_times = data_df[data_df["Name"] == "Game Start"]["Time"]
    # Anchor: first Game Start, or first event if none
    anchor_dt = game_start_times.iloc[0] if len(game_start_times) > 0 else data_df["Time"].iloc[0]

    game_times = []
    time_offset = 0.0
    last_end_game_time = 0.0
    current_segment_dt = anchor_dt  # wall-clock reference for the current segment

    for _, row in data_df.iterrows():
        if row["Name"] == "Game Start" and row["Time"] != anchor_dt:
            # Subsequent Game Start: continue from last Game End + stoppage
            current_segment_dt = row["Time"]
            time_offset = last_end_game_time + game_stopage_time

        game_time = (row["Time"] - current_segment_dt).total_seconds() + time_offset
        game_times.append(game_time)

        if row["Name"] == "Game End":
            last_end_game_time = game_time

    data_df["GameTime"] = game_times


def replace_team_names(data_df: pd.DataFrame, mapping: Dict[str, str]):
    for col in ["Possession Owner", "Action Owner"]:
        if col in data_df:
            data_df[col] = data_df[col].apply(
                lambda team_name: mapping.get(team_name, team_name)
            )


def load_data(data_path: Path, simple: bool = False) -> pd.DataFrame:
    local_team_name, other_team_name = get_names(data_path)
    year, division_name, competition_name = get_year_division_competition(data_path)
    data_df = pd.read_csv(data_path)
    data_df = data_df.dropna(axis=0, how="all", subset="Time")
    data_df["Year"] = year
    data_df["Division"] = division_name
    data_df["Competition"] = competition_name
    if AGAINST_LOCALS_COL in data_df:
        data_df[AGAINST_LOCALS_COL].fillna(False, inplace=True)
    infer_possession_owner(data_df)
    infer_action_owner(data_df)
    replace_team_names(
        data_df, mapping={"Team 1": local_team_name, "Team 2": other_team_name}
    )

    data_df.Time = pd.to_datetime(data_df.Time)
    add_game_time(data_df)

    data_df["To Review"].fillna(False, inplace=True)
    return data_df


def make_fig_1(data_df, local_team_name, other_team_name):
    events = ["Try", "Penalty", "Turnover"]
    subplot_titles = ["Tries (from)", "Penalties (by)", "Turnovers (by)"]
    fig = make_subplots(
        len(events), 1, subplot_titles=subplot_titles, shared_xaxes=True
    )

    for i, event_name in enumerate(events):
        event_data = []
        event_local, event_other = 0, 0
        for j, row in data_df[data_df["Type"] == event_name].iterrows():
            if row["Team"] == local_team_name:
                event_local += 1
            else:
                event_other += 1
            event_data.append(
                [
                    row["Time"],
                    event_local,
                    event_other,
                    row["Youtube Link"],
                ]
            )
        event_df = pd.DataFrame(
            event_data,
            columns=[
                "Time",
                f"{event_name} {local_team_name}",
                f"{event_name} {other_team_name}",
                "Link",
            ],
        )

        fig.add_trace(
            go.Scatter(
                x=event_df["Time"],
                y=event_df[f"{event_name} {local_team_name}"],
                name=local_team_name,
                mode="markers+lines+text",
                marker_color="green",
                legendgroup=local_team_name,
                showlegend=i == 0,
                text=[
                    f"<a href='{row['Link']}'>*</a>" for i, row in event_df.iterrows()
                ],
                textposition="bottom center",
            ),
            i + 1,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=event_df["Time"],
                y=event_df[f"{event_name} {other_team_name}"],
                name=other_team_name,
                mode="markers+lines",
                marker_color="red",
                legendgroup=other_team_name,
                showlegend=i == 0,
            ),
            i + 1,
            1,
        )

    fig.update_layout(
        hovermode="x unified",
        title=f"Statistics for {local_team_name} vs {other_team_name}",
        height=int(FIG_WIDTH * 1.5),
        width=FIG_WIDTH,
    )
    fig.write_html(output_data_root / "events.html")
    return fig


def make_game_fig(data_df, local_team_name, other_team_name):
    fig = go.Figure()
    for n, colour in [(local_team_name, "green"), (other_team_name, "red")]:
        points_x, points_y = [], []
        for i, row in data_df.iterrows():
            if row["ball_owner"] == n:
                points_x.append(row["Time"])
                points_y.append(n)
            else:
                if i > 0 and data_df.iloc[i - 1]["ball_owner"] == n:
                    points_x.append(row["Time"])
                    points_y.append(n)
                    points_x.append(None)
                    points_y.append(None)
                else:
                    continue
        fig.add_trace(
            go.Scatter(
                x=points_x,
                y=points_y,
                mode="lines",
                hoverinfo=None,
                marker_color=colour,
                name=n,
            )
        )

    annotations = []
    for i, row in data_df.iterrows():
        # rich.print(f"{row['Type']}-{row['Name']} (Against Local={row[AGAINST_LOCALS_COL]})")

        if row["Type"] == "Penalty":
            color = "red"
        elif row["Type"] == "Turnover":
            color = "orange"
        elif row["Type"] == "Game Event":
            color = "black"
        elif row["Type"] == "Try":
            color = "green"
        else:
            color = "grey"
        hovertext = f"{row['Type']}-{row['Name']}"
        fig.add_trace(
            go.Scatter(
                x=[row["Time"]],
                y=[row["Team"]],
                mode="markers",
                # name=hovertext,
                hovertext=hovertext,
                marker_color=color,
                showlegend=False,
            )
        )
        annotations.append(
            dict(
                x=row["Time"],
                y=row["Team"],
                text=f"<a href='{row['Youtube Link']}'>*</a>",
                showarrow=False,
                yshift=5,
            )
        )

    fig.update_layout(annotations=annotations, height=FIG_WIDTH // 2, width=FIG_WIDTH)
    # fig.update_layout(hovermode="x unified", annotations)
    # fig.write_html(output_data_root / "events_v2.html")
    return fig


def get_possessions(data_df: pd.DataFrame) -> dict[str, int]:
    possessions = defaultdict(list)
    possession_start_time = None
    prev_ball_owner = None
    for i, row in data_df.iterrows():
        if possession_start_time is None or row.Name == "Game Start":
            possession_start_time = row.Time
            prev_ball_owner = row.ball_owner

        if row.ball_owner != prev_ball_owner:
            possessions[prev_ball_owner].append(
                (row.Time - possession_start_time).total_seconds()
            )
            possession_start_time = row.Time
            prev_ball_owner = row.ball_owner
    return possessions


def get_stats_df(
    data_df: pd.DataFrame, local_team_name: str, other_team_name: str
) -> dict[str, pd.DataFrame]:
    output_dict = dict()
    possession_stats = get_possessions(data_df)
    data = []
    index_names = []
    for n, _ in possession_stats.items():
        index_names.append(n)
        data.append(
            [
                len(_),
                np.round(np.mean(_), 3),
            ]
        )
    possession_stats_df = pd.DataFrame(
        data,
        index=index_names,
        columns=["N Possessions", "Av Possession (in s)"],
    )
    output_dict["Possession"] = possession_stats_df

    for data_type in ["Penalty", "Turnover", "Try"]:
        _data_df = data_df[
            [
                "Type",
                "Name",
                "Team",
            ]
        ][data_df.Type == data_type]
        output = _data_df.groupby(["Type", "Team", "Name"]).size()
        output.to_csv(output_data_root / f"stats_{data_type}.csv", index=True)

        column_names = list(_data_df["Name"].unique())

        new_stats_df = pd.DataFrame(
            np.zeros((2, len(column_names)), dtype=int),
            index=[local_team_name, other_team_name],
            columns=column_names,
        )
        for i, row in _data_df.iterrows():
            new_stats_df.loc[row["Team"], row["Name"]] += 1
        rich.print(new_stats_df.columns, len(new_stats_df.columns))
        if len(new_stats_df.columns) > 1:
            new_stats_df["Total"] = new_stats_df.sum(axis=1)
        # new_stats_df["Average Possession Time"] = [np.mean(local_possesion_ts).round(3), np.mean(other_possesion_ts).round(3)]
        # rich.print(f"{np.mean(local_possesion_ts):.3f}s ({len(local_possesion_ts)} possessions)")
        # rich.print(f"{np.mean(other_possesion_ts):.3f}s ({len(other_possesion_ts)} possessions)")
        # output
        new_stats_df.to_csv(output_data_root / f"stats_{data_type}_v2.csv", index=True)
        output_dict[data_type] = new_stats_df
    return output_dict


def get_names(data_path: Path) -> tuple[str, str]:
    split_data_path = data_path.stem.split("_")
    name_1 = split_data_path[-2]
    name_2 = split_data_path[-1]
    return name_1, name_2


def get_year_division_competition(data_path: Path) -> tuple[int, str, str]:
    split_data_path = data_path.stem.split("_")[:-2]
    if len(split_data_path) > 0:
        year, division_name, competition_name = split_data_path
    else:
        year, division_name, competition_name = 1999, "unknown", "unknown"
    return year, division_name, competition_name
