import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rich
from collections import defaultdict
from touch_rugby_analyzer.constants import ASSETS_ROOT, ROOT, DATA_ROOT

output_data_root = DATA_ROOT / "output"
output_data_root.mkdir(parents=True, exist_ok=True)


def time_to_n_seconds(time_obj):
    return 3600 * time_obj.hour + 60 * time_obj.minute + time_obj.second


def load_data(data_path, local_team_name, other_team_name):
    data_df = pd.read_csv(data_path)
    data_df = data_df.dropna(axis=0, how="all", subset="Time")
    # data_df.Time = pd.to_datetime(data_df.Time).dt.time
    data_df.Time = pd.to_datetime(data_df.Time)
    game_start_events = data_df[data_df["Name"] == "Game Start"]
    game_end_events = data_df[data_df["Name"] == "Game End"]

    assert len(game_start_events) == len(game_end_events) == 2

    half_time_start_time = game_end_events.Time.to_list()[0]
    half_time_end_time = game_start_events.Time.to_list()[1]
    half_time_end_index = game_start_events.index.to_list()[1]

    new_game_end_time = half_time_start_time + pd.Timedelta(minutes=2)
    delta = new_game_end_time - half_time_end_time

    halftime_passed = False
    ids = []
    for i, row in data_df.iterrows():
        if i == half_time_end_index:
            halftime_passed = True
        if halftime_passed:
            ids.append(i)
    data_df.Time[ids] += delta

    data_df["Against France"].fillna(False, inplace=True)
    data_df["To Review"].fillna(False, inplace=True)

    data_df["Team"] = data_df.apply(
        lambda row: (
            local_team_name
            if (
                (row["Against France"] and row["Type"] in ["Penalty", "Turnover"])
                or (
                    not row["Against France"]
                    and row["Type"] not in ["Penalty", "Turnover"]
                )
            )
            else other_team_name
        ),
        axis=1,
    )
    data_df["TeamOld"] = data_df["Against France"].apply(
        lambda x: other_team_name if x else local_team_name
    )

    def add_team_after(data_df: pd.DataFrame):
        ball_owners = []
        for i, row in data_df.iterrows():
            against_local = row["Against France"]
            if i == 0:
                new_expected_ball_owner = (
                    other_team_name if against_local else local_team_name
                )
            else:
                if row["Type"] == "Try":
                    if not (
                        (against_local and ball_owners[-1] == other_team_name)
                        or (not against_local and ball_owners[-1] == local_team_name)
                    ):
                        rich.print(data_df)
                        rich.print(row)
                        raise Exception("wtf")
                    new_expected_ball_owner = (
                        local_team_name if against_local else other_team_name
                    )
                elif row["Type"] in ["Penalty", "Turnover"] and (
                    (against_local and ball_owners[-1] == local_team_name)
                    or (not against_local and ball_owners[-1] == other_team_name)
                ):
                    new_expected_ball_owner = (
                        other_team_name if against_local else local_team_name
                    )
                elif row["Type"] in ["Game Event"]:
                    new_expected_ball_owner = (
                        other_team_name if against_local else local_team_name
                    )
                else:
                    new_expected_ball_owner = ball_owners[-1]
            ball_owners.append(new_expected_ball_owner)
        data_df["ball_owner"] = ball_owners

    add_team_after(data_df)
    return data_df


def make_fig_1(data_df, local_team_name, other_team_name):
    events = ["Try", "Turnover", "Penalty"]
    fig = make_subplots(len(events), 1, subplot_titles=events, shared_xaxes=True)

    for i, event_name in enumerate(events):
        event_data = []
        event_local, event_other = 0, 0
        for j, row in data_df[data_df["Type"] == event_name].iterrows():
            if row["Against France"]:
                event_other += 1
            else:
                event_local += 1
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
        # rich.print(f"{row['Type']}-{row['Name']} (Against Local={row['Against France']})")

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

    fig.update_layout(annotations=annotations)
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
        data.append([
            len(_),
            np.round(np.mean(_), 3),
        ])
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
        new_stats_df["Total"] = new_stats_df.sum(axis=1)
        # new_stats_df["Average Possession Time"] = [np.mean(local_possesion_ts).round(3), np.mean(other_possesion_ts).round(3)]
        # rich.print(f"{np.mean(local_possesion_ts):.3f}s ({len(local_possesion_ts)} possessions)")
        # rich.print(f"{np.mean(other_possesion_ts):.3f}s ({len(other_possesion_ts)} possessions)")
        # output
        new_stats_df.to_csv(output_data_root / f"stats_{data_type}_v2.csv", index=True)
        output_dict[data_type] = new_stats_df
    return output_dict


def get_names(data_path: Path) -> tuple[str, str]:
    name_1 = data_path.stem.split("_")[0]
    name_2 = data_path.stem.split("_")[1]
    return name_1, name_2
