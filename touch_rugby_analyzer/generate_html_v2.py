import rich

from touch_rugby_analyzer.constants import ASSETS_ROOT, ROOT, DATA_ROOT
from touch_rugby_analyzer import utils
from typing import Tuple
import pandas as pd
import altair as alt
from pathlib import Path

def generate_html_from_df(df: pd.DataFrame, output_html_path: Path, team_names: Tuple = ('Team 1', 'Team 2')):
    # --- 1. Global Time Filter ---
    start_events = df[df['Name'] == 'Game Start']
    end_events = df[df['Name'] == 'Game End']

    if not start_events.empty:
        global_start = start_events.iloc[0]['GameTime']
        start_link = start_events.iloc[0]['Youtube Link']
    else:
        global_start = df['GameTime'].min()
        start_link = ''

    if not end_events.empty:
        global_end = end_events.iloc[-1]['GameTime']
        end_link = end_events.iloc[-1]['Youtube Link']
    else:
        global_end = df['GameTime'].max()
        end_link = ''
    df = df[(df['GameTime'] >= global_start) & (df['GameTime'] <= global_end)].copy()
    df = df.sort_values('GameTime')

    # Common visual properties
    chart_width = 800
    team_colors = alt.Scale(domain=team_names, range=['#008000', '#FF0000'])  # Green, Red

    # --- 2. Tries Chart Data & Plot ---
    tries_data = []
    for team in team_names:
        # Start Point
        tries_data.append({
            'GameTime': global_start,
            'Team': team,
            'Event': 'Game Start',
            'Count': 0,
            'Link': start_link
        })

        # Tries
        team_tries = df[(df['Type'] == 'Try') & (df['Action Owner'] == team)].sort_values('GameTime')
        count = 0
        for _, row in team_tries.iterrows():
            count += 1
            tries_data.append({
                'GameTime': row['GameTime'],
                'Team': team,
                'Event': row['Name'],
                'Count': count,
                'Link': row['Youtube Link']
            })

        # End Point
        tries_data.append({
            'GameTime': global_end,
            'Team': team,
            'Event': 'Game End',
            'Count': count,  # Holds final score
            'Link': end_link
        })

    tries_df = pd.DataFrame(tries_data)

    tries_base = alt.Chart(tries_df).encode(
        x=alt.X('GameTime:T', axis=alt.Axis(format='%H:%M:%S', title='Game Time')),
        y=alt.Y('Count:Q', axis=alt.Axis(tickMinStep=1), title='Cumulative Tries'),
        color=alt.Color('Team:N', scale=team_colors)
    )
    tries_lines = tries_base.mark_line(interpolate='step-after').encode(
        order='GameTime:T'
    )
    tries_points = tries_base.transform_filter(
        (alt.datum.Event != 'Game Start') & (alt.datum.Event != 'Game End')
        # Only clickable tries? Or all? Previous request said start/end too. Let's keep all clickable if link exists.
    ).mark_circle(size=100).encode(
        href='Link:N',
        tooltip=['GameTime:T', 'Team:N', 'Event:N', 'Count:Q']
    )
    # Actually, let's keep Start/End points but maybe smaller or just consistent. The user liked Start/End points before.
    tries_chart = (tries_lines + tries_points).properties(
        title='Tries vs Time',
        width=chart_width,
        height=200
    ).interactive()

    # --- 3. Penalties Chart Data & Plot ---
    penalties_data = []
    for team in team_names:
        # Start Point
        penalties_data.append({
            'GameTime': global_start,
            'Team': team,
            'Event': 'Game Start',
            'Count': 0,
            'Link': start_link
        })

        # Penalties
        # Note: 'Action Owner' is the one committing the penalty usually?
        # Or is it the beneficiary?
        # In standard rugby data, "Penalty - Forward Pass - Team 1" usually means Team 1 committed it.
        # The user input has "Action Owner". Let's assume Action Owner is the team getting the penalty count.
        team_penalties = df[(df['Type'] == 'Penalty') & (df['Action Owner'] == team)].sort_values('GameTime')
        count = 0
        for _, row in team_penalties.iterrows():
            count += 1
            penalties_data.append({
                'GameTime': row['GameTime'],
                'Team': team,
                'Event': row['Name'],
                'Count': count,
                'Link': row['Youtube Link']
            })

        # End Point
        penalties_data.append({
            'GameTime': global_end,
            'Team': team,
            'Event': 'Game End',
            'Count': count,
            'Link': end_link
        })

    pen_df = pd.DataFrame(penalties_data)

    pen_base = alt.Chart(pen_df).encode(
        x=alt.X('GameTime:T', axis=alt.Axis(format='%H:%M:%S', title='Game Time')),
        y=alt.Y('Count:Q', axis=alt.Axis(tickMinStep=1), title='Cumulative Penalties'),
        color=alt.Color('Team:N', scale=team_colors)
    )
    pen_lines = pen_base.mark_line(interpolate='step-after').encode(
        order='GameTime:T'
    )
    pen_points = pen_base.mark_circle(size=100).encode(
        href='Link:N',
        tooltip=['GameTime:T', 'Team:N', 'Event:N', 'Count:Q']
    )

    pen_chart = (pen_lines + pen_points).properties(
        title='Penalties vs Time',
        width=chart_width,
        height=200
    ).interactive()

    # --- 4. Possession Chart Data & Plot ---
    # Logic from previous step
    df['Owner_Change'] = df['Possession Owner'] != df['Possession Owner'].shift(1)
    df['Is_Start'] = df['Name'] == 'Game Start'
    df['Prev_Is_End'] = df['Name'].shift(1) == 'Game End'
    df['New_Possession'] = df['Owner_Change'] | df['Is_Start'] | df['Prev_Is_End']
    df.loc[df.index[0], 'New_Possession'] = True
    df['Possession_ID'] = df['New_Possession'].cumsum()

    possessions_raw = []
    for pid, group in df.groupby('Possession_ID'):
        owner = group['Possession Owner'].iloc[0]
        possessions_raw.append({
            'Possession_ID': pid,
            'Team': owner,
            'Raw_Start_Time': group['GameTime'].min(),
            'Raw_End_Time': group['GameTime'].max(),
            'End_Type': group.iloc[-1]['Type'],
            'End_Event': group.iloc[-1]['Name'],
            'Start_Event': group.iloc[0]['Name'],
            'Link': group.iloc[-1]['Youtube Link']
        })

    possessions_final = []
    for i, p in enumerate(possessions_raw):
        if i == 0:
            start_time = p['Raw_Start_Time']
        else:
            prev_p = possessions_raw[i - 1]
            if prev_p['End_Event'] == 'Game End':
                start_time = p['Raw_Start_Time']
            else:
                start_time = prev_p['Raw_End_Time']
        end_time = p['Raw_End_Time']

        if p['End_Type'] == 'Try':
            cat = 'Try'
        elif p['End_Type'] == 'Penalty':
            cat = 'Penalty'
        elif p['End_Event'] == '6th Touch':
            cat = '6th Touch'
        else:
            cat = 'Other'

        possessions_final.append({
            'Team': p['Team'],
            'Start': start_time,
            'End': end_time,
            'Duration_Sec': (end_time - start_time).total_seconds(),
            'End_Type': p['End_Type'],
            'End_Event': p['End_Event'],
            'Color_Category': cat,
            'Link': p['Link']
        })

    poss_df = pd.DataFrame(possessions_final)

    # Visualization
    # Using specific colors to ensure visibility
    poss_domain = ['Try', 'Penalty', '6th Touch', 'Other']
    poss_range = ['Green', 'Red', 'Gold', 'Orange']

    poss_chart = alt.Chart(poss_df).mark_bar().encode(
        x=alt.X('Start:T', axis=alt.Axis(format='%H:%M:%S', title='Game Time')),
        x2='End:T',
        y=alt.Y('Team:N', title='Possession Owner'),
        color=alt.Color('Color_Category:N', scale=alt.Scale(domain=poss_domain, range=poss_range), title='Outcome'),
        href='Link:N',
        tooltip=[
            alt.Tooltip('Start:T', format='%H:%M:%S', title='Start'),
            alt.Tooltip('End:T', format='%H:%M:%S', title='End'),
            'Team:N',
            'End_Event:N',
            'End_Type:N',
            alt.Tooltip('Duration_Sec:Q', format='.1f', title='Duration (s)')
        ]
    ).properties(
        title='Possession Timeline',
        width=chart_width,
        height=200
    )

    # --- 5. Combine Charts ---
    final_chart = alt.vconcat(tries_chart, pen_chart, poss_chart).resolve_scale(color='independent') # .resolve_scale(x='shared')

    # --- 6. Metrics Table ---
    metrics = []
    for team, group in poss_df.groupby('Team'):
        num_possessions = len(group)
        num_tries = len(group[group['End_Type'] == 'Try'])
        success_rate = (num_tries / num_possessions) * 100 if num_possessions > 0 else 0
        completions = len(group[(group['End_Type'] == 'Try') | (group['End_Event'] == '6th Touch')])
        avg_time = group['Duration_Sec'].mean()
        total_time = group['Duration_Sec'].sum()

        metrics.append({
            'Team': team,
            'Possessions': num_possessions,
            'Tries': num_tries,
            'Success Rate (%)': f"{success_rate:.1f}%",
            'Completions': completions,
            'Avg Time (s)': f"{avg_time:.1f}",
            'Total Time (s)': f"{total_time:.1f}"
        })

    metrics_df = pd.DataFrame(metrics)

    # --- 7. Save HTML ---
    chart_json = final_chart.to_json()
    table_html = metrics_df.to_html(index=False, classes='table', border=0)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
      <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        .table {{
            font-family: Arial, Helvetica, sans-serif;
            border-collapse: collapse;
            width: {chart_width}px; /* Match chart width */
            margin: 20px auto;
        }}
        .table td, .table th {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        .table tr:nth-child(even){{background-color: #f2f2f2;}}
        .table tr:hover {{background-color: #ddd;}}
        .table th {{
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: left;
            background-color: #04AA6D;
            color: white;
        }}
        h2 {{ text-align: center; }}
        #vis {{ width: 100%; display: flex; justify-content: center; }}
      </style>
    </head>
    <body>
      <div id="vis"></div>
      <h2>Team Statistics</h2>
      {table_html}
      <script>
        var spec = {chart_json};
        vegaEmbed('#vis', spec);
      </script>
    </body>
    </html>
    """

    with output_html_path.open('w') as f:
        f.write(html_content)

def generate_htmls():
    data_paths = sorted([_ for _ in DATA_ROOT.glob("*.csv") if "parsed" not in _.stem])

    # Create a single final parsed csv with all the data:

    full_data_df = []
    for i, data_path in enumerate(data_paths):
        data_df = utils.load_data(data_path, simple=True)
        simple_data_df = data_df[
            [
                "Time",
                "Type",
                "Name",
                "Comment",
                "Youtube Link",
                "Action Owner",
                "Year",
                "Division",
                "Competition",
            ]
        ]
        simple_data_df["game"] = data_path.stem
        full_data_df.append(simple_data_df)

    full_data_df = pd.concat(full_data_df, ignore_index=True, axis=0)
    full_data_df.to_csv(ASSETS_ROOT / "all_events.csv")

    # Delete all previous renders
    for html_path in ROOT.glob("game_*.html"):
        rich.print(f"[yellow] Removing old html: {html_path} [/yellow]")
        html_path.unlink()

    games_data = []
    for i, data_path in enumerate(data_paths):
        rich.print(data_path)
        try:
            output_html_path = ROOT / f"game_{len(games_data)}.html"
            local_team_name, other_team_name = utils.get_names(data_path)
            year, division_name, competition_name = utils.get_year_division_competition(
                data_path
            )
            data_df = utils.load_data(data_path)
            data_df.to_csv(DATA_ROOT / f"{data_path.stem}_parsed.csv")
            if utils.is_fully_analysable(data_df):
                generate_html_from_df(data_df, output_html_path, team_names=(local_team_name, other_team_name))

                name = f"{local_team_name} vs {other_team_name}"
                if competition_name != "unknown":
                    name += f" ({year} {division_name} {competition_name})"

                games_data.append(
                    dict(
                        name=name,
                        file=output_html_path.name,
                    )
                )
        except Exception as e:
            rich.print(f"[red]Skipping {data_path.stem} plots due to {e}[/red]")
            raise e
    utils.save_json(games_data, ROOT / "games.json")


if __name__ == "__main__":
    generate_htmls()
