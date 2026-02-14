import pandas as pd
from jinja2 import Template
import rich

from touch_rugby_analyzer.constants import ASSETS_ROOT, ROOT, DATA_ROOT
from touch_rugby_analyzer import utils


def generate_html():
    input_template_path = ASSETS_ROOT / "template.html"

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
                "Team",
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
            stats_dict = utils.get_stats_df(data_df, local_team_name, other_team_name)
            trend_fig = utils.make_fig_1(data_df, local_team_name, other_team_name)
            game_fig = utils.make_game_fig(data_df, local_team_name, other_team_name)

            plotly_jinja_data = {
                "game_fig": game_fig.to_html(full_html=False),
                "trend_fig": trend_fig.to_html(full_html=False),
                "stats_tries_table": stats_dict.get("Try", pd.DataFrame()).to_html(),
                "stats_penalties_table": stats_dict.get(
                    "Penalty", pd.DataFrame()
                ).to_html(),
                "stats_turnovers_table": stats_dict.get(
                    "Turnover", pd.DataFrame()
                ).to_html(),
                "stats_possessions_table": stats_dict.get(
                    "Possession", pd.DataFrame()
                ).to_html(),
            }
            # consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above

            with output_html_path.open("w", encoding="utf-8") as output_file:
                with input_template_path.open() as template_file:
                    j2_template = Template(template_file.read())
                    output_file.write(j2_template.render(plotly_jinja_data))

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
    utils.save_json(games_data, ROOT / "games.json")


if __name__ == "__main__":
    generate_html()
