import pandas as pd
from jinja2 import Template

from touch_rugby_analyzer.constants import ASSETS_ROOT, ROOT, DATA_ROOT
from touch_rugby_analyzer import utils

input_template_path = ASSETS_ROOT / "template.html"

for i, data_path in enumerate(
    [
        DATA_ROOT / "france_pays-bas.csv",
        DATA_ROOT / "france_england.csv",
    ]
):
    output_html_path = ROOT / f"game_{i}.html"
    local_team_name, other_team_name = utils.get_names(data_path)
    data_df = utils.load_data(data_path, local_team_name, other_team_name)
    data_df.to_csv(DATA_ROOT / f"{data_path.stem}_parsed.csv")
    stats_dict = utils.get_stats_df(data_df, local_team_name, other_team_name)
    trend_fig = utils.make_fig_1(data_df, local_team_name, other_team_name)
    game_fig = utils.make_game_fig(data_df, local_team_name, other_team_name)

    plotly_jinja_data = {
        "game_fig": game_fig.to_html(full_html=False),
        "trend_fig": trend_fig.to_html(full_html=False),
        "stats_tries_table": stats_dict.get("Try", pd.DataFrame()).to_html(),
        "stats_penalties_table": stats_dict.get("Penalty", pd.DataFrame()).to_html(),
        "stats_turnovers_table": stats_dict.get("Turnover", pd.DataFrame()).to_html(),
        "stats_possessions_table": stats_dict.get("Possession", pd.DataFrame()).to_html(),
    }
    # consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above

    with output_html_path.open("w", encoding="utf-8") as output_file:
        with input_template_path.open() as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render(plotly_jinja_data))
