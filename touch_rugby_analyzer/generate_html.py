import plotly.express as px
from jinja2 import Template
from touch_rugby_analyzer.constants import ASSETS_ROOT, ROOT

data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(data_canada, x='year', y='pop')

input_template_path=ASSETS_ROOT / "template.html"
output_html_path=ROOT / "index.html"

plotly_jinja_data = {"fig":fig.to_html(full_html=False)}
#consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above

with output_html_path.open("w", encoding="utf-8") as output_file:
    with input_template_path.open() as template_file:
        j2_template = Template(template_file.read())
        output_file.write(j2_template.render(plotly_jinja_data))