from touch_rugby_analyzer import gcloud_utils, generate_html_v2
from touch_rugby_analyzer.constants import DATA_ROOT
import rich
import subprocess
import datetime

if __name__ == "__main__":
    sheet_id = "1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k"
    sheet_names = gcloud_utils.get_all_sheet_names(sheet_id)

    for csv_path in DATA_ROOT.glob("*.csv"):
        rich.print(f"[yellow] Removing old csv: {csv_path} [/yellow]")
        csv_path.unlink()

    for sheet_tab in sheet_names:
        csv_name = gcloud_utils.sheet_name_to_csv_name(sheet_tab)
        rich.print(f"{sheet_tab:>20} -> {csv_name}")
        _ = gcloud_utils.fetch_gsheet(sheet_id, sheet_tab, return_raw=False)
        _.to_csv(DATA_ROOT / csv_name, index=False)

    generate_html_v2.generate_html()

    for cmd in [
        ["git", "add", "-u"],
        ["git", "commit", "-m", f"update html {datetime.datetime.now()}"],
        ["git", "push"]
    ]:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = process.communicate()[0]
        rich.print(output)