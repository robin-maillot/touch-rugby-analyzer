import io
import shutil
import string
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import rich
from touch_rugby_analyzer.constants import ASSETS_ROOT, ROOT, DATA_ROOT

# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]
CREDENTIALS_PATH = Path(__file__).resolve().parent / "credentials.json"


def login_to_gcloud(
    token_path: Path = CREDENTIALS_PATH.parent / "token.json",
    service_name: str = "drive",
):
    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    if service_name == "drive":
        service = build(service_name, "v3", credentials=creds)
    elif service_name == "sheets":
        service = build(service_name, "v4", credentials=creds)
    else:
        raise Exception(f"{service_name = } does not exist")
    return service


def get_sheet_values(
    sheet_id: str, range: str, service=None, valueRenderOption: str = "FORMULA"
) -> list:
    if service is None:
        service = login_to_gcloud(
            Path(__file__).resolve().parent / "token.json", service_name="sheets"
        )
    # Call the Sheets API
    sheet = service.spreadsheets()
    result = (
        sheet.values()
        .get(
            spreadsheetId=sheet_id,
            range=range,
            valueRenderOption=valueRenderOption,
            dateTimeRenderOption="FORMATTED_STRING",
        )
        .execute()
    )
    return result.get("values", [])


def get_sheet_info(sheet_id: str, range: str, service=None) -> pd.DataFrame:
    if service is None:
        service = login_to_gcloud(
            Path(__file__).resolve().parent / "token.json", service_name="sheets"
        )
    # Call the Sheets API
    result = get_sheet_values(sheet_id=sheet_id, range=range, service=service)
    for n_columns, c_name in enumerate(result[2]):
        if c_name.lower() == "datasets.yaml":
            break
    result_values = [result_line[:n_columns] for result_line in result[2:]]
    column_names = []
    for c in result_values[0]:
        column_name = stringify(c)
        while column_name in column_names:
            column_name += "_copy"
        column_names.append(column_name)
    df = pd.DataFrame(result_values[1:], columns=column_names)
    df.replace("", None, inplace=True)
    df.dropna(how="all", inplace=True, axis=0)
    df.dropna(how="all", inplace=True, axis=1)
    df["material"] = range.split("!")[0]
    df["n_columns"] = n_columns
    return df


def stringify(s: str) -> str:
    return s.lower().rstrip().lstrip().replace(" ", "_").replace(".", "")


def link_to_id(link_str: str) -> str:
    return link_str.split("/")[-1].split("?")[0]


def formula_to_link(cell_value: str) -> Optional[str]:
    if cell_value.startswith("http"):
        return cell_value
    elif cell_value.startswith("=HYPERLINK("):
        parts = cell_value.split('"')
        assert len(parts) == 5 and parts[1].startswith("http")
        return parts[1]
    else:
        return None


def int_to_gsheet_value(n: int) -> str:
    s = ""
    while n > 0:
        m = (n - 1) % 26
        s = f"{string.ascii_uppercase[m]}{s}"
        n = (n - m) // 26
    return s


def upload_df_to_gsheet(
    spreadsheet_id: str,
    data: Union[pd.DataFrame, Path, str],
    sheet_name: str = "Sheet1",
    include_index: bool = False,
    overwrite: bool = False,
    service=None,
):
    if service is None:
        service = login_to_gcloud(
            Path(__file__).resolve().parent / "token.json", service_name="sheets"
        )

    if isinstance(data, (Path, str)):
        data = pd.read_csv(data, index_col=0)
    if include_index:
        data_to_upload = [[""] + list(data.columns)]
    else:
        data_to_upload = [list(data.columns)]

    def _stringify(value):
        if isinstance(value, np.ndarray):
            return str(list(value))
        elif isinstance(value, list):
            return str(value)
        else:
            return value

    for i, row in data.iterrows():
        row_data = []
        if include_index:
            row_data.append(_stringify(i))
        row_data += [_stringify(v) for v in row.to_dict().values()]
        data_to_upload.append(row_data)

    request = None
    creation_failed = False
    try:
        if sheet_name is not None:
            batch_update_values_request_body = {
                "requests": [{"addSheet": {"properties": {"title": sheet_name}}}]
            }
            request = (
                service.spreadsheets()
                .batchUpdate(
                    spreadsheetId=spreadsheet_id, body=batch_update_values_request_body
                )
                .execute()
            )
    except Exception as e:
        print(e)
        creation_failed = True

    if not creation_failed or overwrite:
        request = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=spreadsheet_id,
                range=sheet_name,
                valueInputOption="USER_ENTERED",
                body={"values": data_to_upload},
            )
            .execute()
        )
    return request


def create_gsheet(
    spreadsheet_name: str,
    parent_id: Optional[str] = None,
    service=None,
) -> Optional[str]:
    if service is None:
        service = login_to_gcloud(Path(__file__).resolve().parent / "token.json")
    file_metadata = {
        "name": spreadsheet_name,
        "parents": [parent_id],
        "mimeType": "application/vnd.google-apps.spreadsheet",
    }
    res = service.files().create(body=file_metadata).execute()
    return res.get("id")


def fetch_gsheet(
    sheet_id: str, sheet_tab: str, return_raw: bool = False
) -> Union[list, pd.DataFrame]:
    """Fetches values in a gsheet and returns them in a Pandas DataFrame. The number of columns in
    the dataframe is dictated by the number of names columns in the gsheet. A named column must have text in it.
        Args:
            sheet_id: code in the url of the sheet, see default example
            sheet_tab: tab name of the sheet
            return_raw: if true, return raw data as list
        Returns:
            pandas DataFrame
    """
    result = get_sheet_values(
        sheet_id=sheet_id,
        range=sheet_tab,
        valueRenderOption="FORMATTED_VALUE",
    )
    if return_raw:
        return result

    column_names = result.pop(0)
    result_with_empty = []
    for r in result:
        if len(r) > len(column_names):
            result_with_empty.append(r[: len(column_names)])
        else:
            r = r + [""] * int(len(column_names) - len(r))
        result_with_empty.append(r)
    df = pd.DataFrame(result_with_empty, columns=column_names)
    return df


if __name__ == "__main__":
    sheet_id = "1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k"
    for sheet_tab, csv_name in [
        ("France/England", "france_england.csv"),
        ("France/Pays-Bas", "france_pays-bas.csv"),
    ]:
        _ = fetch_gsheet(sheet_id, sheet_tab, return_raw=False)
        _.to_csv(DATA_ROOT / csv_name, index=False)
