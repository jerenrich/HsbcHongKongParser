from pdfminer_utils import *
import dateutil
import datetime
import re

footer = "The Hongkong and Shanghai Banking Corporation Limited"
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
header = "{} {} {}".format("\d{1,2}", "|".join(months), "\d{4}")
ccy_default = "HKD"


def ser_carry_foward(ser):
    ser = ser.copy()
    for i in range(1, len(ser)):
        if (ser.iloc[i] is None) or (ser.iloc[i] == ""):
            ser.iloc[i] = ser.iloc[i-1]
    return ser


def collapse_rows(df, data_cols):
    """Collapse rows where the data_cols are all None,
    until a row with data is found
    """

    # Find the rows where the data columns are all Nones
    row_group = []
    prev_row = 0
    for i in range(len(df)):
        if not df[data_cols].iloc[i].isnull().all():
            row_group.append(prev_row)
            prev_row = i + 1
        else:
            row_group.append(prev_row)
    df["RowGroup"] = row_group

    def combine_fn(vals):
        """If single unique val, return it, otherwise return joined str"""
        vals = filter(lambda x: x is not None, vals)
        vals_set = set(vals)
        if len(vals_set) == 1:
            return vals_set.pop()
        else:
            return " ".join(vals)

    fn_dict = {col: combine_fn for col in df.columns}
    df2 = df.groupby(["RowGroup"])[["Transaction Details"] + data_cols].agg(fn_dict)
    cols = ["Date", "Transaction Details"] + data_cols
    if "CCY" in df.columns:
        cols = cols + ["CCY"]
    df2 = df2[cols]
    df2 = df2.reset_index(drop=True)
    return df2


def get_major_sections(pages):
    """Find section titles based on text in boxes
    :param layouts:
    :return:
    """

    headers = get_text_in_boxes(pages)
    sections = sectionize(pages, headers)
    return sections


def parse_date(date_txt, statement_date):
    sd = statement_date

    date = dateutil.parser.parse(date_txt, default=sd)
    if date > statement_date:
        sd2 = datetime.date(sd.year - 1, sd.month, sd.day)
        date = dateutil.parser.parse(date_txt, default=sd2)

    return date


def get_transaction_table(sections, statement_date):
    history_header = "Account Transaction History"
    transactions = single_val(filter(lambda x: re.search(history_header, x["Text"]), sections))

    accounts = get_underlined_text(transactions["Pages"])
    accounts_trans = sectionize(transactions["Pages"], accounts)

    dfs = list()
    for trans in accounts_trans:
        for page in trans["Pages"]:
            df = get_table(page)
            if df is not None:
                for col in ["CCY", "Date"]:
                    if col in df.columns:
                        df[col] = ser_carry_foward(df[col])
                    data_cols = ["Deposit", "Withdrawal", "Balance"]
                    df = collapse_rows(df, data_cols)

                df["Account"] = trans["Text"]
                if "CCY" not in df.columns:
                    df["CCY"] = ccy_default
                dfs.append(df)

    df = pd.concat(dfs)
    df["Date"] = df["Date"].apply(lambda x: parse_date(x, statement_date))
    return df


def get_statement_date(layouts):
    objs = layouts[0]._objs
    date_objs = find_text(objs, header)
    date_obj = get_top_most(date_objs)
    date_txt = date_obj._objs[0].get_text().strip()
    date = dateutil.parser.parse(date_txt).date()
    return date


def pdf_to_transaction_table(filename):
    layouts = get_layouts(filename)
    date = get_statement_date(layouts)
    pages = get_pages_content(layouts, header, footer)
    sections = get_major_sections(pages)

    df = get_transaction_table(sections, date)

    col_order = ["Account", "Date", "Transaction Details", "Deposit", "Withdrawal", "Balance", "CCY"]
    df = df[col_order]

    return df
