import glob
import sys
import argparse
import hsbc_hkg_statement
import pandas as pd

parser = argparse.ArgumentParser(description='Parse HSBC HK eStatements')
parser.add_argument('--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='CSV File for Output')
parser.add_argument('files', type=str, nargs='+', help='CSV File for Output')

args = parser.parse_args()

df_list = []
files = []
for file in args.files:
    files.extend(glob.glob(file))

for filename in files:
    print "Parsing {}".format(filename)
    df = hsbc_hkg_statement.pdf_to_transaction_table(filename)
    df_list.append(df)

df = pd.concat(df_list)
df = df.sort_values(["Account", "Date"])

if args.outfile != sys.stdout:
    print "Saving CSV to {}".format(args.outfile.name)
df.to_csv(args.outfile, index=False)