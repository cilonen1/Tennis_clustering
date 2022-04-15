import argparse
from clustering import full_cycle, count_sum

parser = argparse.ArgumentParser(description='Collect continent and year')
parser.add_argument('continent', type=str)
parser.add_argument('year', type=str)
args = parser.parse_args()
odds_df = full_cycle(args.continent, args.year)
odds_df.to_excel(f'{args.continent}(f).xlsx')