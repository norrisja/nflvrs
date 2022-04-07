import pandas as pd

# YEAR = 2021
data_list = []
for YEAR in range(2010, 2016):
    print(YEAR)
    data = pd.read_csv(f'https://github.com/guga31bb/nflfastR-data/blob/master/data/play_by_play_{str(YEAR)}.csv.gz?raw=True',
    compression='gzip', low_memory=False)
    data.to_csv(f'nfl_{str(YEAR)}_pbp.csv.gz', compression='gzip', index=False)

    # data_list.append(data)



# data = pd.concat(data_list)


