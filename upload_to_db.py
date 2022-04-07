
from os import path
import pandas as pd
import sys

sys.path.append(path.abspath('../../PycharmProjects'))
from DBConnector.connector import SQLServer


def dtype_mapping():
    return {'object': 'TEXT',
        'int64': 'INT',
        'float64': 'NVARCHAR(100)',
        'datetime64': 'DATETIME',
        'bool': 'TINYINT',
        'category': 'TEXT',
        'timedelta[ns]': 'TEXT'}

dmap = dtype_mapping()

data = pd.read_csv(f'nfl_2011_pbp.csv.gz', compression='gzip', low_memory=False)
data.drop(columns=['id', 'desc'], inplace=True)
data.columns = [hdr.replace('_', ' ').title().replace(' ', '') for hdr in data.columns]

data['Year'] = data['GameId'].apply(lambda x: int(x.split('_')[0]))

hdrs = data.dtypes.index
cmap = [(hdr, dmap[str(data[hdr].dtype)]) for hdr in hdrs if hdr != 'desc']
# print(cmap)
db = SQLServer('DESKTOP-DSDLA90', 'Football')
db.create_table('PlayByPlay', ('Id', 'int'),
                                *cmap,
                auto_increment=True)

for YEAR in range(2000, 2022):
    print(YEAR)
    data = pd.read_csv(f'https://github.com/guga31bb/nflfastR-data/blob/master/data/play_by_play_{str(YEAR)}.csv.gz?raw=True',
                   compression='gzip', low_memory=False)
    data.drop(columns=['id', 'desc'], inplace=True)
    data.columns = [hdr.replace('_', ' ').title().replace(' ', '') for hdr in data.columns]
    data['Year'] = data['GameId'].apply(lambda x: int(x.split('_')[0]))
    db.update_table('FootballTest', data)


