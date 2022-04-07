from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import seaborn as sns
from kmeans import KMeans as kmeans

# data = pd.read_csv('nfl_2021_pbp.csv.gz', compression='gzip', low_memory=False)
#
#
# # Analyze
# ## whats better, rush or pass on 1st down
#
# passes = data.loc[(data['pass'] == 1) & (data['down'] == 1)]
# rushes = data.loc[(data['pass'] == 1) & (data['down'] == 1)]
#
# print(passes.iloc[:2, :25])
# print(list(passes.columns))




class NFLVRS:
    COLORS = {'ARI': '#97233F', 'ATL': '#A71930', 'BAL': '#241773', 'BUF': '#00338D', 'CAR': '#0085CA',
              'CHI': '#00143F', 'CIN': '#FB4F14', 'CLE': '#FB4F14', 'DAL': '#B0B7BC', 'DEN': '#002244',
              'DET': '#046EB4', 'GB': '#24423C', 'HOU': '#C9243F', 'IND': '#003D79', 'JAX': '#136677',
              'KC': '#CA2430', 'LA': '#002147', 'LAC': '#2072BA', 'LV': '#C4C9CC', 'MIA': '#0091A0',
              'MIN': '#4F2E84', 'NE': '#0A2342', 'NO': '#A08A58', 'NYG': '#192E6C', 'NYJ': '#203731',
              'PHI': '#014A53', 'PIT': '#FFC20E', 'SEA': '#7AC142', 'SF': '#C9243F', 'TB': '#D40909',
              'TEN': '#4095D1', 'WAS': '#FFC20F'}

    def __init__(self, start_year=None, end_year=None, plt_theme='default'):
        if end_year is None:
            end_year = datetime.today().year
        if start_year is None:
            start_year = datetime.today().year

        self.df = self._load_data(start_year, end_year)

        plt.style.use(plt_theme)

    def _load_data(self, start_year, end_year):
        df_list = []
        for year in range(start_year, end_year):
            df = pd.read_csv(f'nfl_{str(year)}_pbp.csv.gz', compression='gzip', low_memory=False)
            df_list.append(df)
        df = pd.concat(df_list)
        return df

    def _isolate_players(self, players):
        return self.df[(self.df['passer'].isin(players)) |
                       (self.df['rusher'].isin(players)) |
                       (self.df['receiver'].isin(players))].copy()

    def _isolate_passers(self, players):
        return self.df[(self.df['passer'].isin(players))].copy()

    def _isolate_rushers(self, players):
        return self.df[(self.df['rusher'].isin(players))].copy()

    def _isolate_receivers(self, players):
        return self.df[(self.df['receiver'].isin(players))].copy()

    def _map_color(self):
        self.df['color'] = self.df['posteam'].replace(NFLVRS.COLORS)

    def _demean_col(self, col):
        self.df[col] = self.df[col].sub(self.df[col].mean(), fill_value=0)

    def filter(self, passers=[], rushers=[], receivers=[],
               min_pass_attempts=None, min_completions=0,
               min_rush_attempts=0,
               min_targets=0, min_receptions=0,
               all=False):
        if isinstance(passers, str):
            passers = list(passers)
        if isinstance(rushers, str):
            rushers = list(rushers)
        if isinstance(receivers, str):
            receivers = list(receivers)

        if all:
            self.df = self.df[(self.df[passers].isin(passers)) &
                              (self.df[rushers].isin(rushers)) &
                              (self.df[receivers].isin(receivers)) &
                              (self.df.groupby('passer')['passer'].transform('size') >= min_pass_attempts) &
                              (self.df.groupby('rusher')['rusher'].transform('size') >= min_rush_attempts) &
                              (self.df.groupby('receiver')['receiver'].transform('size') >= min_targets)] # &
                              # (self.df.groupby('passer')['passer'].transform('size') >= min_pass_attempts) &
                              # (self.df.groupby('rusher')['rusher'].transform('size') >= min_rush_attempts) &
                              # (self.df.groupby('receiver')['receiver'].transform('size') >= min_targets)]
        else:
            self.df = self.df[(self.df[passers].isin(passers)) |
                              (self.df[rushers].isin(rushers)) |
                              (self.df[receivers].isin(receivers)) |
                              (self.df.groupby('passer')['passer'].transform('size') >= min_pass_attempts) |
                              (self.df.groupby('rusher')['rusher'].transform('size') >= min_rush_attempts) |
                              (self.df.groupby('receiver')['receiver'].transform('size') >= min_targets) |
                              (self.df.groupby('passer')['passer'].transform('size') >= min_pass_attempts) |
                              (self.df.groupby('rusher')['rusher'].transform('size') >= min_rush_attempts) |
                              (self.df.groupby('receiver')['receiver'].transform('size') >= min_targets)]

    def _calculate_game_num(self):
        self.df['game_num'] = self.df.groupby(['game_id', 'passer']).cumcount()+1

    def _draw_scatter_plot(self, player, data, ax, x_col, y_col, parent_kwargs, trendline=True, **kwargs):
        label = None if kwargs.get('color', None) == 'grey' else player
        ax.scatter(x=data[x_col], y=data[y_col], label=label, c=kwargs.get('color', data['color']), alpha=kwargs.get('alpha', 1))
        if trendline:
            z = np.polyfit(data[x_col], data[y_col], parent_kwargs.get('n', 1))
            p = np.poly1d(z)
            ax.plot(data[x_col], p(data[x_col]), color=kwargs.get('color', data['color'].unique()[0]))

    def plot_epa_per_game(self, players=[], show_league=True, min_throws=0, x='game_num', subplots=False, **kwargs):
        """ Plots qb epa per game """

        epa = 'qb_epa'

        # self.filter(min_pass_attempts=min_throws)

        self.df = self.df[self.df.groupby('passer')['passer'].transform('size') >= min_throws]
        self._map_color()
        if show_league:
            league_grouped = self.df.groupby(['game_id', 'passer']).agg({epa: 'mean', 'color': 'first'}).reset_index()
            league_grouped['game_num'] = league_grouped.groupby('passer').cumcount()+1
            league_grouped.reset_index(inplace=True)

        # self._demean_col(epa)
        if len(players) > 0:
            isolated_players = self._isolate_passers(players)
            grouped = isolated_players.groupby(['game_id', 'passer']).agg({epa: 'mean', 'color': 'first'}).reset_index()
            grouped['game_num'] = grouped.groupby('passer').cumcount() + 1
            grouped.reset_index(inplace=True)

        if not subplots:
            fig, ax = plt.subplots(1, 1)

            if show_league:
                for player, data in league_grouped.groupby('passer'):
                    self._draw_scatter_plot(player, data, ax, x, epa, kwargs, trendline=False, color='grey', alpha=0.6)

            if len(players) > 0:
                for player, data in grouped.groupby('passer'):
                    self._draw_scatter_plot(player, data, ax, x, epa, kwargs, n=2)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set the x-axis to integers
            ax.legend(loc=kwargs.get('legend_loc', 'best'), bbox_to_anchor=(1.01, 0.8), ncol=1, fontsize='x-small')
            ax.grid()
            ax.set_title('Average EPA per game')

        else:
            ncols = kwargs.get('ncols', 2)
            nrows = int(len(players) / ncols)
            fig, axs = plt.subplots(nrows, ncols, figsize=(9, 6), sharey=True, squeeze=False)

            for (player, data), ax in zip(grouped.groupby('passer'), axs.ravel()):
                ax.set_yticks(np.arange(-1.25, 1.25, 0.25))
                self._draw_scatter_plot(player, data, ax, x, epa, kwargs)

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set the x-axis to integers
                ax.legend(loc=kwargs.get('legend_loc', 'best'), fontsize='x-small')
                ax.set_title(f'{player} Average EPA per game', fontsize=8)

                ax.grid(which='major')

            fig.suptitle('Average EPA per Game')
            fig.supylabel('Average EPA')
            fig.supxlabel('Game Number')
        fig.tight_layout()
        fig.show()

    def plot_epa_vs_cpoe(self, players, show_league, **kwargs):
        epa = 'qb_epa'

        if show_league:
            league_grouped = self.df.groupby(['season', 'passer']).agg({epa: 'mean', 'cpoe':'mean', 'color': 'first'}).reset_index()
            league_grouped['game_num'] = league_grouped.groupby('passer').cumcount()+1
            league_grouped.reset_index(inplace=True)

        # self._demean_col(epa)
        if len(players) > 0:
            isolated_players = self._isolate_passers(players)
            grouped = isolated_players.groupby(['season', 'passer']).agg({epa: 'mean', 'cpoe':'mean', 'color': 'first'}).reset_index()
            grouped['game_num'] = grouped.groupby('passer').cumcount() + 1
            grouped.reset_index(inplace=True)

        fig, ax = plt.subplots(1, 1)

        if show_league:
            for player, data in league_grouped.groupby('passer'):
                self._draw_scatter_plot(player, data, ax, 'cpoe', epa, kwargs, trendline=False, color='grey', alpha=0.6)

        if len(players) > 0:
            for player, data in grouped.groupby('passer'):
                self._draw_scatter_plot(player, data, ax, 'cpoe', epa, kwargs, trendline=False)

        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set the x-axis to integers
        ax.legend(loc=kwargs.get('legend_loc', 'best'), bbox_to_anchor=(1.01, 0.8), ncol=1, fontsize='x-small')
        ax.grid()

        ax.set_title('Average EPA per game')

        fig.supylabel('Average EPA')
        fig.supxlabel('CPOE')

        fig.tight_layout()
        fig.show()

    def tier_qbs(self, k, min_throws=400):
        """ K-Means algo to separate qbs into k tiers. """

        self.df = self.df[self.df.groupby('passer')['passer'].transform('size') >= min_throws]



        league_grouped = self.df.groupby(['passer']).agg({'qb_epa': 'mean', 'cpoe': 'mean'}).reset_index()
        league_grouped['game_num'] = league_grouped.groupby('passer').cumcount() + 1
        league_grouped.reset_index(inplace=True)
        # league_grouped = league_grouped[~((league_grouped['passer'] == 'C.Keenum') & (league_grouped['season'] == 2020)
        #                                   | (league_grouped['passer'] == 'J.Brissett') & (league_grouped['season'] == 2020))]
        X = league_grouped[['cpoe', 'qb_epa']].values.tolist()
        labels = league_grouped['passer'] #+ ' ' + league_grouped['season'].astype('str')
        labels = labels.values.tolist()
        # k = kmeans.KMeans(X=X, labels=labels, k_clusters=k)
        k = kmeans(X=X, labels=labels, k_clusters=k)
        k.cluster()
        print(k.clusters)
        k.visualize_wcss()
        k.visualize()



    ## Compare effectiveness of rushes on first down with effectiveness of passing on first down





if __name__=="__main__":
    nflvrs = NFLVRS(start_year=2017, end_year=2022, plt_theme='Solarize_Light2')

    # nflvrs.plot_epa_per_game(['T.Brady', 'A.Rodgers',
    #                           'M.Jones', 'T.Lawrence',
    #                           'T.Tagovailoa', 'J.Herbert'], subplots=True, n=2)

    # nflvrs.plot_epa_per_game(players=['T.Brady', 'M.Jones'], subplots=True, min_throws=500, n=2, show_league=False)
    # nflvrs.plot_epa_per_game(players=['L.Jackson', 'P.Mahomes'], subplots=False, min_throws=500, n=1, show_league=True)
    nflvrs.plot_epa_per_game(players=['R.Wilson', 'D.Carr', 'J.Herbert', 'P.Mahomes'], subplots=False, min_throws=500, n=2, show_league=True)
    nflvrs.plot_epa_per_game(players=['R.Wilson', 'D.Carr', 'J.Herbert', 'P.Mahomes'], subplots=False, min_throws=500, n=2, show_league=False)
    nflvrs.plot_epa_per_game(players=['R.Wilson', 'D.Carr', 'J.Herbert', 'P.Mahomes'], subplots=True, min_throws=500, n=2, show_league=False)
    nflvrs.plot_epa_vs_cpoe(players=['R.Wilson', 'D.Carr', 'J.Herbert', 'P.Mahomes'], min_throws=500, show_league=False)
    # nflvrs.tier_qbs(k=5)
