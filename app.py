from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np

app = Dash(__name__)

colors = {
    'background': '#232b2b',
    'text': '#FFFFFF'
}

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x='Fruit', y='Amount', color='City', barmode='group')
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
)


def _draw_scatter_plot(player, data, ax, x_col, y_col, parent_kwargs, trendline=True, **kwargs):
    label = None if kwargs.get('color', None) == 'grey' else player
    ax.scatter(x=data[x_col], y=data[y_col], label=label, c=kwargs.get('color', data['color']),
               alpha=kwargs.get('alpha', 1))
    if trendline:
        z = np.polyfit(data[x_col], data[y_col], parent_kwargs.get('n', 1))
        p = np.poly1d(z)
        ax.plot(data[x_col], p(data[x_col]), color=kwargs.get('color', data['color'].unique()[0]))


def plot_epa_per_game(players=[], show_league=True, min_throws=0, x='game_num', subplots=False, **kwargs):
    """ Plots qb epa per game """

    epa = 'qb_epa'

    # self.filter(min_pass_attempts=min_throws)

    self.df = self.df[self.df.groupby('passer')['passer'].transform('size') >= min_throws]
    self._map_color()
    if show_league:
        league_grouped = self.df.groupby(['game_id', 'passer']).agg({epa: 'mean', 'color': 'first'}).reset_index()
        league_grouped['game_num'] = league_grouped.groupby('passer').cumcount() + 1
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


app.layout = html.Div(style={'backgroundColor': colors['background']},
                      children=[
                          html.H1(children='Welcome',
                                  style={
                                      'textAlign': 'center',
                                         'color': colors['text']
                                         }
                                  ),

                          html.Div(children="Dash: A web application framework for your data.",
                                   style={
                                       'textAlign': 'center',
                                       'color': colors['text']
                                        }
                                   ),

                      dcc.Graph(
                          id='example-graph',
                          figure=fig
                            )
                      ])

if __name__=='__main__':
    app.run_server(debug=True)
