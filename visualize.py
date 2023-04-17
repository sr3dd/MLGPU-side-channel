import sys

import pandas as pd

# Graphing module imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_data(file_paths, output_path):
    dfs = []
    
    for file in file_paths:
        df = pd.read_csv(file, 
                         header=None, parse_dates = [1], 
                         sep = '\t', names = ['power', 'time'])
        dfs.append(df)
        
    pieces = {}
    
    for i, df in enumerate(dfs):
        pieces[f'{file_paths[i]}'] = df

    df_concat = pd.concat(pieces, names = ['source'])
    df_concat = df_concat.reset_index(level=0)
    
    fig = px.line(df_concat, x='time', y='power', color='source')
    fig.update_xaxes(rangeslider_visible=True)
    fig.write_html(output_path)


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print("Requires at least 2 arguments: input file path and output file path")

    plot_data(sys.argv[1:-1], sys.argv[-1])
