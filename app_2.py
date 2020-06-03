import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import Flask

from bi_grams import return_bi_grams
from dash.dependencies import Input, Output



blackbold = {'color': 'black', 'font-weight': 'bold'}
colors = {
    'background': '#FDFEFE',
    'text': '#111111',
    'textAlign': 'center'
}

df = pd.read_csv('20news_group_dataset.csv')
server = flask.Flask(__name__)
app_2 = dash.Dash(__name__, server=server)

app_2.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('Twimintel Dashboard',
                    style={
                        'textAlign': 'center',
                        'color': '#111111'
                    }),
        ], className='container'),
        html.Div([
            dcc.Slider(
                id='data-frac',
                max=1,
                min=0.01,
                step=0.01,
                value=0.01,
                marks={
                    0.01: '1%',
                    0.5: '50%',
                    1: '100%'
                }
            )
        ], className='five columns', style={'width': '48%'}),
        html.Div([
            dcc.Slider(
                id='word_count'
            ),
            html.Div(id='slider_text')
        ], className='five columns',
            style={'width': '48%'})
    ], className='row'),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='distribution',
                        style={"background-image": 'url("/assets/Group-1735.png")'}
                    )
                ], className='six columns'),
                html.Div([
                    dcc.Graph(
                        id='bigram'
                    )
                ], className='six columns'),
            ], className='row'),
            html.Div([
                html.Br(),
                html.Label(['Website:'], style=blackbold),
                html.Pre(id='web_link', children=[],
                         style={'white-space': 'pre-wrap', 'word-break': 'break-all',
                                'border': '1px solid black', 'text-align': 'center',
                                'padding': '12px 12px 12px 12px', 'color': 'blue',
                                'margin-top': '1px'},
                         className='three columns'
                         ),
                html.Div([
                    dcc.Markdown("""**Document Overview**"""),
                    html.Pre(id='hover-data', className='four columns')
                ], className='columns'),
                html.Div([
                    dcc.Graph(id='senti')
                ], style={'width': '55%',
                          'position': 'absolute',
                          'right': '5px',
                          'height': '120px',
                          'border': '3px'}),
            ], className='columns')
        ])
    ])
])


@app_2.callback(
    [Output('word_count', 'marks'),
     Output('word_count', 'min'),
     Output('word_count', 'max'),
     Output('word_count', 'value')],
    [Input('data-frac', 'value')])
def update_slider(val):
    df1 = df.sample(frac=val)
    minimum = df1['w_count'].min()
    maximum = df1['w_count'].max()
    value = int((maximum - minimum) / 8)
    marks = {
        int(minimum): str(minimum),
        '10000': '10000',
        int(maximum): str(maximum)
    }
    return marks, minimum, maximum, value


@app_2.callback(
    Output('slider_text', 'children'),
    [Input('word_count', 'value')])
def slid_text(val):
    if val:
        return "Word limit set as {} words".format(val)
    else:
        return "Waiting for value"


@app_2.callback(
    [Output('distribution', 'figure'),
     Output('senti', 'figure')],
    [Input('data-frac', 'value'),
     Input('word_count', 'value')])
def fraction_data(val1, wc):
    df1 = df.sample(frac=val1)
    df1 = df1[df1['w_count'] < wc]
    figure = {
        'data': [{
            'x': df1['date'],
            'y': df1['w_count'],
            'text': df1['doc_no'],
            'customdata': df1['URL'],
            'hovermode': 'closest',
            'mode': 'markers',
            'marker': {
                'size': df1['w_count'],
                'sizemode': 'area',
                'sizeref': 2. * max(df1['w_count']) / (40. ** 2),
                'sizemin': 4}
        }],
        'layout': {'plot_bgcolor': colors['background'],
                   'paper_bgcolor': colors['background'],
                   "background-image": 'url("/assets/Group-1735.png")',
                   'font': {
                       'family': 'Courier New',
                       'size': '12',
                       'color': '#7f7f7f'},
                   'title': 'Document Plot',
                   'xaxis': {
                       'title': 'Year of publishing'
                   },
                   'yaxis': {
                       'title': 'Word Count'}
                   }
    }

    df1['sentiments'] = np.random.uniform(low=-1.0, high=1.0, size=(len(df1)))
    col_ar = []
    for a in df1['sentiments']:
        if a > 0.5:
            col_ar.append('green')
        elif a < -0.5:
            col_ar.append('red')
        else:
            col_ar.append('blue')
    df1['colors'] = col_ar
    fig = go.Figure(
        data=[go.Bar(x=df1['doc_no'],
                     y=df1['sentiments'],
                     marker_color=df1['colors'])],
        layout={'yaxis':
                    {'title': 'Sentiment Score'},
                'plot_bgcolor': 'white'}
    )
    return figure, fig


@app_2.callback(
    Output('web_link', 'children'),
    [Input('distribution', 'clickData')])
def display_click_data(clickData):
    if clickData is None:
        return 'Click on any bubble'
    else:
        the_link = clickData['points'][0]['customdata']
        if the_link is None:
            return 'No Website Available'
        else:
            return html.A(the_link, href=the_link, target="_blank")


@app_2.callback(
    Output('hover-data', 'children'),
    [Input('distribution', 'hoverData')])
def display_hover_data(hoverData):
    if hoverData is None:
        return 'Hover over the graph to overview doc'
    else:
        val_tem = hoverData['points'][0]['text']
        temp = df.copy()
        temp = temp[temp['doc_no'] == val_tem]
        return temp['text']


@app_2.callback(
    Output('bigram', 'figure'),
    [Input('distribution', 'hoverData')])
def display_bigrams(hoverData):
    if hoverData is None:
        trace = go.Bar(
            x=[0],
            y=[0]
        )
        figure = go.Figure(data=[trace],
                           layout_title_text="Hover over data to see bigram distribution")
        return figure
    else:
        val_tem = hoverData['points'][0]['text']
        temp = df.copy()
        tem = temp[temp['doc_no'] == val_tem]
        te = tem['text'].values[0]
        bi_g = return_bi_grams(te)

        bigram_list = []
        frequ = []

        for words in bi_g:
            bigram, num = words
            w2, w1 = bigram
            te = w1 + ' ' + w2
            bigram_list.append(te)
            frequ.append(num)

        bi_data = pd.DataFrame(zip(bigram_list, frequ), columns=['bigram', 'count'])

        trace = go.Bar(
            x=bi_data['bigram'],
            y=bi_data['count'])
        figure = go.Figure(data=[trace],
                           layout={
                               'xaxis': {'title': 'bigrams'},
                               'yaxis': {'title': 'frequency of appearance'},
                               'plot_bgcolor': 'white'
                           },
                           layout_title_text="Top 10 bigrams in {}".format(val_tem))
        return figure


if __name__ == '__main__':
    app_2.run_server(debug=False)
