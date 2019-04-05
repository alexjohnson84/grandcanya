from datetime import datetime as dt
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from flask_caching import Cache
import plotly.graph_objs as go
import plotly.figure_factory as ff


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app.config.suppress_callback_exceptions = True

def extract_weekday(dt):
    weekday = dt.weekday() + 1
    if dt.weekday() == 6:
        weekday = 0
    return weekday
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 300

@cache.memoize(timeout=TIMEOUT)
def load_actuals():
    df = pd.read_csv('clean_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['size'] == 'Standard Size']
    df['day_of_week'] = df['date'].apply(lambda x: extract_weekday(x))
    df['week_of_year'] = df['date'].apply(lambda x: int(x.strftime('%U')))
    df['month'] = df['date'].apply(lambda x: x.month)
    df['odds'] = 1 / df['n_chances']
    return df

def actuals_dataframe():
    return load_actuals()

@cache.memoize(timeout=TIMEOUT)
def load_forecast():
    forecast = pd.read_csv('forecast.csv')
    forecast['date'] = pd.to_datetime(forecast['date'])
    forecast['day_of_week'] = forecast['date'].apply(lambda x: extract_weekday(x))
    forecast['week_of_year'] = forecast['date'].apply(lambda x: int(x.strftime('%U')))
    forecast['month'] = forecast['date'].apply(lambda x: x.month)
    forecast['day_of_year'] = forecast['date'].apply(lambda x: x.timetuple().tm_yday)
    return forecast

def forecast_dataframe():
    return load_forecast()

month_names = ['January', 'Febuary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

graph_style = {'padding': 10}

def plot_actuals(df, metric):
    dow_data = []
    included_dow = sorted(list(set(df['day_of_week'])))
    included_dow_names = [dow_names[i] for i in included_dow]
    for i in included_dow:
        df_sub = df.loc[df['day_of_week'] == i]
        dow_data.append(df_sub[metric].values)
    print([len(d) for d in dow_data])
    dist_plot = ff.create_distplot(dow_data, included_dow_names, curve_type='normal', show_hist=False, show_rug=False)
    dist_plot['layout'] = go.Layout(
        title='sample title',
        xaxis={'title':'x1'},
        yaxis={'title':'x2'},
        height=350,
    )
    return [
        dcc.Graph(
            id='actuals-graph',
            style=graph_style,
            figure=go.Figure(
                data=[
                    go.Scatter(
                        x = df['date'],
                        y = df[metric]
                    )
                ],
                layout=go.Layout(
                    title='sample title',
                    xaxis={'title':'x1'},
                    yaxis={'title':'x2'},
                    height=300
                )
            )
        ),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    style=graph_style,
                    figure=go.Figure(
                        data=[
                            go.Box(
                                y=df[metric],
                                x=df['day_of_week']
                            )
                        ],
                        layout=go.Layout(
                            title='sample title',
                            xaxis={'title':'x1'},
                            yaxis={'title':'x2'},
                            height=350
                        )
                    )
                )
            ], className='col-sm-6'),
            dbc.Col([
                dcc.Graph(
                    style=graph_style,
                    figure=go.Figure(
                        dist_plot
                    )
                )
            ], className='col-sm-6'),
        ])
    ]

def forecast_chart(forecast, metric, show_actuals=False):
    lower_bound = go.Scatter(
        name = 'Lower Bound',
        x = forecast['date'],
        y = forecast['yhat_lower'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines'
    )

    upper_bound = go.Scatter(
        name = 'Upper Bound',
        x = forecast['date'],
        y = forecast['yhat_upper'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )

    yhat = go.Scatter(
        name = 'Predicted Value',
        x = forecast['date'],
        y = forecast['yhat'],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )

    actual = go.Scatter(
        name = 'Actual Value',
        x = forecast['date'],
        y = forecast['y'],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )
    if show_actuals:
        data = [lower_bound, upper_bound, actual]
    else:
        data = [lower_bound, upper_bound, yhat]
    layout = go.Layout(
        yaxis=dict(title='Number of Chances'),
        title='Grand Canyon Number of Chances',
        showlegend = False,
        height=350
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

def create_trend_chart(forecast, metric_name, date_col):
    cols = [date_col, metric_name, metric_name + '_lower', metric_name + '_upper']
    forecast_sub = forecast[cols]
    for col in cols[1:]:
        forecast_sub[col] = forecast_sub[col].round(4)
    forecast_sub = forecast_sub.drop_duplicates()
    forecast_sub = forecast_sub.sort_values(date_col)

    lower_bound = go.Scatter(
        name = 'Lower Bound',
        x = forecast_sub[date_col],
        y = forecast_sub[metric_name + '_lower'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines'
    )

    upper_bound = go.Scatter(
        name = 'Upper Bound',
        x = forecast_sub[date_col],
        y = forecast_sub[metric_name + '_upper'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )

    metric = go.Scatter(
        name = 'Predicted Value',
        x = forecast_sub[date_col],
        y = forecast_sub[metric_name],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )
    data = [lower_bound, upper_bound, metric]

    layout = go.Layout(
        yaxis=dict(title=metric_name),
        title='sample title',
        showlegend = False,
        height=300)
    fig = go.Figure(data=data, layout=layout)
    return fig

def plot_forecast(df, metric, show_actuals=False):
    fig1 = forecast_chart(df, metric, show_actuals)
    trend = create_trend_chart(df, 'trend', date_col='date')
    weekly = create_trend_chart(df, 'weekly', date_col='day_of_week')
    yearly = create_trend_chart(df, 'yearly', date_col='day_of_year')
    return [
        dcc.Graph(
            id='forecast-graph',
            figure=fig1,
            style=graph_style
        ),
        dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=trend,
                        style=graph_style
                    )
                ], className='col-lg-4'),
                dbc.Col([
                    dcc.Graph(
                        figure=weekly,
                        style=graph_style
                    )
                ], className='col-lg-4'),
                dbc.Col([
                    dcc.Graph(
                        figure=yearly,
                        style=graph_style
                    )
                ], className='col-lg-4')
        ])
    ]

collapse_filters_actual = dbc.Collapse([
                dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P('Select Metric'),
                            dbc.RadioItems(
                                id='metric-selector-actual',
                                options=[
                                    {'label': 'Number of Applicants', 'value': 'n_applicants'},
                                    {'label': 'Number of Chances', 'value': 'n_chances'},
                                    {'label': 'Odds with 1 chance', 'value': 'odds'}
                                ],
                                value='n_chances'
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            html.P('Select Day of Week of Launch'),
                            dbc.Checklist(
                                options=[
                                    {'label': day_name, 'value': str(i)}
                                        for i, day_name in enumerate(dow_names)
                                ],
                                values=[str(i) for i in range(7)],
                                inline=True,
                                id='dow-selector-actual',
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            html.P('Select Metric'),
                            dbc.Checklist(
                                id='month-selector-actual',
                                options=[
                                    {'label': month_name, 'value': i+1}
                                        for i, month_name in
                                        enumerate(month_names)
                                ],
                                values=[i+1 for i in range(12)],
                                inline=True
                            ),
                        ]
                    )
                ]
                ),
            ], id='actual-collapse')
collapse_filters_forecast = dbc.Collapse([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P('Select Metric'),
                                dbc.RadioItems(
                                    id='metric-selector-forecast',
                                    options=[
                                        {'label': 'Number of Applicants', 'value': 'n_applicants'},
                                        {'label': 'Number of Chances', 'value': 'n_chances'},
                                        {'label': 'Odds with 1 chance', 'value': 'odds'}
                                    ],
                                    value='n_chances'
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.P('Select Day of Week of Launch'),
                                dbc.Checklist(
                                    options=[
                                        {'label': day_name, 'value': str(i)}
                                            for i, day_name in enumerate(dow_names)
                                    ],
                                    values=[str(i) for i in range(7)],
                                    inline=True,
                                    id='dow-selector-forecast',
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.P('Plot Actuals?'),
                                dbc.RadioItems(
                                    id='plot-actuals-forecast',
                                    options=[
                                        {'label': 'Yes', 'value': True},
                                        {'label': 'No', 'value': False}
                                    ],
                                    inline=True
                                ),
                            ]
                        )
                    ]),
                ], id='forecast-collapse')

colors = {
         'background': '#0000FF',
         'padding': 10
         # 'color': '#FFA500'
}

tab1_body = dbc.Card(
    [
        dbc.CardBody([
            dbc.Button("Open Filters", id="collapse-button-actual", className="mb-3"),
            collapse_filters_actual,
            dbc.Row([
                dbc.Col([
                    html.Div(id='actual-values', children=plot_actuals(load_actuals(), metric='n_chances'))
                ])
            ])
        ])
    ], style=colors
)
tab2_body = dbc.Card(
    [
        dbc.CardBody([
            dbc.Button("Open Filters", id="collapse-button-forecast", className="mb-3"),
            collapse_filters_forecast,
            dbc.Row([
                dbc.Col([
                    html.Div(id='forecast-values', children=plot_forecast(load_forecast(), metric='n_chances'))
                ])
            ])
        ])
    ], style=colors
)


app.layout = dbc.Tabs([
        dbc.Tab(tab1_body, label='Actual'),
        dbc.Tab(tab2_body, label='Forecast'),
    ],
    style=colors
    )

@app.callback(
    Output('actual-values', 'children'),
    [Input('metric-selector-actual', 'value'),
    Input('dow-selector-actual', 'values'),
    Input('month-selector-actual', 'values'),
    Input('actuals-graph', 'relayoutData')])
def plot_actuals_callback(metric_selection, dow_selection, month_selection, relayoutData):
    df = load_actuals()
    df_sub = df.loc[df['day_of_week'].isin(dow_selection)]

    df_sub.loc[~df_sub['month'].isin(month_selection), metric_selection] = None
    try:
        start_date = relayoutData['xaxis.range[0]']
        end_date = relayoutData['xaxis.range[1]']
        df_sub = df_sub.loc[(df_sub['date'] >= start_date) & (df_sub['date'] <= end_date)]
    except (KeyError, TypeError):
        pass
    return plot_actuals(df_sub, metric=metric_selection)

@app.callback(
    Output('forecast-values', 'children'),
    [Input('metric-selector-forecast', 'value'),
    Input('dow-selector-forecast', 'values'),
    Input('plot-actuals-forecast', 'value'),
    Input('forecast-graph', 'relayoutData')]
)
def plot_forecast_callback(metric_selection, dow_selection, plot_actuals, relayoutData):
    forecast = load_forecast()
    forecast_sub = forecast.loc[forecast['day_of_week'].isin(dow_selection)]
    print(forecast_sub.shape)

    forecast_sub = forecast_sub.loc[forecast_sub['metric'] == metric_selection]
    print(forecast_sub.shape)

    try:
        start_date = relayoutData['xaxis.range[0]']
        end_date = relayoutData['xaxis.range[1]']
        forecast_sub = forecast_sub.loc[(forecast_sub['date'] >= start_date) & (forecast_sub['date'] <= end_date)]
    except (KeyError, TypeError):
        pass
    if plot_actuals:
        forecast_sub = forecast_sub.loc[forecast_sub['date'] <= dt(2019,12,31)]
    return plot_forecast(forecast_sub, metric=metric_selection, show_actuals=plot_actuals)

@app.callback(
    Output("actual-collapse", "is_open"),
    [Input("collapse-button-actual", "n_clicks")],
    [State("actual-collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("forecast-collapse", "is_open"),
    [Input("collapse-button-forecast", "n_clicks")],
    [State("forecast-collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
