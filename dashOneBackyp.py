import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# ------------------------------------------------------------------------
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Input(
                id='input-x',
                placeholder='Insert x value',
                type='number',
                value='',
            ),
            dcc.Input(
                id='input-y',
                placeholder='Insert y value',
                type='number',
                value='',
            ),
            html.Br(),
            html.Br(),

            # dcc.Input(
            #     id='my_txt_input',
            #     type='text',
            #     debounce=True,           # changes to input are sent to Dash server only on enter or losing focus
            #     pattern=r"^[A-Za-z].*",  # Regex: string must start with letters only
            #     spellCheck=True,
            #     inputMode='latin',       # provides a hint to browser on type of data that might be entered by the user.
            #     name='text',             # the name of the control, which is submitted with the form data
            #     list='browser',          # identifies a list of pre-defined options to suggest to the user
            #     n_submit=0,              # number of times the Enter key was pressed while the input had focus
            #     n_submit_timestamp=-1,   # last time that Enter was pressed
            #     autoFocus=True,          # the element should be automatically focused after the page loaded
            #     n_blur=0,                # number of times the input lost focus
            #     n_blur_timestamp=-1,     # last time the input lost focus.
            #     # selectionDirection='', # the direction in which selection occurred
            #     # selectionStart='',     # the offset into the element's text content of the first selected character
            #     # selectionEnd='',       # the offset into the element's text content of the last selected character
            # ),
        ]),

        html.Datalist(id='browser', children=[
            html.Option(value="1"),
            html.Option(value="2"),
            html.Option(value="3")
        ]),

        html.Br(),
        html.Br(),

        html.Label('Dropdown'),
        dcc.Dropdown(
            options=[
                {'label': '10', 'value': '10'},
                {'label': '12', 'value': '12'},
                {'label': '14', 'value': '14'}
            ],
            value='12'
        ),

        html.Br(),
        html.Br(),

        html.Label('Multi-Select Dropdown'),
        dcc.Dropdown(
            options=[
                {'label': '20', 'value': '20'},
                {'label': '22', 'value': '22'},
                {'label': '24', 'value': '24'}
            ],
            value=['20', '24'],
            multi=True
        ),

        html.Br(),
        html.Br(),

        html.Label('Radio Items'),
        dcc.RadioItems(
            options=[
                {'label': '6', 'value': '6'},
                {'label': '8', 'value': '8'},
                {'label': '10', 'value': '10'}
            ],
            value='6'
        ),

        html.Br(),
        html.Br(),

        html.Label('Checkboxes'),
        dcc.Checklist(
            options=[
                {'label': '5', 'value': '5'},
                {'label': '10', 'value': '10'},
                {'label': '15', 'value': '15'}
            ],
            value=['10', '15']
        ),

        html.Br(),
        html.Br(),

        html.Label('Text Input'),
        dcc.Input(value='MTL', type='text'),

        html.Label('Slider'),
        dcc.Slider(
            min=0,
            max=9,
            marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            value=5,
        ),

    ], style={'columnCount': 2}),

    html.Div([html.Button(id='reset-button', n_clicks=0, children='Reset', style={'fontWeight': 'bold',
                        'textAlign':'center'}, title='Click to clear the inputs'),
                        html.Button(id='submit-button', n_clicks=0, children='Submit', style={'fontWeight': 'bold',
                        'textAlign':'center'}, title='Click to optimize')
    ], style={'textAlign': 'center'}),

    html.Div([
        html.Br(),
        html.Br(),

        html.P(['------------------------']),
        html.Div(id='result')
        # html.P(['------------------------']),
        # html.P(['Real-time string output example:']),
        # html.Div(id='div_output'),
        #
        # html.P(['------------------------']),
        # html.P(['Enter clicked:']),
        # html.Div(id='div_enter_clicked'),
        #
        # html.P(['Enter clicked timestamp:']),
        # html.Div(id='div_sub_tmstp'),
        #
        # html.P(['------------------------']),
        #
        # html.P(['Input lost focus:']),
        # html.Div(id='div_lost_foc'),
        #
        # html.P(['Lost focus timestamp:']),
        # html.Div(id='div_lst_foc_tmstp'),
    ], style={'textAlign': 'center'}),
])


# ------------------------------------------------------------------------
@app.callback(
    Output('result', 'children'),
    [Input('input-x', 'value'),
     Input('input-y', 'value')]
)
def update_result(x, y):
    result = x + y
    resultString = "The is: " + str(result)
    return resultString
# @app.callback(
#     [Output(component_id='div_output', component_property='children'),
#      Output(component_id='div_enter_clicked', component_property='children'),
#      Output(component_id='div_sub_tmstp', component_property='children'),
#      Output(component_id='div_lost_foc', component_property='children'),
#      Output(component_id='div_lst_foc_tmstp', component_property='children')],
#     [Input(component_id='my_txt_input', component_property='value'),
#      Input(component_id='my_txt_input', component_property='n_submit'),
#      Input(component_id='my_txt_input', component_property='n_submit_timestamp'),
#      Input(component_id='my_txt_input', component_property='n_blur'),
#      Input(component_id='my_txt_input', component_property='n_blur_timestamp')]
# )
# def update_graph(txt_inserted, num_submit, sub_tmstp, lost_foc, lst_foc_tmstp):
#     if sub_tmstp == -1:
#         submited_dt = sub_tmstp
#     else:
#         submited_dt = datetime.datetime.fromtimestamp(int(sub_tmstp) / 1000)  # using the local timezone
#         submited_dt = submited_dt.strftime("%Y-%m-%d %H:%M:%S")
#
#     if lst_foc_tmstp == -1:
#         lost_foc_dt = lst_foc_tmstp
#     else:
#         lost_foc_dt = datetime.datetime.fromtimestamp(int(lst_foc_tmstp) / 1000)  # using the local timezone
#         lost_foc_dt = lost_foc_dt.strftime("%Y-%m-%d %H:%M:%S")
#
#     return txt_inserted, num_submit, submited_dt, lost_foc, lost_foc_dt



# ------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)