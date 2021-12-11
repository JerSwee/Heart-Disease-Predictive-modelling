# the dashOne.py aims to generate the GUI of the product HeartLink, which serves as a main interface to interact with user, to provide predicted outcomes for clients with heart disease concerns

# libraries and packages needed for GUI
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import classifier as clsf
from flask import Flask
import logging
import webbrowser
from threading import Timer

port = 8050  # specify the port number while displaying the web page

# running the server using Flask package
server = Flask(__name__)

# record the logging message for the running program
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# external css stylesheet linked for use
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# ------------------------------------------------------------------------
# main layout that displays on the webpage for the app
app.layout = html.Div([

    # first section on the HTML webpage which is the title for the product
    html.Div([
        html.H2("Heart Disease Prediction Team 5", style={"margin-top": 5}),
        html.Hr(),
    ], style={'textAlign': 'center'}),

    # logo image to display on the top right corner
    html.Div([
        html.Img(src=app.get_asset_url('heartLinkLogo.png'))],
        style={'position': 'absolute', "top": "8px", "right": "48px"}),

    # second section on the HTML webpage which aims to provide input fields for the criterion to be filled by user
    html.Div([
        html.Div([
            # label to name the input attribute and style to add in aesthetic features
            html.Label('Typical Chest Pain', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Dropdown(
                id='Typical Chest Pain',  # id to identify the attribute
                options=[
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'},
                ],
            ),
            html.Br(),
            # Specification for Typical Chest Pain Input
            html.Label('Atypical Chest Pain', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Dropdown(
                id='Atypical',
                options=[
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'},
                ],
            ),
            html.Br(),
            # Specification for Age Input
            html.Label('Age', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Input(
                id='Age',
                type='number',
                value='',
            ),
            html.Br(),
            html.Br(),
            # Specification for Weight Input
            html.Label('Weight (kg)', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Input(
                id='Weight',
                type='number',
                value='',
            ),
            html.Br(),
            html.Br(),
            # Specification for Hypertension Input
            html.Label('Hypertension (HTN)', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Dropdown(
                id='HTN',
                options=[
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'},
                ],
            ),
            html.Br(),
            # Specification for Regional Wall Motion Abnormalities Input
            html.Label('Regional Wall Motion Abnormalities (RWMA)', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Dropdown(
                id='RWMA',
                options=[
                    {'label': 'None', 'value': '0'},
                    {'label': 'Low', 'value': '1'},
                    {'label': 'Med', 'value': '2'},
                    {'label': 'High', 'value': '3'},
                    {'label': 'Very High', 'value': '4'},

                ],
            ),
            html.Br(),
            # Specification for Nonanginal Chest Pain Input
            html.Label('Nonanginal Chest Pain', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Dropdown(
                id='Nonanginal',
                options=[
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'},
                ],
            ),

            html.Br(),
            # Specification for Ejection Fraction Input
            html.Label('Ejection Fraction (%)', style={"color": "#ff99bb", "font-weight": "bold"}),
            html.Br(),
            dcc.Slider(
                id='EF-TTE',
                min=0,
                max=100,
                marks={i: str(i) for i in range(0, 105, 5)},
            ),

            html.Br(),
            html.Br(),
            html.Br(),
            # Specification for Diabetes Mellitus Input
            html.Label('Diabetes Mellitus (DM)', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Dropdown(
                id='DM',
                options=[
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'},
                ],
            ),

            html.Br(),
            # Specification for T-Wave Inversion Input
            html.Label('T-Wave Inversion', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Dropdown(
                id='Tinversion',
                options=[
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'},
                ],
            ),
            html.Br(),
            # Specification for Body Mass Index Input
            html.Label('Body Mass Index (kb/m^2)', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Input(
                id='BMI',
                type='number',
                value='',
            ),
            html.Br(),
            html.Br(),
            # Specification for Triglyceride Input
            html.Label('Triglyceride (mg/dl)', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Input(
                id='TG',
                type='number',
                value='',
            ),
            html.Br(),
            html.Br(),
            # Specification for Erythrocyte Sedimentation rate Input
            html.Label('Erythrocyte Sedimentation rate (mm/h)', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Input(
                id='ESR',
                type='number',
                value='',
            ),
            html.Br(),
            html.Br(),
            # Specification for Blood Pressure Input
            html.Label('Blood Pressure', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Input(
                id='BP',
                type='number',
                value='',
            ),
            html.Br(),
            html.Br(),
            # Specification for Classifier Input
            html.Label('Classifier', style={"color": "#ff99bb", "font-weight": "bold"}),
            dcc.Dropdown(
                id='Classifier',
                options=[
                    {'label': 'Voting Classifier', 'value': 'voting'},
                    {'label': 'Stochastic Gradient Boosting', 'value': 'sgb'},
                    {'label': 'ADA Classifier', 'value': 'ada'},
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'Bagging with DT', 'value': 'bag'}
                ],
            ),
            html.Br(),
            # Specification for Neutrophil Input
            html.Label('Neutrophil (%)', style={"color": "#ff99bb", "font-weight": "bold"}),
            html.Br(),
            dcc.Slider(
                id='Neut',
                min=0,
                max=100,
                marks={i: str(i) for i in range(0, 105, 5)},
            ),
        ]),
    ], style={'columnCount': 2}),

    # third section on the HTML webpage which aims to display the predicted outcome for user
    html.Div([
        html.Hr(),
        html.H4("Predicted Outcome"),
        html.P(['---------------------------------------------']),
        html.Div(id='result', style={"color": "#ff99bb", "font-weight": "bold"}),
        html.Div(id='accuracy', children='Over Here', style={"color": "#ff99bb", "font-weight": "bold"}),
        html.P(['---------------------------------------------']),
        html.Br(),
    ], style={'textAlign': 'center'}),
])


# ------------------------------------------------------------------------
# callback aims to update the HTML elements in real-time, output components are updated with whatever were returned by the update_result function.
# new value of the input property as an input argument, and whenever an input property changes, the callback is being called automatically.
@app.callback(
    [Output('result', 'children'),
     Output('accuracy', 'children')],
    [Input('Typical Chest Pain', 'value'),
     Input('Atypical', 'value'),
     Input('HTN', 'value'),
     Input('Age', 'value'),
     Input('RWMA', 'value'),
     Input('Nonanginal', 'value'),
     Input('Tinversion', 'value'),
     Input('DM', 'value'),
     Input('EF-TTE', 'value'),
     Input('Weight', 'value'),
     Input('BMI', 'value'),
     Input('TG', 'value'),
     Input('ESR', 'value'),
     Input('BP', 'value'),
     Input('Neut', 'value'),
     Input('Classifier', 'value')
     ],
)
def update_result(TypicalChestPain, Atypical, HTN, Age, RWMA, Nonanginal, Tinversion, DM, EF, Weight, BMI, TG, ESR, BP,
                  Neut, Classifier):
    """
    Real-time update function that used with app callback to update the output components using input properties.
    :param TypicalChestPain: Input field that records Typical Chest Pain for user
    :param Atypical: Input field that records Atypical Chest Pain for user
    :param HTN: Input field that records HTN for user
    :param Age: Input field that records Age for user
    :param RWMA: Input field that records Regional Wall Motion Abnormalities for user
    :param Nonanginal: Input field that records Nonanginal Chest Pain for user
    :param Tinversion: Input field that records T-Wave Inversion for user
    :param DM: Input field that records Diabetes Mellitus for user
    :param EF: Input field that records Ejection Fraction for user
    :param Weight: Input field that records Weight for user
    :param BMI: Input field that records Body Mass Index for user
    :param TG: Input field that records Triglyceride  for user
    :param ESR: Input field that records Erythrocyte Sedimentation rate for user
    :param BP: Input field that records Blood Pressure for user
    :param Neut: Input field that records Neutrophil for user
    :param Classifier: Input field that records Classifier for user
    :return: The predicted heart disease outcome of either Normal or CAD label for user with the accuracy of the prediction
    """
    # try except block to perform exception handling for all non-integer values, if not return None, None then no output will be displayed
    try:
        intTypicalChestPain = int(TypicalChestPain)
        intAtypical = int(Atypical)
        intHTN = int(HTN)
        intAge = int(Age)
        intRWMA = int(RWMA)
        intNonanginal = int(Nonanginal)
        intTinversion = int(Tinversion)
        intDM = int(DM)
        intEF = int(EF)
        intWeight = int(Weight)
        intBMI = int(BMI)
        intTG = int(TG)
        intESR = int(ESR)
        intBP = int(BP)
        intNeut = int(Neut)
    except:
        return [None, None]

    # try using the input values to call the predictResult funciton using classifier.py file, if not return None, None then no output will be displayed
    try:
        result = clsf.predictResult(intTypicalChestPain, intAtypical, intHTN, intAge, intRWMA,
                                    intNonanginal, intTinversion, intDM, intEF, intWeight, intBMI, intTG,
                                    intESR, intBP, intNeut, Classifier)
        accuracy = clsf.getAccuracy(Classifier)
    except:
        return [None, None]

    # different labels returned based on the predicted results
    if (result == 0):
        return "Patient Prediction: Normal", "Classifier Accuracy: " + str(accuracy) + "%"
    else:
        return "Patient Prediction: CAD", "Classifier Accuracy: " + str(accuracy) + "%"


def open_browser():
    """
    open browser to auto prompt the webpage display whenever program is ran
    :return: None
    """
    webbrowser.open_new("http://localhost:{}".format(port))


# ------------------------------------------------------------------------
if __name__ == '__main__':
    Timer(1, open_browser).start()  # auto pop the web browser to display the GUI
    app.run_server(debug=False, port=port)
