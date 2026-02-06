from dash import Dash,html,dcc
from dash.dependecies import Input,Output

app=Dash(__name__)




app.run(debug=True)

app.layout=html.Div([
    html.Button("submit",id='number'),
    dcc.Input(placeholder='')
])