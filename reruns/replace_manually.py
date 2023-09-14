import dash
import pandas as pd
import numpy as np
import sys, os
pynebDir = os.path.expanduser("~/pyneb/src")
if pynebDir not in sys.path:
    sys.path.insert(0,pynebDir)
import pyneb
from dash.dependencies import Input, Output, State, ALL
dcc = dash.dcc
html = dash.html
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
# import json
import glob
import scipy.interpolate

import datetime

externalStyles = [dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=externalStyles,
                suppress_callback_exceptions=True)
app.title = "PES Replacement"


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

coordStrings = ["q20","q30"]
expectedCoords = ["expected_"+c for c in coordStrings]
otherPlotKeys = ["q30","q5","q6","q7","q8"]

inertiaKeys = ["M_22","M_32","M_33"]

colsToInterp = ["EHFB",] + inertiaKeys

refreshFile = sorted(glob.glob("refresh.dat"))[0]
df = pd.read_csv(refreshFile,sep="\t")

coords = df[expectedCoords].to_numpy(copy=False)
df.set_index(expectedCoords,inplace=True)
df = df.sort_index(level=expectedCoords)

#Fixing NaN values in the dataframe, at least to make a nice-looking plot initially
dfCopy = df.copy()
indsToInterpAt = df.loc[df["EHFB"].isna()].index
indsToInterpAtArr = np.array(indsToInterpAt.to_list())
indsToFitTo = df[df["isCorrect"]].index
indsToFitToArr = np.array(indsToFitTo.to_list())

if len(indsToInterpAtArr) > 0:
    pesInterp = scipy.interpolate.RBFInterpolator(indsToFitToArr,
                                                  df.loc[indsToFitTo,colsToInterp].to_numpy())
    df.loc[indsToInterpAt,colsToInterp] = pesInterp(indsToInterpAtArr)


df["isUpdatedManually"] = [False]*len(df)

uniqueCoords = [np.unique(c) for c in coords.T]
shp = [len(u) for u in uniqueCoords]
coordMeshTuple = np.meshgrid(*uniqueCoords)

zz = df["EHFB"].to_numpy(copy=True).reshape(shp)
gsInds = pyneb.SurfaceUtils.find_local_minimum(zz)
gsEneg = zz[gsInds]
df["EHFB"] -= gsEneg

#For resetting later
originalDf = df.copy()

def make_plot_layout(name,hidden=False):
    divWithButtons = html.Div([
        html.Div([
            dcc.Graph(
                id=name+"-plot",
                ),
            ]),
    
        html.Div([
            html.Div([
                html.Pre(id=name+'-click-data', style=styles['pre']),
                ])
            ]),
    
        html.Div([
            dbc.Row([
                dbc.Col(
                    dcc.Input(
                        id=name+"-plot-min",
                        type="number",
                        placeholder="Minimum plot value:",
                        ),
                    ),
                
                dbc.Col(
                    dcc.Input(
                        id=name+"-plot-max",
                        type="number",
                        placeholder="Maximum plot value:",
                        ),
                    ),
                ])
            ]),
        ],
        hidden=hidden,
        id=name+"-base-div"
        )
    return divWithButtons

#%% Layout
plotLayouts = []
for key in otherPlotKeys:
    plotLayouts.append(html.Button(children="Show "+key,id=key+"-show",n_clicks=0))
    plotLayouts.append(make_plot_layout(key,hidden=True))

app.layout = html.Div([
    html.H1("PES Replacement"),
    
    dbc.Row([
        dbc.Col([
            html.Button("Update plots",
                        id="update-plots",n_clicks=0),
            ]),
        
        dbc.Col([
            html.Button("Dump to file",id="dump-file",n_clicks=0)
            ]),
        
        dbc.Col([
            html.Button("Dump to PES",id="dump-pes",n_clicks=0)
            ]),
        
        dbc.Col([
            html.Button("Reset",id="reset-plots",n_clicks=0),
            ]),
        ]),
    html.Hr(),
    
    make_plot_layout("EHFB"),
    ]
    + plotLayouts
    +[
    html.Hr(),
    ])

#%% Functions
def make_fig(df,uniqueCoords,arr,plotMin,plotMax,colors,styles,dsetName):
    fig = go.Figure(
        data=go.Contour(z=arr.T,x=uniqueCoords[0],y=uniqueCoords[1],
                        colorscale="Spectral_r",
                        zmin=plotMin,zmax=plotMax,ncontours=100)
        )
    
    fig.add_trace(go.Scatter(x=df.index.get_level_values(0),y=df.index.get_level_values(1),
                             mode="markers",marker={"color":colors,
                                                    "symbol":styles,
                                                    "line":{
                                                        "width":1,
                                                        "color":"black"}}))
    fig.update_traces()
    fig.update_layout(
                      xaxis_title=coordStrings[0],
                      yaxis_title=coordStrings[1],
                      title=dsetName
                      )
    return fig

def set_colors_and_styles(df):
    colors = []
    for i in df["isCorrect"]:
        if i:
            colors.append("black")
        else:
            colors.append("white")
    styles = []
    for i in df["isUpdatedManually"]:
        if i:
            styles.append("x")
        else:
            styles.append("circle")
            
    return colors, styles

def make_plot_with_converged_marks(dsetName):
    @app.callback(
        [
         Output(dsetName+'-plot','figure'),
         Output(dsetName+"-plot","clickData"),
         Output(dsetName+"-plot-min","value"),
         Output(dsetName+"-plot-max","value"),
        ],
        [
         Input(dsetName+'-plot','clickData'),
         Input("update-plots","n_clicks"), #Necessary to tell Dash to listen for this trigger
         Input("reset-plots","n_clicks"), #Necessary to tell Dash to listen for this trigger
         Input(dsetName+"-plot-min","value"),
         Input(dsetName+"-plot-max","value"),
        ]
        )
    def callback(clickData,updatePlots,resetPlots,minVal,maxVal):
        global df
        
        #Multiple different callbacks should update the plots
        if dash.ctx.triggered_id == "reset-plots":
            df = originalDf.copy()
            minVal = None
            maxVal = None
        
        if minVal is None:
            plotMin = df[dsetName].min()
        else:
            plotMin = minVal
        if maxVal is None:
            plotMax = df[dsetName].max()
        else:
            plotMax = maxVal
        
        if clickData is None:
            colors, styles = set_colors_and_styles(df)
            
            zz = df[dsetName].to_numpy(copy=True).reshape(shp)
            
            fig = make_fig(df,uniqueCoords,zz,plotMin,plotMax,colors,styles,dsetName)
        else:
            #Because otherwise clicking on it converts to int -_-
            x = float(clickData["points"][0]["x"])
            y = float(clickData["points"][0]["y"])
            
            df.loc[(x,y),"isUpdatedManually"] = ~df.loc[(x,y),"isUpdatedManually"]
            df.loc[(x,y),"isCorrect"] = False
            
            colors, styles = set_colors_and_styles(df)
            
            zz = df[dsetName].to_numpy(copy=True).reshape(shp)
            
            fig = make_fig(df,uniqueCoords,zz,plotMin,plotMax,colors,styles,dsetName)
            
        #Multiple different callbacks should update the plots
        if dash.ctx.triggered_id == "update-plots":
            colors, styles = set_colors_and_styles(df)
            
            dfCopy = df.copy()
            indsToInterpAt = dfCopy[dfCopy["isUpdatedManually"]].index
            indsToInterpAtArr = np.array(indsToInterpAt.to_list())
            indsToFitTo = dfCopy[~dfCopy["isUpdatedManually"]].index
            indsToFitToArr = np.array(indsToFitTo.to_list())
            
            pesInterp = scipy.interpolate.RBFInterpolator(indsToFitToArr,
                                                          dfCopy.loc[indsToFitTo,colsToInterp].to_numpy())
            df.loc[indsToInterpAt,colsToInterp] = pesInterp(indsToInterpAtArr)
            
            zz = df[dsetName].to_numpy(copy=True).reshape(shp)
            
            fig = make_fig(df,uniqueCoords,zz,plotMin,plotMax,colors,styles,dsetName)
            
        return fig, None, minVal, maxVal
    return callback

def make_plot_div_visible(name):
    @app.callback(
        [
         Output(name+"-show","n_clicks"),
         Output(name+"-show","children"),
         Output(name+"-base-div","hidden")
         ],
        [
         Input(name+"-show","n_clicks"),
         Input(name+"-base-div","hidden")
         ],
        prevent_initial_update=True
        )
    def callback(nClicks,isHidden):
        if nClicks > 0:
            if isHidden: #Update for what the *next* button press should do
                text = "Hide "+name
            else:
                text = "Show "+name
            isHidden = not isHidden
        else:
            isHidden = True
            text = "Show "+name
        return 0, text, isHidden
    
    return callback

@app.callback(
    [
     Output("dump-file","n_clicks"),
     ],
    [
     Input("dump-file","n_clicks"),
     ]
    )
def write_file(nClicks):
    if nClicks > 0:
        df.to_csv(datetime.datetime.now().isoformat()+"_refresh.dat",sep="\t")
    return 0,

@app.callback(
    [
     Output("dump-pes","n_clicks"),
     ],
    [
     Input("dump-pes","n_clicks"),
     ]
    )
def dump_pes(nClicks):
    if nClicks > 0:
        dfOut = df[inertiaKeys+["EHFB",]].copy()
        dfOut = dfOut.reset_index()
        dfOut = dfOut.rename(columns={e:c for (e,c) in zip(expectedCoords,coordStrings)})
        dfOut.to_csv("PES.dat",sep="\t",index=False)
    return 0,

make_plot_with_converged_marks("EHFB")

for key in otherPlotKeys:
    make_plot_with_converged_marks(key)
    make_plot_div_visible(key)

if __name__ == '__main__':
    app.run_server(debug=True)
