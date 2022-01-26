import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import io
from base64 import b64encode



buffer = io.StringIO()

df = px.data.iris()
fig = px.scatter(
    df, x="sepal_width", y="sepal_length", 
    color="species")
fig.write_html(buffer)

html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig),
    # dcc.Graph(id="world", figure=world),
    # dcc.Graph(id="usa", figure=usa),


    # dcc.Graph(id="topic_time", figure=fig_topic_time),

    # html.A(
    #     html.Button("Download HTML"), 
    #     id="download",
    #     href="data:text/html;base64," + encoded,
    #     # download="plotly_graph.html"
    # )
])

app.run_server(debug=False, port = 8070)