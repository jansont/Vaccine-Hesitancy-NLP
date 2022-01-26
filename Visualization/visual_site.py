import io
from base64 import b64encode

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import bertopic
import pandas as pd
import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as pg
import plotly.validators.choropleth
import requests
from shapely.geometry import mapping, shape
from shapely.prepared import prep
from shapely.geometry import Point
from ast import literal_eval
import reverse_geocoder
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class TopicModel:
  def __init__(self, n_words = 10, topic_size = 10, n_topics = 10):
    self.model = BERTopic(top_n_words = 8,
                      min_topic_size = topic_size,
                      nr_topics = n_topics,
                      low_memory = True
                      )
  def fit_model(self, data):
    tweets = df['ProcessedTweet']
    docs = tweets.to_list()
    del tweets
    topics, probabilities = self.model.fit_transform(docs)
    return topics, probabilities

  def transform_topics(self, data):
    tweets = df['ProcessedTweet']
    tweets.map(lambda t: str(t))
    docs = tweets.to_list()
    topics_pred, probabilities = self.model.transform(docs)

  
  def save_topic_model(self, path):
    self.model.save(path)

  def vis_intertopic_dist(self):
    fig = model.visualize_topics()
    return fig

  def vis_term_decline(self, topics_class):
    fig = self.model.visualize_term_rank(topics = topics)

  def vis_barchart(topics = None, top_n_topics = None):
    fig = self.model.visualize_barchart(topics = None,top_n_topics = None,width = 800,height= 800)

  def load_topic_model(self, model):
    self.model = bertopic.load(model)

  def vis_topics_over_time(self, df, topics):
    tweets = df['ProcessedTweet']
    tweets.map(lambda t: str(t))
    docs = tweets.to_list()
    topics_time = self.model.topics_over_time(docs, topics = topics, timestamps = df['Month'].tolist(), nr_bins = 12, datetime_format = '%b')
    fig = self.model.visualize_topics_over_time(topics_time, topics = topics, top_n_topics = 5, width= 1250, height = 500)
    return fig

  def vis_topics_over_class(self, df, topics):
    topics, probabilities = model.fit_model(df)
    tweets = df['Sentiment']
    tweets.map(lambda t: str(t))
    docs = tweets.to_list()
    topics_class = self.model.topics_over_class(docs, topics = topics, timestamps = df['Sentiment'].tolist())
    fig = self.model.visualize_topics_over_time(topics_time, topics = topics, top_n_topics = 5, width= 1250, height = 500)
    return fig

  def get_topic(self, topic):
    model.get_topic(topic)

  def get_topic_sample(self, topic):
    model.get_representative_docs(topic)

#--------------------------------------------------------------------
# topic_model = TopicModel()
# topic_model.load_model(path = '/Users/theojanson/Project/Capstone/Vaccine-Hesitancy-NLP/TopicModels/models/sample_model')

# df = pd.read_csv(path = '/Users/theojanson/Project/Capstone/Data/models/sample_data.csv')
# topics = pd.read_csv(path = '/Users/theojanson/Project/Capstone/Data/sample_topics.csv')

# topic_model = TopicModel(model = model)
# fig_topic_time = topic_model.vis_topics_over_time(df, topics = topics)
#---------------------------------------------------------------------
# df = pd.read_csv('/Users/theojanson/Project/Capstone/Data/DailyStats.csv')

# std_mean_error, std_prob_error = [], []
# for i in range(len(df)):
#   std_mean_error.append(df['StdSentiment'].iloc[i]/ ((df['SampleSize'].iloc[i])**0.5))
#   std_prob_error.append(df['StdPositiveProb'].iloc[i]/ ((df['SampleSize'].iloc[i])**0.5))


# time_series = make_subplots(rows=3, cols=1, shared_xaxes=True,
#                     subplot_titles=('Mean Sentiment',  'Mean Probability of Positive Tweet', 'Sample Size'),
#                     vertical_spacing = 0)
# time_series.add_trace(
#     go.Scatter(x=df['Day'], y=df['MeanSentiment'],name='Mean Sentiment', 
#     error_y=dict(type='data', array=std_mean_error,visible=True)),
#     row=1, col=1
# )
# time_series.add_trace(
#     go.Scatter(x=df['Day'], y=df['MeanPositiveProb'],name='Prob. of Positive Tweet', 
#     error_y=dict(type='data', array=std_prob_error,visible=True)),
#     row=2, col=1
# )
# time_series.add_trace(
#     go.Scatter(x=df['Day'], y=df['SampleSize'],name='Sample Size', ),
#     row=3, col=1
# )
# time_series.update_layout(height=900, width=1000, title_text="Time Series")
# time_series.update_layout(legend_orientation="h", 
#              xaxis3_rangeslider_visible=True, xaxis3_rangeslider_thickness=0.1 )

# #-------------------------------------------------------------------------------------
# geo_dataset = pd.read_csv('/Users/theojanson/Project/Capstone/Data/Predicted_Geo_Tweets.csv')
# world = px.choropleth(geo_dataset, 
#                     locations=geo_dataset['Country'],
#                     color="Sentiment",
#                     animation_frame="Month",
#                     locationmode = 'country names',
#                     title = 'Vaccine Hesitancy',
#                     color_continuous_scale=px.colors.sequential.PuRd,)
# world["layout"].pop("updatemenus")
# usa = px.choropleth(geo_dataset,
#                     locations='US State',
#                     locationmode="USA-states",
#                     color='Sentiment',
#                     scope="usa", 
#                     animation_frame = 'Month',
#                     )
 #-------------------------------------------------------------------------------------



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

    html.A(
        html.Button("Download HTML"), 
        id="download",
        href="data:text/html;base64," + encoded,
        download="plotly_graph.html"
    )
])

app.run_server(debug=True)