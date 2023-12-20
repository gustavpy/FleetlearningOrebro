import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


df = pd.read_csv("FLEET LEARNING/metadata.csv")


app = dash.Dash(__name__)

#Skapar density map
densitymap = px.density_mapbox(df, lat=df["latitude"], lon=df["longitude"], radius=6,
                        center=dict(lat=58, lon=10), zoom=2.5,
                        mapbox_style="carto-positron", 
                        hover_data={"latitude": True, "longitude": True, "frame_id": True},)

#Ändrar höjd på kartan och titel.
densitymap.update_layout(
    height = 800,
    title_text='Density Map',
    title_x=0.5,
    title_font_size=24)

#Skapar en dictionary "figs" och fyller den med pie chart-figurer baserade på kolumnerna i DataFrame "metadata.csv"
figs = {}
for column in df.columns:
    fig = px.pie(df, names=column, title=f'{column}',hole=0.3)
    figs[column] = fig


#Layout för pie charts och densitymap
app.layout = html.Div([
    # First row
    html.Div([
        dcc.Graph(id='country-code-pie-chart', figure=figs['country_code']),
        dcc.Graph(id='road-condition-pie-chart', figure=figs['road_condition']),
    ], style={'display': 'flex'}),

    # Second row
    html.Div([
        dcc.Graph(id='scraped-weather-pie-chart', figure=figs['scraped_weather']),
        dcc.Graph(id='time-of-day-pie-chart', figure=figs['time_of_day']),
    ], style={'display': 'flex'}),

    # Third row
    html.Div([
        dcc.Graph(id='road-type-pie-chart', figure=figs['road_type']),
        dcc.Graph(id='collection-car-pie-chart', figure=figs['collection_car']),
    ], style={'display': 'flex'}),

    dcc.Graph(id='density-map',figure=densitymap),])


#Callback för varje pie chart
@app.callback(
    [Output('country-code-pie-chart', 'figure'),
     Output('road-condition-pie-chart', 'figure'),
     Output('scraped-weather-pie-chart', 'figure'),
     Output('time-of-day-pie-chart', 'figure'),
     Output('road-type-pie-chart', 'figure'),
     Output('collection-car-pie-chart', 'figure'),
     Output('density-map','figure')],
    [Input('country-code-pie-chart', 'clickData'),
     Input('road-condition-pie-chart', 'clickData'),
     Input('scraped-weather-pie-chart', 'clickData'),
     Input('time-of-day-pie-chart', 'clickData'),
     Input('road-type-pie-chart', 'clickData'),
     Input('collection-car-pie-chart', 'clickData')],)


#Funktion för att uppdatera pie chartsen när man klickar på någon av de.
def update_pie_charts(clickData_country_code, clickData_road_condition, clickData_scraped_weather, clickData_time_of_day, clickData_road_type,clickData_collection_car):
    updated_figs = {}

    if clickData_country_code is not None: #Om någon piechart blivit klickad på
        selected = clickData_country_code['points'][0]['label'] #Vilken del av pie charten som klickats på (ex. SE) 
        filtered_df = df[df['country_code'] == selected] #Nya dataframen baserat på vilket land man klickat på
        for column in df.columns:
            updated_fig = px.pie(filtered_df, names=column, title=f'{column} distribution for {selected}') #Uppdaterar pie chartsen
            updated_figs[column] = updated_fig

        densitymap.update_traces(lat=filtered_df["latitude"],lon=filtered_df["longitude"])
        
    
    elif clickData_road_condition is not None:
        selected = clickData_road_condition['points'][0]['label'] 
        filtered_df = df[df['road_condition'] == selected] 
        for column in df.columns:
            updated_fig = px.pie(filtered_df, names=column, title=f'{column} distribution for {selected}')
            updated_figs[column] = updated_fig

        densitymap.update_traces(lat=filtered_df["latitude"],lon=filtered_df["longitude"])
        
        
    elif clickData_scraped_weather is not None:
        selected = clickData_scraped_weather['points'][0]['label'] 
        filtered_df = df[df['scraped_weather'] == selected]
        for column in df.columns: 
            updated_fig = px.pie(filtered_df, names=column, title=f'{column} distribution for {selected}')
            updated_figs[column] = updated_fig

        densitymap.update_traces(lat=filtered_df["latitude"],lon=filtered_df["longitude"])
        

    elif clickData_time_of_day is not None:
        selected = clickData_time_of_day['points'][0]['label']
        filtered_df = df[df['time_of_day'] == selected] 
        for column in df.columns: 
            updated_fig = px.pie(filtered_df, names=column, title=f'{column} distribution for {selected}')
            updated_figs[column] = updated_fig
        
        densitymap.update_traces(lat=filtered_df["latitude"],lon=filtered_df["longitude"])
        
    
    elif clickData_road_type is not None:
        selected = clickData_road_type['points'][0]['label']
        filtered_df = df[df['road_type'] == selected] 
        for column in df.columns:
            updated_fig = px.pie(filtered_df, names=column, title=f'{column} distribution for {selected}')
            updated_figs[column] = updated_fig
        
        densitymap.update_traces(lat=filtered_df["latitude"],lon=filtered_df["longitude"])


    elif clickData_collection_car is not None:
        selected = clickData_collection_car['points'][0]['label']
        filtered_df = df[df['collection_car'] == selected] 
        for column in df.columns:
            updated_fig = px.pie(filtered_df, names=column, title=f'{column} distribution for {selected}')
            updated_figs[column] = updated_fig
        
        densitymap.update_traces(lat=filtered_df["latitude"], lon=filtered_df["longitude"])
        

    else: #Om ingen klickats
        for column in df.columns:
            updated_figs[column] = figs[column]

        densitymap.update_traces(lat=df["latitude"], lon=df["longitude"])
    
    
    updated_figs['density_map'] = densitymap
    return updated_figs['country_code'], updated_figs['road_condition'], updated_figs['scraped_weather'], updated_figs['time_of_day'], updated_figs['road_type'], updated_figs['collection_car'], updated_figs['density_map']

if __name__ == '__main__':
    app.run_server(debug=True)  #Lägg till så man kan klicka flera 