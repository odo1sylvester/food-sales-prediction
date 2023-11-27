import streamlit as st
from datetime import date
from datetime import datetime

import yfinance as yf
from prophet import Prophet 
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Foodstuff Alert System")
# st.subheader("Foodstuff Restock Alert and Price Prediction System")
item1 = "Rice"
item2 = "Beans"
item3 = "Yam"
item4 = "Garri"

stocks = (item1, item2, item3, item4)

selected_stock = st.selectbox("Select dataset for prediction", stocks)
food_item = selected_stock
if selected_stock is item1:   
     selected_stock = "AAPL"    
elif selected_stock is item2:
     selected_stock = "GOOG"
     st.write(item2)
elif selected_stock is item3:
     selected_stock = "MSFT"
elif selected_stock is item4:
     selected_stock = "GME"
else:
     st.write("No item seleted")
    
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data 
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if ticker is "AAPL": ## Rice
         data = data * (3500/191)  
    elif ticker is "GOOG":## Beans
         data = data * (3500/140)
    elif ticker is "MSFT":## Yam
         data = data * (1000/378) 
    else: ## Garri
         data = data * (1300/13) 
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
column_mapping = {"Date": "Date", "Open": "Opening Market Price", "Close": "Closing Market Price", "Low": "Lowest Market Price", "Volume": "Trading Quantity"}
selected_columns = list(column_mapping.keys())

# Display only the selected and renamed columns
data_display = data[selected_columns].rename(columns=column_mapping).tail()

# Remove 'Opening Market Price' column
data_display = data_display.drop(columns=['Trading Quantity'])

# Add 'Quantity' column calculated as 'Lowest Market Price' * 10
data_display['Quantity'] = (data_display['Lowest Market Price'] * 10).astype(int)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
# Convert all numeric columns to integers in data_display
numeric_columns_data = data_display.select_dtypes(include=['number']).columns
data_display[numeric_columns_data] = data_display[numeric_columns_data].astype(int)

st.write(data_display)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=(data['Open']).astype(int), name='market_openning_price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'].astype(int), name='market_closint_price'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Forcasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

##st.write("The future holds:::::::::::::::")

############################################################
# Create a textbox for user input
st.subheader(food_item+' Prediction Section ')

# Create a date picker
selected_date = st.date_input("Select a date", datetime.today())

# Display the selected date
st.write("Selected date:", selected_date)

if selected_date:
        ##########################################################

        # Create a dataframe with the date you want to 

        future_date = pd.to_datetime(selected_date)
        future_df = pd.DataFrame({'ds': [future_date]})

        # Use the model to make predictions for the future date
        forecast = m.predict(future_df)

        # Print the forecast for 2/3/2024

        
        print(forecast[['ds', 'yhat']])

        # Display the forecast with renamed columns
st.subheader('Forecast data')

# Rename the columns
forecast_display = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Price'})
forecast_display["Predicted Price"] = forecast_display["Predicted Price"].astype(int)
st.write(forecast_display.tail())
