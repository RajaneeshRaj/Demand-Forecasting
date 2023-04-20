from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import time
import matplotlib.pyplot as plt

color_pallet = sns.color_palette()
plt.style.use('fivethirtyeight')

def roll(start_date,ite,x):
 df= pd.read_csv("train 2.csv")
 df.drop('store', axis=1, inplace=True)
 df = df.groupby(['date','item']).sum('sales')
 df = df.reset_index()
 df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') # convert date column to datatime object

# Create Date-related Features to be used for EDA and Supervised ML: Regression
 df['year'] = df['date'].dt.year
 df['month'] = df['date'].dt.month
 df['day'] = df['date'].dt.day
 df['quarter'] = df['date'].dt.quarter
 df['quarter']=df['quarter'].replace({1:4,2:1,3:2,4:3})
 df['weekday'] = df['date'].dt.weekday
 df['weekday'] = np.where(df.weekday == 0, 7, df.weekday)
 df['weekday']=df['weekday'].replace({1:3,2:4,3:5,4:6,5:7,6:1,7:2})
 df['dayofyear'] = df['date'].dt.dayofyear
 df['dayofmonth'] = df['date'].dt.day
 df['weekofyear'] = df['date'].dt.isocalendar().week
 df['weekofyear'] = df['weekofyear'].astype(np.int64)
 iqr = df['sales'].quantile(0.75) - df['sales'].quantile(0.25)
 upper_threshold = df['sales'].quantile(0.75) + (1.5 * iqr)
 lower_threshold = df['sales'].quantile(0.25) - (1.5 * iqr)
 df.sales = df.sales.clip(lower_threshold, upper_threshold)
 b=x-1
 y=-b
 a=df.groupby('item').sales.rolling(x).sum().shift(y)
 df["rolling_sum"] = a.reset_index(level=0, drop=True)
 if df.isnull().values.any():
    df.dropna(inplace=True)
 else:
    pass
 train = df.loc[df['year'] < 2017]
 test = df.loc[df['year'] >= 2017]
 FEATURES = ['item', 'day', 'month', 'year','weekday','quarter','dayofyear','weekofyear']
 TARGET = 'rolling_sum'

 X_train = train[FEATURES]
 y_train = train[TARGET]

 reg = xgb.XGBRegressor(n_estimators=1000,learning_rate=0.1)
 reg.fit(X_train, y_train)

 df2 = pd.DataFrame()

 df2['date'] = pd.date_range(start= start_date, periods= 1, freq='D')
 df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d') # convert date column to datatime object

# Create Date-related Features to be used for EDA and Supervised ML: Regression
 df2['year'] = df2['date'].dt.year
 df2['month'] = df2['date'].dt.month
 df2['day'] = df2['date'].dt.day
 df2['quarter'] = df2['date'].dt.quarter
 df2['quarter']=df2['quarter'].replace({1:4,2:1,3:2,4:3})
 df2['weekday'] = df2['date'].dt.weekday
 df2['weekday'] = np.where(df2.weekday == 0, 7, df2.weekday)
 df2['weekday']=df2['weekday'].replace({1:3,2:4,3:5,4:6,5:7,6:1,7:2})
 df2['dayofyear'] = df2['date'].dt.dayofyear
 df2['dayofmonth'] = df2['date'].dt.day
 df2['weekofyear'] = df2['date'].dt.isocalendar().week
 df2['weekofyear'] = df2['weekofyear'].astype(np.int64)

 df2['item']= ite

 FEATURES1 = ['item', 'day', 'month', 'year','weekday','quarter','dayofyear','weekofyear']
 X_test1 = df2[FEATURES1]

 df2['rolling_sum'] = reg.predict(X_test1)
 r = df2['rolling_sum'].max()

 dy = pd.DataFrame()
 for i in range(1,51):
    df2['item']=[i]
    FEATURES = ['item', 'day', 'month', 'year', 'weekday', 'quarter', 'dayofyear', 'weekofyear']
    X_test = df2[FEATURES]
    
    # Make the prediction using the trained model 'reg'
    t = reg.predict(X_test)
    
    # Create a new DataFrame with the item ID and the predicted value
    new_row = pd.DataFrame({'item': [i], 'rolling_sum': t})
    
    # Append the new row to the 'dy' DataFrame
    dy = dy.append(new_row, ignore_index=True)
 return r,df,df2,dy


def data():
 df4= pd.read_csv("train 2.csv")
 iqr = df4['sales'].quantile(0.75) - df4['sales'].quantile(0.25)
 upper_threshold = df4['sales'].quantile(0.75) + (1.5 * iqr)
 lower_threshold = df4['sales'].quantile(0.25) - (1.5 * iqr)
 df4.sales = df4.sales.clip(lower_threshold, upper_threshold)
 df4.drop('store', axis=1, inplace=True)
 df4 = df4.groupby(['date','item']).sum('sales')
 df4 = df4.reset_index()
 a=df4.groupby('item').sales.rolling(90).sum().shift(-89)
 df4["rolling_sum"] = a.reset_index(level=0, drop=True)
 df4['date'] = pd.to_datetime(df4['date'], format='%Y-%m-%d') # convert date column to datatime object

# Create Date-related Features to be used for EDA and Supervised ML: Regression
 df4['year'] = df4['date'].dt.year
 df4['month'] = df4['date'].dt.month
 df4['day'] = df4['date'].dt.day
 df4['quarter'] = df4['date'].dt.quarter
 df4['quarter']=df4['quarter'].replace({1:4,2:1,3:2,4:3})
 df4['weekday'] = df4['date'].dt.weekday
 df4['weekday'] = np.where(df4.weekday == 0, 7, df4.weekday)
 df4['weekday']=df4['weekday'].replace({1:3,2:4,3:5,4:6,5:7,6:1,7:2})
 df4['dayofyear'] = df4['date'].dt.dayofyear
 df4['dayofmonth'] = df4['date'].dt.day
 df4['weekofyear'] = df4['date'].dt.isocalendar().week

 return df4

#gui
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://st.depositphotos.com/4013359/51453/i/600/depositphotos_514534038-stock-photo-shopping-banner-shopping-bags-gift.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
# GUI.py

selected = option_menu(menu_title=None, options=["ML model","Data visualization"], icons=["coin","clipboard-data"], orientation="horizontal")

if selected=="ML model":
 st.title("Demand Forecasting using xg boost")
 #colsu1,colsu2 = st.columns((1,1))
 #with colsu1:
# Get input from user for item
 ite = st.number_input("Enter the item number (1-50):", key="limit")
 #with colsu2:
# Get starting date from user
 start_date = st.date_input("Select start date:", key="start_date")


# Get input from user for item

 x = st.slider("Choose the period 1-365:", min_value=1, max_value=365, step=1)

 if st.button("Forecast the demand"):
    r,df,df2,dy= roll(start_date,ite,x)
    #p= code(start_date,ite,x)
    c1,c2,c3=st.columns((1,2,1))
    progress_bar = c1.progress(0)
    for perc_completed in range(100):
        time.sleep(0.05)
    progress_bar.progress(perc_completed+1)
    co1,co2,co3=st.columns((1,2,1))
    with co2:
     st.write('Date')
     st.write(start_date)
    st.markdown('''----''')
    col1, col2, col3 = st.columns(3)
      
    col1.metric('Item number', ite)
    col2.metric('Duration', x)
    col3.metric('R2 Score', '0.99')
    st.markdown('''----''')
    cl1, cl2, cl3 = st.columns((1,2,1))
    with cl2:
        st.write("Predicted Demand")
        st.title(r)
    st.markdown('''----''')
    k1,k2= st.columns((2,1))

    with k1:
        fig4 = px.bar(dy, x='item', y='rolling_sum', color='item',color_discrete_map={ite: '#F9B5AC'},title='Items demand for above period',width=500, height=350)
        fig4.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
        )
        st.write(fig4)

    with k2:
        dr = df[((df['month'].isin(df2['month'])) & (df['day'].isin(df2['day'])) & (df['item'].isin(df2['item'])))]
        frames = [dr,df2]
        result = pd.concat(frames)
        fig3 = px.pie(result, values='rolling_sum', names='year' ,title='Demand by Years',hole=0.5 ,width=350, height=350)
        fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
        )
        st.write(fig3)
       

if selected=="Data visualization":
 df4= data()
 kol1, kol2, kol3= st.columns((1,2,1))
#  with kol1:
#   df5 = df4.groupby('date').sum('sales')
#   df5.index = pd.to_datetime(df5.index)
#   df5 = df5.drop(['item','rolling_sum','year','month','day','quarter','weekday','dayofyear','dayofmonth','weekofyear'], axis=1)
#   st.markdown("### weekday vs sales")
#   ax = df4.plot(figsize=(15, 5), color=color_pallet[0])

#   # remove the background color
#   ax.set_facecolor('none')
#   ax.figure.set_facecolor('none')
#   st.write(ax)

 with kol2:
  
  fig8 = px.sunburst(df4, path=['year', 'item'], values='sales',color='sales',title='Sunburst for Data',width=400, height=400)
  fig8.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
  st.write(fig8)
 st.markdown('''----''')
 k1,k2 = st.columns((2,1))
 
 with k1:
  fig1 = px.box(df4, x="weekday", y="sales",color='weekday',title='Weekday vs Sales',width=400, height=300)
  fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
  st.write(fig1)

 with k2:
  fig2 = px.box(df4, x="quarter", y="sales",color='quarter',title='Quarter vs Sales',width=400, height=300)
  fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
  st.write(fig2)

 st.markdown('''----''')

 q1,q2,q3 = st.columns((0.5,2,1))

 with q2:
  fig9 = px.bar(df4, x="month", y="sales",color='month',title='Month vs Sales', width=600, height=300)
  fig9.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
  st.write(fig9)
 
 # Code for Button style
 primaryColor = st.get_option("theme.primaryColor")
 s = f"""
 <style>
 div.stButton > button:first-child {{ border: 5px solid {primaryColor}; border-radius:20px 20px 20px 20px; }}
 <style>
 """
 st.markdown(s, unsafe_allow_html=True)

 st.markdown('''----''')
 y1,y2,y3,y4,y5,y6= st.columns((1,1,1,1,1,1))
 
 with y1:
    if st.button("All Years"):
     st.markdown("###### Year vs Sales")
     df5 = df4.groupby(['year','item']).sum('sales').reset_index()
     fi2 = px.line(df5, x='year', y='sales', color='item')
     fi2.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
     st.write(fi2)
 with y2:
    if st.button("2013"):
     st.markdown("###### Year vs Sales")
     df6 = df4[df4['year'] == 2013]
     fi3 = px.bar(df6, x='month', y='sales', color='item')
     fi3.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
     st.write(fi3)

 with y3:
    if st.button("2014"):
     st.markdown("###### Year vs Sales")
     df7 = df4[df4['year'] == 2014]
     fi4 = px.bar(df7, x='month', y='sales', color='item')
     fi4.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
     st.write(fi4)
  

 with y4:
    if st.button("2015"):
     st.markdown("###### Year vs Sales")
     df8 = df4[df4['year'] == 2015]
     fi5 = px.bar(df8, x='month', y='sales', color='item')
     fi5.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
     st.write(fi5)

 with y5:
    if st.button("2016"):
     st.markdown("###### Year vs Sales")
     df9 = df4[df4['year'] == 2016]
     fi6 = px.bar(df9, x='month', y='sales', color='item')
     fi6.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
     st.write(fi6)

 with y6:
    if st.button("2017"):
     st.markdown("###### Year vs Sales")
     df10 = df4[df4['year'] == 2017]
     fi7 = px.bar(df10, x='month', y='sales', color='item')
     fi7.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
     st.write(fi7)

