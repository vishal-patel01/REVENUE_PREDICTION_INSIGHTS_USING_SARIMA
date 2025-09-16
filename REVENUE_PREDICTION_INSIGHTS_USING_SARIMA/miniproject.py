import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import pickle
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Revenue", page_icon=":hotel:", layout="wide")

# Sidebar
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Select Page", ["Dashboard", "Category-wise Prediction", "Forecast Prediction"])

# Main Page
if app_mode == "Dashboard":
    st.title(":hotel: Hotel Revenue Analysis")
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)


    fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
    if fl is not None:
      # Read the uploaded file directly from the file object (fl)
      df = pd.read_csv(fl, encoding="ISO-8859-1")
      
    else:
      # If no file is uploaded, use the specified file path
      filename = "Revenue.csv"
      df = pd.read_csv(filename, encoding="ISO-8859-1")
      


    col1, col2 = st.columns((2))
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

    # Getting the min and max date
    startDate = pd.to_datetime(df["date"]).min()
    endDate = pd.to_datetime(df["date"]).max()

    with col1:
      date1 = pd.to_datetime(st.date_input("Start Date", startDate))

    with col2:
      date2 = pd.to_datetime(st.date_input("End Date", endDate))

    df = df[(df["date"] >= date1) & (df["date"] <= date2)].copy()

    st.sidebar.header("Choose your filter: ")
    # Create for Hotel Name
    property = st.sidebar.multiselect("Pick the Hotel", df["property_name"].unique())
    if not property:
      df2 = df.copy()
    else:
      df2 = df[df["property_name"].isin(property)]

    # Create for City
    city_name = st.sidebar.multiselect("Pick the City", df2["city"].unique())
    if not city_name:
      df3 = df2.copy()
    else:
      df3 = df2[df2["city"].isin(city_name)]

    # Create for Room Class
    room = st.sidebar.multiselect("Pick the Room Class", df3["room_class"].unique())

    if not property and not city_name and not room:
      filtered_df = df
    elif not city_name and not room:
      filtered_df = df[df["property_name"].isin(property)]
    elif not property and not room:
      filtered_df = df[df["city"].isin(city_name)]
    elif city_name and room:
      filtered_df = df3[df["city"].isin(city_name) & df3["room_class"].isin(room)]
    elif property and room:
      filtered_df = df3[df["property_name"].isin(property) & df3["room_class"].isin(room)]
    elif property and city_name:
      filtered_df = df3[df["property_name"].isin(property) & df3["city"].isin(city_name)]
    elif room:
      filtered_df = df3[df3["room_class"].isin(room)]
    else:
      filtered_df = df3[df3["property_name"].isin(property) & df3["city"].isin(city_name) & df3["room_class"].isin(room)]

    with col1:
    # Apply filters to the DataFrame
      filtered_day_type_df = filtered_df.groupby(by="day_type", as_index=False)["revenue_realized"].sum()
      st.subheader("Daytype wise Revenue")
      fig = px.bar(filtered_day_type_df, x="day_type", y="revenue_realized", text=['Rs.{:,.2f}'.format(x) for x in filtered_day_type_df["revenue_realized"]],
                 template="seaborn")
      fig.update_layout(xaxis_title='Day Type', yaxis_title='Revenue')
      st.plotly_chart(fig, use_container_width=True, height=200)

    with col2:
      st.subheader("Property wise Revenue")
      fig = px.pie(filtered_df, values="revenue_realized", names="property_name", hole=0.5)
      fig.update_traces(text=filtered_df["property_name"], textposition="outside")
      st.plotly_chart(fig, use_container_width=True)

    cl1, cl2 = st.columns((2))
    with cl1:
      with st.expander("Daytype ViewData"):
        st.write(filtered_day_type_df.style.background_gradient(cmap="Blues"))
        csv = filtered_day_type_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Category.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')

    with cl2:
      with st.expander("Property ViewData"):
        region = filtered_df.groupby(by="property_name", as_index=False)["revenue_realized"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Property.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')

    # Time Series Plot
    st.subheader('Time Series Analysis')

    linechart = pd.DataFrame(filtered_df.groupby("mmm yy")["revenue_realized"].sum()).reset_index()
    linechart['mmm yy'] = pd.to_datetime(linechart['mmm yy'], format="%b-%y")
    linechart = linechart.sort_values('mmm yy')
    fig2 = px.line(linechart, x="mmm yy", y="revenue_realized", labels={"revenue_realized": "Revenue"}, height=500, width=1000, template="gridon")
    fig2.update_layout(xaxis_title='Month', yaxis_title='Revenue')
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View Data of TimeSeries:"):
      st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data=csv, file_name="TimeSeries.csv", mime='text/csv')

    chart1, chart2 = st.columns(2)

    with chart1:
      booking_by_platform = filtered_df.groupby('booking_platform')['successful_bookings'].sum()
      st.subheader('Booking Across Different Platforms')
      fig = px.bar(x=booking_by_platform.index, y=booking_by_platform.values, template="plotly_dark")
      fig.update_traces(text=booking_by_platform.values, textposition="inside")
      fig.update_layout(xaxis_title="Booking Platform", yaxis_title="Total Successful Bookings")
      st.plotly_chart(fig, use_container_width=True)

    with chart2:
      filtered_df['RevPAR'] = filtered_df['revenue_generated'] / filtered_df['capacity']
      st.subheader('Revenue Per Available Room (RevPAR)')
      fig = px.pie(filtered_df, values='RevPAR', names='room_class', template='plotly_dark')
      fig.update_traces(textinfo='percent+label', textposition='inside')
      fig.update_layout(title='RevPAR Distribution by Room Class')
      st.plotly_chart(fig, use_container_width=True)

    # ADR lineplot
    filtered_df['ADR'] = filtered_df['revenue_realized'] / filtered_df['successful_bookings']
    adr_by_city_month = filtered_df.groupby(['city', 'mmm yy'])['ADR'].mean().reset_index()
    adr_by_city_month['mmm yy'] = pd.to_datetime(adr_by_city_month['mmm yy'], format="%b-%y")

    st.subheader('Average Daily Rate (ADR) in Different Cities Across Months')
    adr_by_city_month = adr_by_city_month.sort_values('mmm yy')
    fig = px.line(adr_by_city_month, x='mmm yy', y='ADR', color='city')
    fig.update_layout(xaxis_title='Month', yaxis_title='ADR', legend_title='City')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Data of ADR in Different Cities Across Months:"):
      st.write(adr_by_city_month.T.style.background_gradient(cmap="Blues"))
      csv = adr_by_city_month.to_csv(index=False).encode("utf-8")
      st.download_button('Download Data', data=csv, file_name="ADR.csv", mime='text/csv')

    # Occupancy % 
    filtered_df['Occupancy %'] = filtered_df['successful_bookings'] / filtered_df['capacity'] * 100
    Occupancy_by_city_month = filtered_df.groupby(['city', 'mmm yy'])['Occupancy %'].mean().reset_index()
    Occupancy_by_city_month['mmm yy'] = pd.to_datetime(Occupancy_by_city_month['mmm yy'], format="%b-%y")

    st.subheader('Occupancy % in Different Cities Across Months')
    Occupancy_by_city_month = Occupancy_by_city_month.sort_values('mmm yy')
    fig = px.line(Occupancy_by_city_month, x='mmm yy', y='Occupancy %', color='city')
    fig.update_layout(xaxis_title='Month', yaxis_title='Occupancy %', legend_title='City')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Data of Occupancy % in Different Cities Across Months:"):
      st.write(Occupancy_by_city_month.T.style.background_gradient(cmap="Blues"))
      csv = Occupancy_by_city_month.to_csv(index=False).encode("utf-8")
      st.download_button('Download Data', data=csv, file_name="Occupancy %.csv", mime='text/csv')

    ch1, ch2 = st.columns(2)

    with ch1:
      revenue_by_city = filtered_df.groupby('city')['revenue_realized'].sum()
      st.subheader('Total Revenue in Different Cities')
      fig1 = px.bar(x=revenue_by_city.index, y=revenue_by_city.values, template="plotly_dark")
      fig1.update_traces(text=revenue_by_city.values, textposition="inside")
      fig1.update_layout(xaxis_title="Cities", yaxis_title="Revenue")
      st.plotly_chart(fig1, use_container_width=True)

    with ch2:
      st.subheader('Booking Status in Different Room Classes')
      fig2 = px.histogram(filtered_df, x='room_class', color='booking_status',
                        barmode='group', labels={'room_class': 'Room Class', 'count': 'Number of Bookings'})
      fig2.update_layout(xaxis_title='Room Class', yaxis_title='Number of Bookings')
      st.plotly_chart(fig2)

    # Convert 'mmm yy' column to datetime
    filtered_df['mmm yy'] = pd.to_datetime(filtered_df['mmm yy'], format='%b-%y')

    # Group by 'mmm yy' instead of 'date'
    adr_by_month = filtered_df.groupby(pd.Grouper(key='mmm yy')).agg({'ADR': 'mean'}).reset_index()
    revpar_by_month = filtered_df.groupby(pd.Grouper(key='mmm yy')).agg({'RevPAR': 'mean'}).reset_index()
    occupancy_by_month = filtered_df.groupby(pd.Grouper(key='mmm yy')).agg({'Occupancy %': 'mean'}).reset_index()

    # Merge Data
    combined_data = pd.merge(adr_by_month, revpar_by_month, on='mmm yy', how='outer')
    combined_data = pd.merge(combined_data, occupancy_by_month, on='mmm yy', how='outer')

    # Plot Combined Graph
    st.subheader('Trend By Key Metrics')
    fig_combined = px.line(combined_data, x='mmm yy', y=['ADR', 'RevPAR', 'Occupancy %'],
                        labels={'value': 'Metrics', 'mmm yy': 'Month-Year'},
                        color_discrete_map={'ADR': 'blue', 'RevPAR': 'red', 'Occupancy %': 'green'})
    st.plotly_chart(fig_combined, use_container_width=True)

    with st.expander("View Data of Trend by Key Metrics"):
      st.write(combined_data.T.style.background_gradient(cmap="Blues"))
      csv = combined_data.to_csv(index=False).encode("utf-8")
      st.download_button('Download Data', data=csv, file_name="Trend.csv", mime='text/csv')

    pd.set_option("styler.render.max_elements", 11000)

    # Display the filtered data in an expander with all columns
    with st.expander("View Data"):
      st.write(df.iloc[:500].style.background_gradient(cmap="Oranges"))

    # Download the original DataSet
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")



# About Project
elif app_mode == "Category-wise Prediction":
    import streamlit as st
    import numpy as np
    import pickle
    # Load the trained model
    pipe = pickle.load(open('CatBoostModel.pkl', 'rb'))


    # Define numerical values for room categories, property names, categories, cities, and room classes
    room_categories = ['RT1', 'RT2', 'RT3', 'RT4']
    property_names = ['Atliq Grands', 'Atliq Exotica', 'Atliq City', 'Atliq Blu', 'Atliq Bay', 'Atliq Palace', 'Atliq Seasons']
    categories = ['Luxury', 'Business']
    cities = ['Delhi', 'Mumbai', 'Hyderabad', 'Banglore']
    room_classes = ['Standard', 'Elite', 'Premium', 'Presidential']

    def predict_revenue(room_category, property_name, category, city, room_class, no_guests, revenue_generated,
                        successful_bookings, capacity, ADR, RevPAR, occupancy):
        # Convert categorical inputs to numerical
        room_category_num = room_categories.index(room_category)
        property_name_num = property_names.index(property_name)
        category_num = categories.index(category)
        city_num = cities.index(city)
        room_class_num = room_classes.index(room_class)

        # Make prediction
        test = np.array([[room_category_num, property_name_num, category_num, city_num, room_class_num, no_guests,
                          revenue_generated, successful_bookings, capacity, ADR, RevPAR, occupancy]])
        prediction = pipe.predict(test)
        return prediction

    # Streamlit UI
    st.title('Hotel Category-wise Revenue Prediction')

    # Input fields
    room_category = st.selectbox('Room Category', ['Select'] + room_categories)
    property_name = st.selectbox('Property Name', ['Select'] + property_names)
    category = st.selectbox('Category', ['Select'] + categories)
    city = st.selectbox('City', ['Select'] + cities)
    room_class = st.selectbox('Room Class', ['Select'] + room_classes)
    no_guests = st.number_input('Number of Guests', min_value=0, step=1)
    revenue_generated = st.number_input('Revenue Generated', min_value=0, step=1)
    successful_bookings = st.number_input('Successful Bookings', min_value=0, step=1)
    capacity = st.number_input('Capacity', min_value=0, step=1)
    ADR = st.number_input('ADR', step=0.01)
    RevPAR = st.number_input('RevPAR', step=0.01)
    occupancy = st.number_input('Occupancy %', min_value=0.0, max_value=100.0, step=0.1)

    if st.button('Predict Revenue'):
        if room_category == 'Select' or property_name == 'Select' or category == 'Select' or city == 'Select' or \
                room_class == 'Select' or no_guests == 0 or revenue_generated == 0 or successful_bookings == 0 or \
                capacity == 0 or ADR == 0.0 or RevPAR == 0.0 or occupancy == 0:
            st.warning('Please fill all required fields before predicting.')
        else:
            prediction = predict_revenue(room_category, property_name, category, city, room_class, no_guests,
                                         revenue_generated, successful_bookings, capacity, ADR, RevPAR, occupancy)
            st.success(f'Realized Revenue: Rs. {prediction[0][0]}')



# Prediction Page
elif app_mode == "Forecast Prediction":
  
    import streamlit as st
    import pandas as pd
    import pickle
    import plotly.graph_objects as go
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import csv

    # Load pickle files for each hotel
    with open('sarimax_model1.pkl', 'rb') as file:
      hotel1_model = pickle.load(file)

    with open('sarimax_model2.pkl', 'rb') as file:
      hotel2_model = pickle.load(file)

    with open('sarimax_model3.pkl', 'rb') as file:
      hotel3_model = pickle.load(file)

    with open('sarimax_model4.pkl', 'rb') as file:
      hotel4_model = pickle.load(file)

    with open('sarimax_model5.pkl', 'rb') as file:
      hotel5_model = pickle.load(file)

    with open('sarimax_model6.pkl', 'rb') as file:
      hotel6_model = pickle.load(file)

    with open('sarimax_model7.pkl', 'rb') as file:
      hotel7_model = pickle.load(file)

    hotel_category = ['Atliq Grands', 'Atliq Exotica', 'Atliq City', 'Atliq Blu', 'Atliq Bay', 'Atliq Palace',
                  'Atliq Seasons']

    # Function to make predictions
    def make_predictions(model, forecast_period):
      # Make predictions
      forecast = model.forecast(steps=forecast_period)
      return forecast

    # Main content
    st.title('Hotel Forecast Revenue Prediction')
    
    selected_hotel = st.selectbox('Select Hotel', ['Select'] + hotel_category)

    if selected_hotel:
      forecast_period = st.number_input('Enter number of days to predict', min_value=1, max_value=30,
                                              value=1, step=1)

    forecast = None

    # Button to trigger prediction
    if st.button('Predict') and selected_hotel:
    # Call function to make predictions based on the selected hotel
      if selected_hotel == 'Atliq Grands':
        forecast = make_predictions(hotel1_model, forecast_period)
      elif selected_hotel == 'Atliq Exotica':
        forecast = make_predictions(hotel2_model, forecast_period)
      elif selected_hotel == 'Atliq City':
        forecast = make_predictions(hotel3_model, forecast_period)
      elif selected_hotel == 'Atliq Blu':
        forecast = make_predictions(hotel4_model, forecast_period)
      elif selected_hotel == 'Atliq Bay':
        forecast = make_predictions(hotel5_model, forecast_period)
      elif selected_hotel == 'Atliq Palace':
        forecast = make_predictions(hotel6_model, forecast_period)
      elif selected_hotel == 'Atliq Seasons':
        forecast = make_predictions(hotel7_model, forecast_period)
      else:
        st.write("Invalid selection. Please select a hotel.")

      # Display the predictions
      if forecast is not None:
        st.write(f'Predicted Revenue for the next {forecast_period} days for {selected_hotel}:')
        st.write(forecast)

        # Plot the predicted revenue
        fig = go.Figure(data=go.Scatter(x=list(range(1, forecast_period + 1)), y=forecast, mode='lines+markers'))
        fig.update_layout(title=f'Predicted Revenue for {selected_hotel}', xaxis_title='Days',
                          yaxis_title='Revenue')
        st.plotly_chart(fig)
        
      start_date = pd.Timestamp(2023, 5, 1).date()  # Start date is May 1, 2023
      date_range = pd.date_range(start=start_date, periods=forecast_period)
      forecast_df = pd.DataFrame({'Date': date_range, 'Predicted Revenue': forecast})

      # Provide download link for CSV
      csv_data = forecast_df.to_csv(index=False)
      st.download_button(label="Download Predicted Revenue", data=csv_data, file_name='predicted_revenue.csv', mime='text/csv')

