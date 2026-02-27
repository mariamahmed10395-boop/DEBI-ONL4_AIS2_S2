import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Ford GoBike Dashboard", layout="wide")
st.title("🚲 Ford GoBike Interactive Dashboard")

# 2. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_fordgobike_data.csv')
    # Note: Assuming start_time is converted to datetime in preprocessing
    # If not, we ensure it here for filtering
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    return df

df = load_data()

# 3. SIDEBAR FILTERS (Part 3, Point 2) [cite: 40, 86]
st.sidebar.header("Filters")

# User Type Filter
user_types = st.sidebar.multiselect("Select User Type", 
                                    options=df['user_type'].unique(), 
                                    default=df['user_type'].unique())

# Gender Filter
genders = st.sidebar.multiselect("Select Gender", 
                                 options=df['member_gender'].unique(), 
                                 default=df['member_gender'].unique())

# Age Group Filter
age_groups = st.sidebar.multiselect("Select Age Group", 
                                    options=df['age_group'].unique(), 
                                    default=df['age_group'].unique())

# Filtering the dataframe
filtered_df = df[
    (df['user_type'].isin(user_types)) &
    (df['member_gender'].isin(genders)) &
    (df['age_group'].isin(age_groups))
]

# 4. SECTION 1: OVERVIEW KPIs [cite: 47, 63]
st.header("1. Overview KPIs")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Trips", f"{len(filtered_df):,}") # [cite: 60]
with col2:
    avg_dur = filtered_df['duration_min'].mean()
    st.metric("Avg Duration", f"{avg_dur:.1f} mins") # [cite: 64]
with col3:
    st.metric("Active Users", f"{filtered_df['bike_id'].nunique():,}") # [cite: 67]
with col4:
    popular_station = filtered_df['start_station_name'].mode()[0] if not filtered_df.empty else "N/A"
    st.subheader("Most Popular Station")
    st.write(popular_station) # [cite: 68]

# 5. SECTION 2: TIME ANALYSIS [cite: 47, 61]
st.header("2. Time Analysis")
# Trips by Duration Distribution
fig_dur = px.histogram(filtered_df[filtered_df['duration_min'] < 60], 
                       x='duration_min', nbins=30, 
                       title="Trip Duration Distribution (under 60 min)",
                       color_discrete_sequence=['skyblue'])
st.plotly_chart(fig_dur, use_container_width=True)

# 6. SECTION 3: USER ANALYSIS [cite: 48, 103]
st.header("3. User Analysis")
u_col1, u_col2 = st.columns(2)

with u_col1:
    # Subscriber vs Customer (Pie Chart) [cite: 43, 84, 91]
    fig_user = px.pie(filtered_df, names='user_type', title="Subscriber vs Customer Usage", hole=0.4)
    st.plotly_chart(fig_user)

with u_col2:
    # Gender Distribution (Bar Chart) [cite: 92, 104]
    fig_gender = px.bar(filtered_df['member_gender'].value_counts().reset_index(), 
                    x='member_gender', y='count', 
                    labels={'member_gender': 'Gender', 'count': 'Trips'},
                    title="Trips by Gender")
    st.plotly_chart(fig_gender)

# 7. SECTION 4: STATION & TRIP ANALYSIS [cite: 49, 78]
st.header("4. Station Analysis")

# Top 10 Stations [cite: 44, 93]
# التعديل: استخدام المسميات الجديدة start_station_name و count
top_stations = filtered_df['start_station_name'].value_counts().head(10).reset_index()

fig_station = px.bar(top_stations, 
                     x='count',                    # ده العمود اللي فيه عدد الرحلات
                     y='start_station_name',       # ده العمود اللي فيه أسامي المحطات
                     orientation='h', 
                     title="Top 10 Most Popular Start Stations",
                     labels={'count': 'Number of Trips', 'start_station_name': 'Station Name'})

st.plotly_chart(fig_station, use_container_width=True)

st.success("Analysis and Dashboard Generated Successfully!")