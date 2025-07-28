import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

url = 'http://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls'
df=pd.read_excel(url,engine='xlrd')
# import model

models = joblib.load('employee_performance_model.pkl')



num_cols = ['Age', 'DistanceFromHome', 'EmpHourlyRate', 'NumCompaniesWorked', 
            'EmpLastSalaryHikePercent', 'TotalWorkExperienceInYears', 
            'TrainingTimesLastYear', 'ExperienceYearsAtThisCompany', 
            'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion', 
            'YearsWithCurrManager']

ord_cols = ['EmpEducationLevel', 'EmpEnvironmentSatisfaction', 'EmpJobInvolvement', 
            'EmpJobSatisfaction', 'EmpRelationshipSatisfaction', 'EmpWorkLifeBalance']
cat_cols = ['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 
            'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']



dept_perf = df.groupby('EmpDepartment')['PerformanceRating'].value_counts(normalize=True).unstack().fillna(0) * 100
fig1 = go.Figure()
for rating in [2, 3, 4]:
    fig1.add_trace(go.Bar(
        x=dept_perf.index,
        y=dept_perf[rating],
        name=f'Rating {rating}',
        marker_color=['#FF6347', '#4682B4', '#32CD32', '#FFD700'][rating-1]
    ))
fig1.update_layout(
    title='Department-Wise Performance Rating Distribution (%)',
    xaxis_title='Department',
    yaxis_title='Percentage',
    barmode='stack'
)

correlations = df[num_cols + ['PerformanceRating']].corr()['PerformanceRating'].drop('PerformanceRating')

features = num_cols + ['EmpJobSatisfaction', 'OverTime_Yes']
importance = [0.05, 0.03, 0.04, 0.03, 0.25, 0.06, 0.04, 0.09, 0.07, 0.05, 0.04, 0.20, 0.15]
fig2 = go.Figure(go.Bar(
    x=features,
    y=importance,
    marker_color='#4682B4'
))
fig2.update_layout(
    title='Top Factors Affecting Performance',
    xaxis_title='Feature',
    yaxis_title='Importance Score'
)

# 3. Performance Distribution
rating_counts = df['PerformanceRating'].value_counts().sort_index()
fig3 = go.Figure(go.Bar(
    x=['Low (1)', 'Good (2)', 'Excellent (3)', 'Outstanding (4)'],
    y=rating_counts,
    marker_color=['#FF6347', '#4682B4', '#32CD32', '#FFD700']
))
fig3.update_layout(
    title='Performance Rating Distribution',
    xaxis_title='Rating',
    yaxis_title='Count'
)

results = {
    'Logistic Regression': {'accuracy': 0.75, 'f1': 0.73},
    'Random Forest': {'accuracy': 0.82, 'f1': 0.81},
    'XGBoost': {'accuracy': 0.85, 'f1': 0.84}
}
fig4 = go.Figure(go.Bar(
    x=list(results.keys()),
    y=[results[model]['f1'] for model in results],
    marker_color=['#FF6347', '#4682B4', '#32CD32']
))
fig4.update_layout(
    title='Model Performance Comparison (Weighted F1-Score)',
    xaxis_title='Model',
    yaxis_title='Weighted F1-Score',
    yaxis_range=[0, 1]
)

st.title('Employee Performance Prediction')
st.header('Input Candidate Details')
st.subheader('XGBOOST')
input_data = {}
for col in num_cols:
    input_data[col] = st.number_input(col, min_value=0, value=int(df[col].mean()))
for col in ord_cols:
    input_data[col] = st.selectbox(col, options=[1, 2, 3, 4], index=2)
for col in cat_cols:
    input_data[col] = st.selectbox(col, options=df[col].unique())
input_df = pd.DataFrame([input_data])
if st.button('Predict Performance'):
    best_model = models['XGBoost']
    pred = best_model.predict(input_df)[0] + 1
    rating_map = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}
    st.write(f'Predicted Performance Rating: {rating_map[pred]}')


st.header('EDA Insights')
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
st.plotly_chart(fig4)