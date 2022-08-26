# -*- coding: utf-8 -*-
"""
Multiple Disease Prediction

"""

import pickle 
import streamlit as st
from streamlit_option_menu import option_menu
from xgboost import XGBClassifier
import pandas as pd 
from PIL import Image
from sklearn.preprocessing import RobustScaler

# loading the models and data 
loan_prediction = pickle.load(open('loan_prediction.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Home', 
                           ['About The Dataset',
                            'Loan Prediction'],
                           icons = ['bar-chart', 'pc-display'],
                           default_index= 0)
    
# -------------------------------------------------- ABOUT THE DATASET ------------------------------------------------------
msno_raw = Image.open("msno_raw.png")
msno_clean = Image.open("msno_clean.png")
approve = Image.open("approved.jpg")
reject = Image.open("reject.jpg")

if (selected == 'About The Dataset'):
    st.title("Predicting Loan Payment")
    img = Image.open("loan.jpg")
    st.image(img)
    st.write("""
             This application is made to provide an overview of loan_data_2007_2014 from Kaggle. It is a collection of data that provides customer information on 
             the company that offers the loan.
             
             The data were provided from this [source](https://www.kaggle.com/datasets/devanshi23/loan-data-2007-2014)
             
             The description from each column can you see [here](https://docs.google.com/spreadsheets/d/1iT1JNOBwU4l616_rnJpo0iny7blZvNBs/edit#gid=1666154857)
                                                                 
             This prediction is made to predict whether customers fail to make loan payments or not.
            """)
    st.header("Model Definition")
    st.write("""
             The model is trained with train data, then 4 candidate models are selected. 
             
             That is:
            
            - Logistic Regression
                 
            - Decission Tree
                 
            - Random Forest
                 
            - Xgboost
                 
            The four models were optimized using cross validation. The final model chosen is Xgboost which has been optimized. I use Recall and FPR as metrics, 
            I want to minimize FPR to reduce the number of errors predicting positive value (Customer who can pay off the loan).
             """)
    
# ---------------------------------------------MODEL PREDICTION---------------------------------------------
else: 
    st.title("Predicting Loan Payment")
    st.header("Model Prediction")
    
    # get grade
    grade = st.selectbox(
        "Select Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        help= "Loan Grade"
        )
    
    if grade == 'A':
        grade_a = 1
        grade_b = 0
    elif grade == 'B':
        grade_a = 0
        grade_b = 1
    else: 
        grade_a = 0
        grade_b = 0
        
    
    # get emp_length
    emp_length = st.selectbox(
        "Years of Work", ['1 Year', '2 Years', '3 Years', '4 Years', '5 Years', '6 Years', '7 Years', '8 Years', '9 Years', '10 Years or more'],
        help= "How many years have you worked ?"
        )
    
    if emp_length == '10 Years or more':
        emp_length = 1
    else:
        emp_length = 0
        
    
    # get home_ownership
    home_ownership = st.selectbox(
        "Type of Home Ownership", ['Mortage', 'Rent', 'Own', 'Other'],
        help= "Home ownership status"
        )
    
    if home_ownership == 'Rent':
        home_ownership = 1
    else:
        home_ownership = 0
    
    
    # get verification status 
    verification_status = st.selectbox(
        "Verification Status", ['Verified', 'Source Verified', 'Not Verified'],
        help= "Account Verification Status"
        )
    
    if verification_status == 'Not Verified':
        verif_not = 1
        verif_yes = 0
    elif verification_status == 'Verified':
        verif_yes = 1
        verif_not = 0
    else: 
        verif_yes = 0
        verif_not = 0
             
    
    # get purpose 
    purpose = st.selectbox(
        "Loan Purpose", ['Credit Card', 'Debt Consolidation', 'Home Improvement', 'Other'],
        help= "A category provided by the borrower for the loan request"
        )
    
    if purpose == 'Debt Consolidation':
        purpose = 1
    else:
        purpose = 0
    
    
    # get list initial 
    list_initial = st.radio("Initial Listing Status", ('Whole', 'Fractional'),
                    help = 'The initial listing status of the loan')
    
    if list_initial == 'Whole':
        list_initial = 1
    else:
        list_initial = 0
        
    
    # application type 
    application_type = 1
    
    # interest rate on the loan 
    interest_rate = st.number_input(
        "Interest Rate on the loan", min_value = 1.0, step = 0.01
        )
    st.write("The interest rate received is", interest_rate, "%")
    
    # interest received to date
    interest_received_to_date = st.number_input(
        "Total interest received to date", min_value = 1.0, step = 0.01
        )
    st.write("Total interest received is $", interest_received_to_date)
    
    # inquriry in the last 6 months
    inq_last_6mths = st.slider(
        "The number of inquiries in the last 6 months", 0, 50, 1
        )
    
    # remaining principal credit
    remaining_principal_credit = st.number_input(
        "Remaining principal credit", min_value = 0.0, step = 0.01,
        help = "Remaining outstanding principal for a portion of the total amount funded by investors"
        )   
    
    # total_rec_late_fee
    late_fees = st.number_input(
        "Late fees received to date", min_value = 0.0, step = 0.01
        ) 
    
    # collection_recovery_fee
    recovery_fee = st.number_input(
        "Charge of collection fee", min_value = 0.0, step = 0.01
        ) 
    
    # last_payment
    last_payment = st.number_input(
        "Last total payment amount received", min_value = 1.0, step = 0.01
        )
    
    # month_since_last_crdt
    month_last_crdt = st.number_input(
        "The month since last credit pull", min_value = 1.0, step = 0.01
        )
    
    # code for Prediction
    predict_result = ''
    
    # pre-processing
    data_pred = {'grade_a':grade_a, 'grade_b':grade_b, 'emp_length_10':emp_length, 'home_ownership_rent':home_ownership, 'verif_yes':verif_yes, 'verif_not':verif_not, 'purpose':purpose,
                 'list_initial':list_initial, 'application_type':application_type, 'interest_rate':interest_rate, 'interest_received_to_date':interest_received_to_date, 'inq_last_6mths':inq_last_6mths,
                 'remaining_principal_credit':remaining_principal_credit, 'late_fees':late_fees, 'recovery_fee':recovery_fee, 'last_payment':last_payment, 'month_last_crdt':month_last_crdt}
    df_pred = pd.DataFrame([data_pred])
    rs = RobustScaler()
    pred_scaled = pd.DataFrame(rs.fit_transform(df_pred), columns= rs.feature_names_in_)
    
    # creating a button for Prediction    
    if st.button("Predicting Result"):
        
        loan_prediction = loan_prediction.predict(pred_scaled)                      
        
        if (loan_prediction[0] == 1):
          predict_result = "Customer can pay the loan"
          st.image(approve)
        else:
          predict_result = "Customer can not pay the loan"
          st.image(reject)
        
    st.success(predict_result)
