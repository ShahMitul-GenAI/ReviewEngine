import time

import streamlit as st
from subprocess import call
import pandas as pd
import pathlib
import os.path
import shutil
import time
# import warnings
# warnings.filterwarnings("ignore")
import pickle

if 'data_collect' not in st.session_state:
    st.session_state.data_collect = {}
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = []

# function to execute backend python file
def run_file():
    call(["python", "customer_reviews.py"])

# function to export user inputs from form
def export_inputs(data):
    st.session_state['user_inputs'].append(data)
    with open("./notifications/data_inputs.pkl", "wb") as fp:
        pickle.dump(st.session_state.user_inputs, fp)

# displaying page title and header
st.title("User Inputs for e-Commerce Customer Reviews")
st.header("Provide the requested information")

# develop form

with st.form(key="user_interaction"):
    # get new data processing choice
    new_data_st = st.selectbox(label = "Please select your choice from the dropdown", options = ["Yes", "No"])

    # get name of product in couple to few words
    product = st.text_input(
        label = "Please describe your product in 2-5 phrases",
        max_chars = 30)

    # get number of total customer reviews to process
    max_cust = st.number_input(
        label="Select number of Amazon customer reviews to be collected",
        max_value=50,
        min_value=10,
        step=5,
    )
    st.session_state.user_inputs = [{"new_data_st": new_data_st, "product": product, "max_cust": max_cust}]
    submit_button = st.form_submit_button("Submit")

if submit_button:
    export_inputs(st.session_state.data_collect)

    if new_data_st == "Yes":
        file_path = "C:/Users/Mast_Nijanand/customer_review_app/"
        if os.path.exists(str(file_path) + "data_output.pkl"):
            shutil.move(str(file_path) + "data_output.pkl",
                        str(file_path) + "review_docs/data_output.pkl")
        if os.path.exists(str(file_path) + "amazon_reviews_df"):
            shutil.move(str(file_path) + "amazon_reviews_df",
                        str(file_path) + "review_docs/amazon_reviews_df")


        run_file()
        new_file_path = str(file_path) + "data_output.pkl"
        check_notification = str(file_path) + "notifications/"

        with st.spinner("Processing your request....."):
            while not os.path.exists(str(check_notification) + "PROG EXIT.txt"):
                i = 0
                j = 0
                display = []
                while not os.path.exists(str(check_notification) + "PROG EXIT.txt"):
                    if os.path.exists(str(check_notification) + ("note" + str(i) + ".txt")):
                        with open((str(check_notification) + ("note" + str(i) + ".txt")), 'r') as fp:
                            lines = fp.read()
                            line = lines.splitlines()
                            for each in line:
                                display.append(each)
                        i += 1
                    if (os.path.exists(str(check_notification) + 'LARGE.txt')) and (j == 0):
                        i = 3
                        j = 1
                    if os.path.exists(str(check_notification) + ("SMALL.txt")) or (i == 6):
                        break
            for each in display:
                st.write(each)
        st.success("Data Processing Complete!")
        df = pd.DataFrame()
        df = pd.read_pickle("amazon_reviews_df.pkl")
        st.write(df)
    else:
        st.write("Displaying results of previous product reviews")
        df = pd.DataFrame()
        df = pd.read_pickle("amazon_reviews_df.pkl")
        st.write(df)

















