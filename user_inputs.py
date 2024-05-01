import time

import streamlit as st
from subprocess import call
import pandas as pd
import pathlib
import os.path
import shutil
import time
import warnings
warnings.filterwarnings("ignore")


def run_file():
    call(["python", "customer_reviews-Krish.py"])

# displaying page title and header
st.title("User Inputs for e-Commerce Customer Reviews")
st.header("Provide the requested information")

# develop form

with st.form(key="user_inputs"):
    # get new data processing choice
    new_data_st = st.selectbox(label = "Please select your choice from the dropdown", options = ["Yes", "No"])

    # get name of product in couple to few words
    product = st.text_input(
        label = "Please describe your product in 2-5 phrases",
        max_chars = 30)

    # get number of total customer reviews to process
    max_cus = st.number_input(
        label="Select number of Amazon customer reviews to be collected",
        max_value=50,
        min_value=10,
        step=5,
    )
    submit_button = st.form_submit_button("Submit")

if submit_button:
    if new_data_st == "Yes":
        file_path = "C:/Users/Mast_Nijanand/customer_review_app/data output"
        if os.path.isfile(file_path):
            shutil.move(file_path, "C:/Users/Mast_Nijanand/customer_review_app/review_docs/data output")
            shutil.move("C:/Users/Mast_Nijanand/customer_review_app/amazon_reviews_df", \
                        "C:/Users/Mast_Nijanand/customer_review_app/review_docs/amazon_reviews_df")

        run_file()

        file_exist = os.path.isfile(file_path)
        with st.spinner("Processing your request....."):
            while not os.path.isfile(file_path):
                time.sleep(9)
                file_exist = os.path.isfile(file_path)
        st.success("Data Processing Complete! Displaying results of new product reviews now.")
        df = pd.DataFrame()
        df = pd.read_pickle("amazon_reviews_df")
        summary = pd.read_csv("data output.csv")
        st.write(df)
        st.write(summary)
    else:
        st.write("Displaying results of previous product reviews")
        df = pd.DataFrame()
        df = pd.read_pickle("amazon_reviews_df")
        summary = pd.read_csv("data output.csv")
        st.write(df)
        st.write(summary)

















