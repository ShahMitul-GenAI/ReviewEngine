import pandas as pd
import streamlit as st
from customer_reviews.reviews_summary import get_review_summary

# displaying page title and header
st.title("E-Commerce Customer Review Engine")
st.header("Please provide the requested information")

# develop form
with st.form(key="user_interaction"):
    
    #select your product input option
    input_options = st.empty()
    prodcut_query = st.empty()  

    # get number of total customer reviews to process
    cust_count = st.slider(
        label="Select maximum number of Amazon customer reviews to be collected",
        max_value=200,
        min_value=10,
        step=5,
        key = "max_cust"
    )
    submit_button = st.form_submit_button("Submit")

# setting up product query fields based on the input selection option
with input_options: 

    inp_opt = st.radio(
        label = "Please select your product selection option.",
        options = ["ASIN", "Description"],
        captions = ["I have a product ASIN", "I will use most relevant product phrases"]
    )

with prodcut_query: 
   
    input_selection = dict(
        ASIN = ["Please input your product ASIN: ", 10 ],
        Description = ["Please describe your product in a couple of words: ", 50]
        )
    
    prod_query = st.text_input(
    label = input_selection[inp_opt][0],
    max_chars = input_selection[inp_opt][1]

    )

if submit_button:
    
    with st.spinner("Processing your data now...."):
        # getting outputs now 
        tokens, df_reviews, summary_small, summary_map, summary_refine = get_review_summary(inp_opt, prod_query, cust_count)
        
        # displaying customer reviews in dataframe format 
        df = df_reviews.head(10)
        st.markdown(" ### Top 10 Customer Reviews: \n")
        st.dataframe(df, hide_index=True)

    st.success("Data Processing Complete!")

    if tokens <= 3500:
        
        # displying small customer reviews summary
        st.markdown(" ### The customer reviews can be summarized as: \n")
        st.write(summary_small["text"])

    else:
        # for larger review content
        st.markdown(" ##### The Customer Reviews content is large. It warrants the use of 'Map Reduce' and 'Refine Method' for summary generation. \n")
        
        # displaying Map Reduce customer review summary
        st.markdown(" ### The customer reviews summary using Map Reduce Method: \n")
        st.write(summary_map["output_text"])

        # displaying Refine Method customer review summary
        st.markdown(" ### The customer reviews summary using Refine Method:: \n")
        st.write(summary_refine["output_text"])


















