#!/usr/bin/env python
# coding: utf-8

# ## Instaling the required packages

# In[1]:


import os
import time
import json
import requests
import tiktoken
import pandas as pd
from amz_scraper import AmazonScraper
from bs4 import BeautifulSoup
from selenium import webdriver
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv, dotenv_values
# from langchain.docstore.document import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_text_splitters import CharacterTextSplitter
# from langchain.chains import LLMChain, StuffDocumentsChain
# from langchain_community.document_loaders import TextLoader
# from langchain.chains.summarize import load_summarize_chain
# from langchain_community.callbacks import get_openai_callback
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

# load_dotenv()


# #### Decide about loading new data

# In[2]:


# importing inputs from the UI 
from user_inputs import new_data_st, max_cus, product
# new_data_st = "Yes"
# max_cus = 15
# product = "ipad air"


# In[3]:


print(product)
print(max_cus)
print(new_data_st)

# #### Setting up small vs. large content code toggle

# In[3]:


if new_data_st == "Yes":
    new_data = True
else:
    new_data = False


# In[4]:


# Setting up logical code block execution toggle
class StopExecution(Exception):
    def _render_traceback_(self):
        pass


# #### Getting API Keys

# In[5]:


# Activating the API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
X_RapidAPI_Key = os.environ.get("X_RapidAPI_Key")
Rapid_AI_URL = os.environ.get("Rapid_AI_URL")
Rapid_AI_Host = os.environ.get("Rapid_AI_Host")


# ## New Product Selection

# ### Getting Amazon Reivews

# In[6]:


if new_data:
    search_query = str(product)   # 'premier protein shake, chocolate'
    scraper = AmazonScraper()
    reviews = scraper.get_closest_product_reviews(search_query, num_reviews = max_cus, debug=True)
    print(reviews)


# In[ ]:


search_query


# In[ ]:


print("Total number of reviewes received: ", len(reviews))


# In[ ]:


import sys
sys.exit(0)


# In[ ]:


# Transfering the webscrapped data into a dataframe
if new_data:
    df = pd.DataFrame.from_dict(reviews)
    df.sort_values(by=["rating"], ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df.to_pickle("amazon_reviews_df")  # storing df to ease its future usage
    df.head(5)
else:
    df = pd.read_pickle("amazon_reviews_df")


# In[ ]:


df


# ### Preparing Review Data for Display 

# # extracting API reponse in json file
# temp_dict = response.json()
# 
# # declaring dataframe to collect final review data
# df = pd.DataFrame()
# 
# df["review"] = [t["body"] for t in temp_dict["reviews"]]
# df["review date"] = [(t["date"]) for t in temp_dict["reviews"]]
# df["ratings"] = [t["rating"] for t in temp_dict["reviews"]]
# df["review title"] = [t["title"] for t in temp_dict["reviews"]]
# df["user name"]= [t["user-name"] for t in temp_dict["reviews"]]
# df["verified"] = [t["verified"] for t in temp_dict["reviews"]]
# 
# # triming trailiing text from date column 
# df["review date"] = df["review date"].str[32:]
# 
# # diplaying dataframe data
# df

# ## Review Summary Generation

# #### Develop Data Summary

# In[ ]:


# Developing Summary of Reviews for Each Web
amz_cust_reviews = df["review"]
amz_reviews_str = "".join(each for  each in amz_cust_reviews)
print(f"Total length of all reviews text chain is {len(amz_reviews_str)} characters.")


# In[ ]:


# Storing review data into different formats 
# converting the dataframe to CSV format for checking purpose
if new_data:
    df.to_csv("amz_reviews.csv", mode="w", index=False)                       # storing in CSV format
    file = open('./review_docs/amz_reviews.txt','w', encoding='utf-8')        # storing in text format
    file.writelines(amz_reviews_str)
    file.close()


# In[ ]:


# Setting LLM 
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
print("LLM gets loaded successfully")


# #### Check for Review Content Length

# In[ ]:


# Counting AutoScraper output tokens

def count_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

total_tokens = count_tokens(str(amz_reviews_str), "cl100k_base")
print("Total actual number of input tokens =", total_tokens)


# In[ ]:


if total_tokens <= 3500: 
    print("Reveiew summary will be generated using the small content context method.")
    
    summary_statement = """You are an expeienced copy writer providing a world-class summary of product reviews {reviews} from numerous customers \
                        on a given product from different leading e-commerce platforms. You write summary of all reviews for a target audience \
                        of wide array of product reviewers ranging from a common man to an expeirenced product review professional."""
    summary_prompt = PromptTemplate(input_variables = ["reviews"], template=summary_statement)
    llm_chain = LLMChain(llm=llm, prompt=summary_prompt)
    amz_review_summary_smp = llm_chain.run(amz_reviews_str)
    print("Amazon Review Summary: \n\n", amz_review_summary_smp)


# ##### Define Function for Sentiment Analysis

# In[ ]:


# define a function for sentiment analysis
# https://python.langchain.com/docs/use_cases/tagging/

class Classification(BaseModel):
    Overall_Sentiment: str = Field(..., enum=["Positive", "Neutral", "Negative"])
    Review_Aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the  properties mentioned in the 'Classification' function from the following text.
    Paragraph:
    {input}
    """
)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
    Classification
)

tagging_chain = tagging_prompt | llm


# ### Generating Customer Review Sentiment for smaller inputs

# In[ ]:


if total_tokens <= 3500:
    output_smp = tagging_chain.invoke({"input": amz_review_summary_smp})
    print("Customer Reviews' Sentiment \n\n", output_smp)
    print("\n *** PROGRAM EXECUTION ABORTED HERE ***")
    raise StopExecution


# In[ ]:


# Splitting the doc into sizeable chunks

raw_documents = TextLoader("./review_docs/amz_reviews.txt", encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
split_text = text_splitter.split_documents(raw_documents)
# docs = [Document(page_content=each) for each in split_text]
print("Total number of documents =", len(docs))


# In[ ]:


print(docs[:2])


# ### Apply Map Reduce Method 
# #### (Summarize large Document)

# In[ ]:


# Applying map reduce to summarize large document
# https://python.langchain.com/docs/use_cases/summarization/
print("Map Reduce Process is initiated now") 

map_template = """Based on the following docs {docs}, please provide summary of reviews presented in these documents. 
Review Summary is:"""

map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)


# The ReduceDocumentsChain handles taking the document mapping results and reducing them into a single output. It wraps a generic CombineDocumentsChain (like StuffDocumentsChain) but adds the ability to collapse documents before passing it to the CombineDocumentsChain if their cumulative size exceeds token_max. In this example, we can actually re-use our chain for combining our docs to also collapse our docs.
# 
# So if the cumulative number of tokens in our mapped documents exceeds 4000 tokens, then we’ll recursively pass in the documents in batches of \< 4000 tokens to our StuffDocumentsChain to create batched summaries. And once those batched summaries are cumulatively less than 4000 tokens, we’ll pass them all one last time to the StuffDocumentsChain to create the final summary.

# In[ ]:


# Reduce
reduce_template = """The following is set of summaries: 
{doc_summaries}
Take these document and return your consolidated summary in a professional manner addressing the key points of the customer reviews. 
Review Summary is:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)


# In[ ]:


# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=3500,
)


# Combining our map and reduce chains into one

# In[ ]:


# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)


# #### Generating Map Reduce Summary

# In[ ]:


amz_review_summary_mr = map_reduce_chain.invoke(docs)


# In[ ]:


print("Amazon Review Summary as per Map Reduced Method: \n\n", amz_review_summary_mr['input_documents'][0])


# ### Apply Refine Method 
# #### (Summarize large Document)

# In[ ]:


# Checking the Refine Method for comparison
# https://medium.com/@abonia/summarization-with-langchain-b3d83c030889
print("Document Refine Method is initiated now")

prompt_template = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """

question_prompt = PromptTemplate(
    template=prompt_template, input_variables=["text"]
)

refine_prompt_template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in that covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
              """

refine_prompt = PromptTemplate(
    template=refine_prompt_template, input_variables=["text"])

# Load refine chain
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=question_prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_text",
   output_key="output_text",
)
amz_review_summary_ref = chain.invoke({"input_text": docs}, return_only_outputs=True)


# #### Generating Refine Method Summary

# In[ ]:


print("Amazon Review Summary as per Refine Method: \n\n", amz_review_summary_ref['intermediate_steps'][0])


# ### Customer Sentiment from Review Summaries

# In[ ]:


# Generating customer reviews sentiment based on map reduce method summary
output_mr = tagging_chain.invoke({"input": amz_review_summary_mr['input_documents'][0]})


# In[ ]:


print("Sentiment Output for Map Reduce Summary :", output_mr)


# In[ ]:


# Generating customer reviews sentiment based on refine method summary
output_ref = tagging_chain.invoke({"input": amz_review_summary_ref})


# In[ ]:


print("Sentiment Output for Refined Method Summary :", output_ref)


# ### Generating Comparative Output Summary

# In[ ]:


print("Generating Review Summary and Sentiment Output Dataframe now")

data_output = pd.DataFrame({"Review Summary":[amz_review_summary_mr["input_documents"][0], amz_review_summary_ref["intermediate_steps"][0]], 
                            "Sentiments":[output_mr, output_ref]},
                           index=(["Map Reduce Method", "Refine Method"]))

pd.set_option("display.colheader_justify","center")
pd.set_option('display.max_colwidth', None)


# ### Exporting Summary

# In[ ]:


data_output.to_pickle("data_output")
data_output.to_csv("data output.csv", mode="w", index=False)


# ### Displaying Summary

# In[ ]:


data_output = data_output.style.set_properties(**{'text-align': 'left'})
data_output.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

print("Final Data Output generation has been complete now")
data_output

