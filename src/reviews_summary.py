import os
import tiktoken
import pandas as pd
from typing import Any
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from src.cust_reviews.amazon_scraper import AmazonScraper
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

# packages for print output notifications
from io import StringIO
import sys

tmp = sys.stdout
my_result = StringIO()
sys.stdout = my_result
load_dotenv()

# importing api keys and initiate llm
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

# Counting AutoScraper output tokens
def count_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# generate review summary for smaller revieww
def small_reviews_summary(cust_reviews: str) -> str:
    summary_statement = """You are an expeienced copy writer providing a world-class summary of product reviews {cust_reviews} from numerous customers \
                        on a given product from different leading e-commerce platforms. You write summary of all reviews for a target audience \
                        of wide array of product reviewers ranging from a common man to an expeirenced product review professional."""
    summary_prompt = PromptTemplate(input_variables = ["cust_reviews"], template=summary_statement)
    llm_chain = LLMChain(llm=llm, prompt=summary_prompt)
    review_summary = llm_chain.invoke(cust_reviews)
    return review_summary

# split large reviews
def document_split(cust_reviews: str, chunk_size: int, chunk_overlap: int) -> Any:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.split_documents(cust_reviews)
    return split_docs

# Applying map reduce to summarize large document
def map_reduce_summary(split_docs: Any) -> str: 
    map_template = """Based on the following docs {docs}, please provide summary of reviews presented in these documents. 
    Review Summary is:"""

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries: 
    {doc_summaries}
    Take these document and return your consolidated summary in a professional manner addressing the key points of the customer reviews. 
    Review Summary is:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

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
    
    # generating review summary for map reduce method
    cust_review_summary_mr = map_reduce_chain.invoke(split_docs)

    return cust_review_summary_mr

# Applying refine method to summarize large document
def refine_method_summary(split_docs) -> str:
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
    
    # generating review summary using refine method
    cust_review_summary_refine = chain.invoke({"input_text": split_docs}, return_only_outputs=True)
    return cust_review_summary_refine


def get_review_summary(prod_query: str, cust_count: int) -> tuple[int, Any, str, str, str]:
    
    # getting amazon customer reviews using amazon scrapper
    scraper = AmazonScraper()
    cust_reviews = scraper.get_closest_product_reviews(str(prod_query), num_reviews = cust_count, debug=False)

    # generating df from the scrap data
    df = pd.DataFrame.from_dict(cust_reviews)
    df.sort_values(by=["rating"], ascending=False, inplace=True)
    df.reset_index(inplace=True)

    # extracting customer reviews from the df in string format 
    amz_cust_reviews = df["review"]
    amz_reviews_str = "".join(each for each in amz_cust_reviews)

    # Checking review length
    total_tokens = count_tokens(str(amz_reviews_str), "cl100k_base")

    if total_tokens <= 3500:
        cust_review_summary = small_reviews_summary(amz_reviews_str)
        cust_review_summary_map = "N.A."
        cust_review_summary_refine = "N.A."
    else:
        split_docs = document_split(amz_reviews_str, 1000, 50)
        cust_review_summary_map = map_reduce_summary(split_docs)
        cust_review_summary_refine = refine_method_summary(split_docs)
        cust_review_summary = "N.A."

    return total_tokens, df, cust_review_summary, cust_review_summary_map, cust_review_summary_refine












