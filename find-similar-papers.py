import PyPDF2
import io
import json
import sys

# Find all of the JATS papers in the current directory
def find_papers():
    # Import the os library
    import os

    # Create an empty list to store the papers
    papers = []

    # Recursively loop through all of the directories in
    # the current directory to find all of the JATS files
    for root, dirs, files in os.walk("."):
        # Loop through all of the files
        for file in files:
            # Check if the file is a PDF file and includes 'joss' in the name somewhere            
            if file.endswith(".pdf") and "joss" in file.lower():
                # Create the path to the paper
                path = os.path.join(root, file)

                # Add the path to the list of papers
                papers.append(path)

    # Return the list of papers
    return papers

# Import the os library
import os

# Require the OpenAI library
import openai as ai

# Require the embedding utilities
from openai.embeddings_utils import get_embedding, cosine_similarity

# Require the Pandas library
import pandas as pd

# Set the API key for OpenAI
ai.api_key = os.environ['openai_api_key']

# List the available models and print them
models = ai.Model.list()

# Create embedding for the paper
def create_embedding(paper):
    
    try:
        # Create the prompt by reading the contents of the paper
        pdf = open(paper, "rb").read()
        reader = PyPDF2.PdfReader(io.BytesIO(pdf)) # Wrap the bytes object in a BytesIO object
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text() # Use the pages attribute instead of getPage

        # Limit the prompt size to 8191 tokens
        text = text[:8191]

        # Create the embedding using the text-embedding-ada-002 engine
        embedding = ai.Embedding.create(engine = "text-embedding-ada-002", input=text)

        # Return the embedding
        return embedding
    
    except PyPDF2.utils.PdfReadError as e:
        print(f"Error reading PDF file {paper}: {e}")
        return None
 
# This function converts 10.21105.joss.00899.pdf to https://joss.theoj.org/10.21105/joss.00899.json
def convert_pdf_string(incoming):
    # Split the incoming string on the period character
    parts = incoming.split(".")

    # Return the joined string
    return f"https://joss.theoj.org/papers/{parts[0]}.{parts[1]}/{parts[2]}.{parts[3]}.json"

# Call the JOSS API and parse the JSON response
def call_joss_api(paper):
    # Import the requests library
    import requests

    # Create the URL for the JOSS API
    url = convert_pdf_string(paper)

    # Call the JOSS API
    response = requests.get(url)

    # Check if the response was successful
    if response.status_code == 200:
        # Return the JSON response
        return response.json()
    else:
        # Return None
        return None
    

# Extract reviewer array from the JSON response and join fields as string
def extract_reviewers(response):
    # Check if the response is None
    if response is None:
        # Return None
        return None

    # Create an empty list to store the reviewers
    reviewers = []

    # Loop through the reviewers in the response
    for reviewer in response['reviewers']:
        # Add the reviewer to the list of reviewers
        reviewers.append(reviewer)

    # Return the reviewers as a comma-separated string
    return ", ".join(reviewers)

# Extract the review url from the JSON response
def extract_review_url(response):
    # Check if the response is None
    if response is None:
        # Return None
        return None

    # Return the review url
    return response['paper_review']


# Extract the paper title from the JSON response
def extract_title(response):
    # Check if the response is None
    if response is None:
        # Return None
        return None

    # Return the paper title
    return response['title']

# Takes a string like ./joss.04738/10.21105.joss.04738.pdf
# first returns 10.21105.joss.04738.pdf
# then calls out to the JOSS API and returns the title
def title_from_paper(paper):
    # Import the os library
    import os

    # Split the paper on the period character
    filename = os.path.basename(paper)
    response = call_joss_api(filename)

    return extract_title(response)

# Print out a summary of the paper, together with the URL and reviewers.
def print_summary(paper):
    # Call the JOSS API
    response = call_joss_api(paper)

    # Extract the reviewers from the response
    reviewers = extract_reviewers(response)

    # Extract the review url from the response
    review_url = extract_review_url(response)

    # Extract the title from the response
    title = extract_title(response)

    # Print out the summary
    return f"[{title}]({review_url})\nReviewers: {reviewers}"

    
# Run the script
if __name__ == "__main__":

# Read the embeddings from the embeddings.csv file
    embeddings = pd.read_csv('embeddings.csv.gz', compression='gzip')

    # Set the path to the paper_path environment variable from Actions
    incoming_paper = os.environ['pdf_path']

    # Create the embedding for the paper
    incoming_embedding = create_embedding(incoming_paper)

    # Create a Pandas dataframe to store the paper embeddings
    only_embeddings = pd.DataFrame(columns = ["paper", "embedding"])

    # Loop through the embeddings Pandas dataframe
    # creating a new Pandas dataframe with just the paper and embedding
    for index, row in embeddings.iterrows():
        dict = json.loads(row.embedding)
        embedding = dict['data'][0]['embedding']
        only_embeddings = pd.concat([only_embeddings, pd.DataFrame([[row.paper, embedding]], columns = ["paper", "embedding"])])

    # Find the top 5 most similar papers in the only_embeddings DataFrame using a cosine similarity

    # Create a breakpoint here to inspect the only_embeddings DataFrame
    # breakpoint()

    target_embedding = incoming_embedding.data[0]['embedding']
    only_embeddings['similarity'] = only_embeddings['embedding'].apply(lambda x: cosine_similarity(x, target_embedding))

    # Sort the only_embeddings DataFrame by similarity in descending order
    only_embeddings = only_embeddings.sort_values(by=['similarity'], ascending=False)

    # Loop through the first 5 rows of the only_embeddings DataFrame
    # skipping the first row as this will be the paper itself
    # printing out the paper and similarity
    
    print("The top 5 most similar papers are:")
    for index, row in only_embeddings[1:6].iterrows():
        print(f"{print_summary(row.paper)}")
        print(f"Similarity score: {row.similarity}")
        print("\n")

    # Print out logs in GitHub Actions 
    sys.stdout.flush()
               

