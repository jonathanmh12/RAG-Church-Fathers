# RAG-Church-Fathers
This repo is meant to display a retrieval augmented generating algorithm. It involves a locally stored vector-store and will report what a particular church father says upon request. 

# Explanation of Current State
At present only documents from Ignatius of Antioch are available due to cost reasons. However there are plans to obtain more. I've found that in process automation algorithms, the best practice is to define functions that are called from the main function (similar to a java type code). The process is as follows:
Step 1: Create a vector database, hosted locally, of all of Igantius' letters. We do this using the FAISS library (it's pretty intuitive). 
Step 2: Create a process for identifying similar documents. We start by creating an embedding of the query itself to compare and then use the index search function from the FAISS library to identify k documents.
Step 3: Call an LLM from a local Ollama server (in this case llama3) and then construct a prompt that instructs the llm to only use the letters as context. Generate the response.
I put the last two together in a main function at the end in an effort to consolidate the amount of user input required to get a response. As of now, one only has to update the query and then run the final cell.

# Future Goal
My hope is to create a website where a user can ask what a church father (anyone who had significant impact on the church between the time period 30 AD - 450/500 AD) thought about any particular topic and have the answer be reliable because of the database of materials I have prepared. I have a mockup of python code but I imagine it would be easier to build this in Java so thats on my list to learn.
