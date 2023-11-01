import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pinecone

pinecone.init(api_key='4d743ef1-9f63-45dd-9555-3486fb1d1d60', environment='gcp-starter')
if __name__ =='__main__':
    print("Hello VectorStore!")

    loader=TextLoader("mediumblogs\mediumblog1.txt", encoding='UTF-8')
    document = loader.load()

    text_spilitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts= text_spilitter.split_documents(document)

    embeddings= OpenAIEmbeddings(openai_api_key="") 
    docsearch= Pinecone.from_documents(texts,embeddings,index_name="medium-blog-embeddings-index")

    qa= RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=""),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )
    query="What is a vector DB? Give me 15 words answer for beginners" 
    result=qa({"query":query})
    result1=qa.run("What is a vector DB? Give me 15 words answer for beginners")
    
    print(len(texts))
    print(result['result'])
    print(result1['result'])
