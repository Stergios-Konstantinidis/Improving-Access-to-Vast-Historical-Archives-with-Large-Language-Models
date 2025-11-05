import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import sql_setup

import chromadb.utils.embedding_functions as embedding_functions
import chromadb

import chromadb.utils.embedding_functions as embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="YOUR_OPENAI_APIKEY",
                model_name="text-embedding-3-small"
            )
google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="YOUR_VERTEXAI_APIKEY")



def add_to_collection_apple_ocr_openai_embeddings():
    """
    Add articles to the 'apple_ocr_openai' collection in ChromaDB.
    """
    collection_name = "apple_ocr_openai"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    #get the articles in the collection
    ids_in_collection =collection.get(ids=[str(463)])



    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.ocr_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' and llm_method = 'none'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")
    collection.add(
        embeddings=openai_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_apple_ocr_google_embeddings():
    """
    Add articles to the 'apple_ocr_google' collection in ChromaDB.
    """
    collection_name = "apple_ocr_google"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.ocr_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'none'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")
    collection.add(
        embeddings=google_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_apple_ocr_allminiLML6v2_embeddings():
    """
    Add articles to the 'apple_ocr_all-MiniLM-L6-v2_embeddings' collection in ChromaDB.
    """
    collection_name = "apple_ocr_all-MiniLM-L6-v2_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.ocr_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'none'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")
    collection.add(
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )
#################################################################################################################################

def add_to_collection_all_methods_openai_embeddings():
    """
    Add articles to the 'all_methods_openai' collection in ChromaDB.
    """
    collection_name = "all_methods_openai"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE llm_method = 'none'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")
    collection.add(
        embeddings=openai_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_all_methods_google_embeddings():
    """
    Add articles to the 'all_methods_google' collection in ChromaDB.
    """
    collection_name = "all_methods_google"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE llm_method = 'none'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")
    collection.add(
        embeddings=google_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_all_methods_allminiLML6v2_embeddings():
    """
    Add articles to the 'all_methods_all-MiniLM-L6-v2_embeddings' collection in ChromaDB.
    """
    collection_name = "all_methods_all-MiniLM-L6-v2_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE llm_method = 'none'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")
    collection.add(
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

#################################################################################################################################

def add_to_collection_apple_ocr_gpt_correction_openai_embeddings():
    """
    Add articles to the 'apple_ocr_gpt_correction_openai' collection in ChromaDB.
    """
    collection_name = "apple_ocr_gpt_correction_openai_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'GPT4o'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")
    collection.add(
        embeddings=openai_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_apple_ocr_gpt_correction_google_embeddings():
    """
    Add articles to the 'apple_ocr_gpt_correction_google' collection in ChromaDB.
    """
    collection_name = "apple_ocr_gpt_correction_google_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'GPT4o'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(len(articles))
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        embeddings=google_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_apple_ocr_gpt_correction_allminiLML6v2_embeddings():
    """
    Add articles to the 'apple_ocr_gpt_correction_all-MiniLM-L6-v2_embeddings' collection in ChromaDB.
    """
    collection_name = "apple_ocr_gpt_correction_all-MiniLM-L6-v2_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'GPT4o'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "GPT4o", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

#################################################################################################################################

def add_to_collection_apple_ocr_gemini_correction_openai_embeddings():
    """
    Add articles to the 'apple_ocr_gemini_correction_openai' collection in ChromaDB.
    """
    collection_name = "apple_ocr_gemini_correction_openai_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'Gemini-2.5-Pro'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        embeddings=openai_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "Gemini-2.5-Pro", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_apple_ocr_gemini_correction_google_embeddings():
    """
    Add articles to the 'apple_ocr_gemini_correction_google' collection in ChromaDB.
    """
    collection_name = "apple_ocr_gemini_correction_google_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'Gemini-2.5-Pro'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        embeddings=google_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "Gemini-2.5-Pro", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_apple_ocr_gemini_correction_allminiLML6v2_embeddings():
    """
    Add articles to the 'apple_ocr_gemini_correction_all-MiniLM-L6-v2_embeddings' collection in ChromaDB.
    """
    collection_name = "apple_ocr_gemini_correction_all-MiniLM-L6-v2_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'Gemini-2.5-Pro'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        embeddings=google_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "Gemini-2.5-Pro", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

#################################################################################################################################

def add_to_collection_apple_ocr_qwen3_correction_openai_embeddings():
    """
    Add articles to the 'apple_ocr_qwen3_correction_openai' collection in ChromaDB.
    """
    collection_name = "apple_ocr_qwen3_correction_openai_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'qwen3_235b'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        embeddings=openai_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "qwen3_235b", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )


def add_to_collection_apple_ocr_qwen3_correction_google_embeddings():
    """
    Add articles to the 'apple_ocr_qwen3_correction_google' collection in ChromaDB.
    """
    collection_name = "apple_ocr_qwen3_correction_google_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'qwen3_235b'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        embeddings=google_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "qwen3_235b", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_apple_ocr_qwen3_correction_allminiLML6v2_embeddings():
    """
    Add articles to the 'apple_ocr_qwen3_correction_all-MiniLM-L6-v2_embeddings' collection in ChromaDB.
    """
    collection_name = "apple_ocr_qwen3_correction_all-MiniLM-L6-v2_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'qwen3_235b'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "qwen3_235b", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

#################################################################################################################################

def add_to_collection_apple_ocr_geminiflash_correction_openai_embeddings():
    """
    Add articles to the 'apple_ocr_geminiflash_correction_openai' collection in ChromaDB.
    """
    collection_name = "apple_ocr_geminiflash_correction_openai_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'geminiflash'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        embeddings=openai_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "geminiflash", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )


def add_to_collection_apple_ocr_geminiflash_correction_google_embeddings():
    """
    Add articles to the 'apple_ocr_geminiflash_correction_google' collection in ChromaDB.
    """
    collection_name = "apple_ocr_geminiflash_correction_google_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'geminiflash'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        embeddings=google_ef([article[1] for article in articles[0:1000]]),
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "geminiflash", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

def add_to_collection_apple_ocr_geminiflash_correction_allminiLML6v2_embeddings():
    """
    Add articles to the 'apple_ocr_geminiflash_correction_all-MiniLM-L6-v2_embeddings' collection in ChromaDB.
    """
    collection_name = "apple_ocr_geminiflash_correction_all-MiniLM-L6-v2_embeddings"
    chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT afr.id, afr.llm_text, npi.date, np.title FROM articles afr join newspaper_issues npi on npi.id = afr.newspaper_issue_id join newspapers np on np.id = npi.newspaper_id WHERE ocr_method = 'apple_vision_ocr' AND llm_method = 'geminiflash'")
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles for '{collection_name}' collection.")
    articles = list(filter(lambda x: collection.get(ids=[str(x[0])])["ids"] == [], articles))  # Filter out existing articles
    print(f"Adding {len(articles[0:1000])} articles to '{collection_name}' collection.")

    collection.add(
        ids=[str(article[0]) for article in articles[0:1000]],
        documents=[article[1] for article in articles[0:1000]],
        metadatas=[{"ocr_method": "apple_vision_ocr", "llm_method": "geminiflash", "date": article[2], "title": article[3]} for article in articles[0:1000]],
    )

#################################################################################################################################

for i in range(59):
    print(f"Iteration {i+1}")
    #try:
    #    add_to_collection_apple_ocr_openai_embeddings()
    #    
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_openai_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_google_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_google_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_allminiLML6v2_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_allminiLML6v2_embeddings: {e}")
##
    #try:
    #    add_to_collection_apple_ocr_gpt_correction_openai_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_gpt_correction_openai_embeddings: {e}")
    try:
        add_to_collection_apple_ocr_gpt_correction_google_embeddings()
    except Exception as e:
        print(f"Error adding to apple_ocr_gpt_correction_google_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_gpt_correction_allminiLML6v2_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_gpt_correction_allminiLML6v2_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_gemini_correction_openai_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_gemini_correction_openai_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_gemini_correction_google_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_gemini_correction_google_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_gemini_correction_allminiLML6v2_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_gemini_correction_allminiLML6v2_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_qwen3_correction_openai_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_qwen3_correction_openai_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_qwen3_correction_google_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_qwen3_correction_google_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_qwen3_correction_allminiLML6v2_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_qwen3_correction_allminiLML6v2_embeddings: {e}")
    #
    #try:
    #    add_to_collection_apple_ocr_geminiflash_correction_openai_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_geminiflash_correction_openai_embeddings: {e}")
#
    #try:
    #    add_to_collection_apple_ocr_geminiflash_correction_google_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_geminiflash_correction_google_embeddings: {e}")
    #try:
    #    add_to_collection_apple_ocr_geminiflash_correction_allminiLML6v2_embeddings()
    #except Exception as e:
    #    print(f"Error adding to apple_ocr_geminiflash_correction_allminiLML6v2_embeddings: {e}")