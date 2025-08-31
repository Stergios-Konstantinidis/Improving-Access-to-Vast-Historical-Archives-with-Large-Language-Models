from openai import OpenAI
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import sql_setup
import chromadb.utils.embedding_functions as embedding_functions
import chromadb
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="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",
                model_name="text-embedding-3-small"
            )
google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="AIzaSyAB-aVF3-4qNk_tnogbA3bNTxq1-mDXNe4")



api_key = "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"
from openai import OpenAI

import os



def LLM_prompt_with_rag(model="openai/gpt-4o-2024-11-20", documents=[], context=[], prompt=[], api_key=api_key):
    """
    Use Ollama's LLM model to correct OCR text.

    :param text: The OCR text to be corrected.
    :return: Corrected text.
    """
    if len(context) != len(documents):

        try:
            client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            )
            completion = client.chat.completions.create(
                extra_body={},
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in historical documents and you have access to the following documents: {}".format(documents)
                    },
                    {
                        "role": "user",
                        "content": "You are an expert in historical documents, answer the following question using the provided documents: {}".format(prompt),
                    }
                ],
                temperature=0
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM correction: {e}")
        return None
    else:
        try:
            client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            )
            rag_content = []
            for i, doc in enumerate(documents):
                rag_content.append(context[i] + "\n" + doc)
            completion = client.chat.completions.create(
                extra_body={},
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in historical documents and you have access to the following documents: {}".format(context,documents)
                    },
                    {
                        "role": "user",
                        "content": "You are an expert in historical documents, answer the following question using the provided documents: {}. Answer in one sentence maximum.".format(prompt),
                    }
                ],
                temperature=0
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM correction with context: {e}")
        return None

def get_non_calculated_ragasGT(not_used_embedding=None, not_used_llm=None, ocr_method=None, llm_correction=None, k_value=5):
    """
    Get the RAG (Retrieval-Augmented Generation) results for a specific embedding and LLM model.
    """
    if not_used_embedding is None or not_used_llm is None:
        print("Embedding and LLM model must be specified.")
        return None

    # Simulate retrieval of RAG results
    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    if llm_correction is None:
        llm_correction = 'None'
    query = "SELECT question, groundtruth  FROM ragasGT where answer is null and question not in (SELECT question FROM ragasGT where embedding = '" + str(not_used_embedding) + '_k' + str(k_value) + "' and LLM = '" + not_used_llm + "' and ocr_method = '" + ocr_method + "' and llm_correction = '" + llm_correction + "')"
    print(query)
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


llm_correction = None  # Default value for LLM correction

if __name__ == "__main__":
    if llm_correction is None:
        collections = ['apple_ocr_openai', 'apple_ocr_google']
    elif llm_correction == "gpt4o":                                 
        collections = ['apple_ocr_gpt_correction_openai_embeddings','apple_ocr_gpt_correction_google_embeddings']#, 'apple_ocr_gpt_correction_all-MiniLM-L6-v2_embeddings']
    elif llm_correction == "gemini":
        collections = ['apple_ocr_gemini_correction_openai_embeddings', 'apple_ocr_gemini_correction_google_embeddings', 'apple_ocr_gemini_correction_all-MiniLM-L6-v2_embeddings']
    elif llm_correction == "qwen":
        collections = ['apple_ocr_qwen3_correction_openai_embeddings', 'apple_ocr_qwen3_correction_google_embeddings', 'apple_ocr_qwen3_correction_all-MiniLM-L6-v2_embeddings']

    llms = ["openai/gpt-4o-2024-11-20", "google/gemini-2.5-pro"]#, "qwen/qwen3-235b-a22b"]
    for collection in collections:
        embedding = collection.split('_')[-1] if collection.split('_')[-1] != "embeddings" else collection.split('_')[-2]  # Extract the embedding type from the collection name
        ocr_method = collection.split('_')[0]  # Extract the OCR method from the collection name

        for llm in llms:
            chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection}")
            chroma_index = chroma_client.get_collection(name=collection)
            print(chroma_index.count(), chroma_index.id)
            for k_value in [1,10]:
                ef = openai_ef if 'openai' in collection else embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="AIzaSyAB-aVF3-4qNk_tnogbA3bNTxq1-mDXNe4")
                print("openai ef" if 'openai' in collection else "google ef")
                questions = get_non_calculated_ragasGT(not_used_embedding=embedding, not_used_llm=llm, ocr_method=ocr_method, llm_correction=llm_correction, k_value=k_value)
                if questions is None:

                    continue
                else:
                    for question in questions:
                        question_, ground_truth = question[0], question[1]
                        print(question_)
                        results = chroma_index.query(query_embeddings=ef([question_]), n_results=k_value, include=["documents", "embeddings"])

                        if results:
                            #print(results)
                            documents = results['documents']
                            #print(f"Retrieved documents: {documents}")

                            print(chroma_index.count(), chroma_index.id)

                            # Call the LLM with the retrieved documents
                            answer = LLM_prompt_with_rag(model=llm, documents=documents, prompt=question_)

                            # Store the answer in the database
                            conn = sql_setup.sqlConnection()
                            cursor = conn.cursor()
                            cursor.execute("Insert into ragasGT (answer, question, groundtruth, context, embedding, LLM, ocr_method, llm_correction, k_value) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", (answer, question_, ground_truth, str(documents), embedding + '_k' + str(k_value), llm, ocr_method, llm_correction if llm_correction != None else 'None', k_value))
                            conn.commit()
                            cursor.close()
                            conn.close()

                            print(f"Answer for '{question}': {answer}")
                        else:
                            print(f"No relevant documents found for question: {question}")

