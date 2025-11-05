import numpy as np
import vptree
import os
import sys
import inspect
import pandas as pd
from numpy.linalg import norm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import sql_setup as models
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb.utils.embedding_functions as embedding_functions
import chromadb.utils.embedding_functions as embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="#YOUROPENAIKEYHERE#",
                model_name="text-embedding-3-small"
            )
google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="#YOURVERTEXAIKEYHERE#")

import chromadb
collection_name = "YOURCHROMACOLLECTIONNAMEHERE" # collection created during step 4
chroma_client = chromadb.PersistentClient(path=f"./chroma/{collection_name}")
collection = chroma_client.get_collection(name=collection_name)

values = collection.get(include=["embeddings", "documents", "metadatas"])

def get_document_from_embedding(embedding):

    index = values["embeddings"].index(embedding[1].tolist())

    return values["ids"][index], values["documents"][index], values["metadatas"][index]

# Define distance function.
def euclidean(p1, p2):
  return np.sqrt(np.sum(np.power(p2 - p1, 2)))
def cosine(p1, p2):
    return 1 - np.dot(p1, p2) / (norm(p1) * norm(p2))

print(euclidean(np.array([1, 2, 3]), np.array([4, 5, 6])))
print(cosine(np.array([1, 2, 3]), np.array([4, 5, 6])))
# Get embeddings from ChromaDB collection.
points = np.array(values["embeddings"])


# Build tree in O(n log n) time complexity.
#print(len(points), "points in collection.")
treel = vptree.VPTree(points, euclidean)


conn = models.sqlConnection()
cursor = conn.cursor()
cursor.execute("SELECT question FROM ragasGT WHERE answer is null ORDER BY RAND()")
questions = cursor.fetchall()

ef = DefaultEmbeddingFunction()
df = pd.DataFrame(columns=['tree_size', 'retrieved_documents_average', 'retrieved_documents_std', 'retrieved_documents', 'count', 'retrieved_docs_%', 'retrieved_docs_%_std'])
df = df.sample(frac=1).reset_index(drop=True)
for i in [10000, 20000, 30000, 40000, 50000, 60000]:
    print(f"Building tree with {i} points.")
    print(points[0:i])
    
    
    for j in [1,5,10]:
        retrieved_documents = []
        for x in range(20): # repeat 20 times
            #shuffle the points
            np.random.shuffle(points)
            tree = vptree.VPTree(points[0:i], euclidean)

            for question in questions:
                question_embedding = ef(str(question[0]))


                retrieved_documents.append(tree.get_n_nearest_neighbors(question_embedding, j)[-1])
                print("retrieved_documents:", retrieved_documents[-1])

        print("avg retrieved_documents:", sum(retrieved_documents) / len(retrieved_documents))
        print("std retrieved_documents:", np.std(retrieved_documents))
        df.loc[len(df)] = ({
            'tree_size': i,
            'retrieved_documents_average': sum(retrieved_documents) / len(retrieved_documents),
            'retrieved_documents_std': np.std(retrieved_documents),
            'retrieved_documents': j,
            'count': len(retrieved_documents),
            'retrieved_docs_%': (sum(retrieved_documents) / (len(retrieved_documents) * j)) * 100,
            'retrieved_docs_%_std': np.std(retrieved_documents) / (len(retrieved_documents)) * 100
        })
        df.to_csv("vptree_results_euclidean_shuffled_20_iterations.csv", index=False)