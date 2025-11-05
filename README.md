## Project Description
We use LLMs for Cultural Digital Archives. We use the LLMs to correct OCR errors (remove noise) and for RAG purposes. Our results show that LLMs and improve OCR (WER & CER metrics), and read to improved information retrieval compared to the non-noise-cleaned versions of the digital archives.

<img width="528" height="334" alt="image" src="https://github.com/user-attachments/assets/540be5c1-7836-4226-99d5-776d690ce222" />

We also use metric trees to speedup the kNN search of RAG for faster retrieval of large archives. We see a drastic reduction in the number of embeddings retrieved for the RAG search, without any false dismissals (eg compared to using approximate nearest neighbors).

<img width="718" height="244" alt="image" src="https://github.com/user-attachments/assets/1c7fd1d4-49b3-452f-ba71-50bfc6eb6cda" />


  
## Project Structure

This project is organized into a series of scripts designed to process and analyze historical document archives using large language models. Follow the steps below for a smooth workflow:

### **1-LayoutParser.py**
   - **Purpose:** Parses the layout and structure of historical documents, preparing them for further processing.

### **2-Unsupervised_ocr.py**
   - **Purpose:** Applies unsupervised Optical Character Recognition (OCR) to extract text from the parsed document layouts.

### **3-LLMCorrections.py**
   - **Purpose:** Utilizes large language models to clean, correct, and post-process the OCR results, improving text accuracy.

### **4-ChromaIndex.py**
   - **Purpose:** Indexes the processed documents for efficient retrieval using ChromaDB, designed to work with all-mini-lm6, open-ai and vertex ai embeddings

### **5-RAG.py**
   - **Purpose:** Implements Retrieval-Augmented Generation for contextual querying.

### **6-VP tree.py**
   - **Purpose:** Vantage-point index for fast kNN search across embeddings.

> **Note:**  
> Run each script sequentially for optimal results.

---

## Queries
We evaluate the historical archives using [131 questions](data/ragas%20evaluation%20questions.csv), which are of the following categories:

| Category          | Count |
|:------------------|------:|
| Sociocultural     | 59 |
| Numeric           | 17 |
| Historic          | 17 |
| Politics          | 13 |
| Address           | 7 |
| Reliability check | 5 |


## Configuration Instructions

Before running the scripts, ensure the following setup steps are completed:

1. **SQL Environment Setup**
   - Configure your SQL environment.
   - Execute the cells in `create_sql_tables.ipynb` (run once).
   - Manually populate the `newspapers` and `newspaper_issues` tables.  
     - **Tip:** The `directory` field is crucial and is case-sensitive.
     
     <img src="data/github ressources/UML.png" />

2. **API Keys**
   - In `3-LLMCorrections.py`, `4-ChromaIndex.py` and `6-VpTree`, insert your API keys where indicated to enable access to required services.

---

By following these steps and updating the necessary parameters, you will be able to process and analyze historical archives efficiently using this pipeline.

If you encounter any issues, please contact me: stergios@unil.ch 

Library data in this repository are not complete due to copyright.


## Demo
A demo application is provided in the `demo` folder. It uses Streamlit to showcase the LayoutParser functionality and Tesseract OCR integration.
To run the demo, navigate to the `demo` folder and execute:
```
bash streamlit run demo.py
```