  
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
   - **Purpose:** (Implements Retrieval-Augmented Generation for advanced querying and analysis.)

### **6-VP tree.py**
   - **Purpose:** (Vantage-point index for fast search)

> **Note:**  
> Run each script sequentially for optimal results.

---

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
