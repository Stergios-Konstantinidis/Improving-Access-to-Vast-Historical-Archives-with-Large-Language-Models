open_router_api_key = "YOUR_OPEN_ROUTER_API_KEY"
from openai import OpenAI
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import sql_setup


def llm_ocr_correction(text, api_key=open_router_api_key):
    """
    Use Ollama's LLM model to correct OCR text.

    :param text: The OCR text to be corrected.
    :return: Corrected text.
    """
    try:
        print("Correcting text using LLM...", text[:50])  # Print the first 50 characters of the text for debugging
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            )
        completion = client.chat.completions.create(
            extra_body={},
            model="openai/gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": "We have scanned an ensemble of documents and we possess OCR segments of the text. These segments often contain OCR errors, correct those errors as well as possible. Only return the text corrected, nothing else, sometimes the segment can be numbers and signs, return the same signs/numbers when applicable, and remove markdown elements. Here is a the segments to correct: {}".format(text)
                }
            ]
        )
    
        return completion.choices[0].message.content


    except Exception as e:
        print(f"Error in LLM correction: {e}")
        return None
    
def get_next_article():
    """
    Get the next article from the database that has not been corrected by the LLM.
    """
    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("""select * from articles where llm_method = 'none' 
                   and (ocr_text not in (SELECT ocr_text FROM articles WHERE llm_method = 'GPT-4o') 
                   or ocr_method not in (SELECT ocr_method FROM articles WHERE llm_method = 'GPT-4o'))
                   order by id asc LIMIT 100""")
    articles = cursor.fetchall()

    cursor.close()
    conn.close()
    return articles

def create_article_entry(
    newspaper_issue_id:int,
    page_number:int,
    title:str,
    content:str,
    author:str,
    date:str,
    directory:str,
    ocr_text:str,
    llm_text:str,
    ocr_method:str,
    llm_method:str,
    manual_correction_text:str,
    font:str,
    coordinates:list,
    segmentation_method:str):

    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
   

    cursor.execute("""
        INSERT INTO articles (newspaper_issue_id, page_number, title, content, author, date, directory, ocr_text, llm_text, ocr_method, llm_method, manual_correction_text, font, coordinates, segmentation_method)
        VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, ( newspaper_issue_id, page_number, title, content, author, date, directory, ocr_text, llm_text, ocr_method, llm_method, manual_correction_text, font, str(coordinates), segmentation_method))
    return cursor, conn

if __name__ == "__main__":
    # Example usage
    
    # get the next article from the database, not corrected by qwen3
    for i in range(60000):  # Limit to 600 iterations for testing
        try:
            articles = get_next_article()
            print(f"Found {len(articles)} articles to process.")
            for article in articles:
                print(f"Processing article ID: {article[0]}")
                text = article[8]  # Assuming the OCR text is in the 8th column

                corrected_text = llm_ocr_correction(text)
                if corrected_text:
                    print("Corrected Text:")
                    print(article[12])
                    # Create a new article entry with the corrected text
                    cursor, conn = create_article_entry(
                        newspaper_issue_id=article[1],
                        page_number=article[2],
                        title=article[3],
                        content=article[4],
                        author=article[5],
                        date=article[6],
                        directory=article[7],
                        ocr_text=text,
                        llm_text=corrected_text,
                        ocr_method=article[10],
                        llm_method='GPT-4o',
                        manual_correction_text=article[12],
                        font=article[13],
                        coordinates=article[14],
                        segmentation_method=article[15]
                    )
                    #cursor.execute("update articles_for_rag set corrected_with_gpt = 1 where id = %s", (article[0],))
                    cursor.close()
                    conn.close()
            else:
                print("Failed to correct the text.")
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

