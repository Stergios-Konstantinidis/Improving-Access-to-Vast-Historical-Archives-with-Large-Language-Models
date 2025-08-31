import os
import sys
import pdf2image
import inspect
from ocr_tools import pytesseract_ocr, easyocr_ocr, apple_ocr_fun, docling_ocr, docling_default_pipeline, abby_fine_reader
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import sql_setup
import faulthandler
faulthandler.enable()
import PIL
from PIL import Image

def get_next_article():
    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM Layout_Parser_for_eval 
        WHERE processed = 0 
        ORDER BY id asc 
        LIMIT 1
    """)
    article = cursor.fetchone()
    cursor.close()
    conn.close()
    if article:
        return article
    else:
        return None
    
    

def mark_article_as_processed(article_id):
    conn = sql_setup.sqlConnection()
    print(article_id)
    cursor = conn.cursor()
    cursor.execute(f"UPDATE Layout_Parser_for_eval SET processed = 1 WHERE id = {article_id}")
    conn.commit()
    cursor.close()
    conn.close()

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

    font:str,
    coordinates:list,
    segmentation_method:str):
    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO articles (newspaper_issue_id, page_number, title, content, author, date, directory, ocr_text, llm_text, ocr_method, llm_method, font, coordinates, segmentation_method)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (newspaper_issue_id, page_number, title, content, author, date, directory, ocr_text, llm_text, ocr_method, llm_method, font, str(coordinates), segmentation_method))
    conn.commit()
    cursor.close()
    conn.close()

def submit_groundtruth(pytesseract_val, easyocr_val, apple_text, docling_text, docling_default_pipeline, abbyy_ocr_val):
        # save the groundtruth text to the database
        conn = sql_setup.sqlConnection()
        cursor = conn.cursor()

        # get the last article

        parser = get_next_article()

        cursor.execute("SELECT * FROM newspaper_issues WHERE id = %s", (parser[2],))
        newspaper_issue = cursor.fetchone()
        print('creating article entries')

        if pytesseract_val is not None and pytesseract_val != "" and pytesseract_val != "None" and pytesseract_val != " ":
            create_article_entry(
            newspaper_issue_id=1288,
            page_number=parser[3],
            title='',
            content="",
            author='',
            date=None,
            directory=parser[4],
            ocr_text=pytesseract_val,
            llm_text='',
            ocr_method="tesseract",
            llm_method='none',
            font='',
            coordinates=parser[1],
            segmentation_method='Layout_Parser'
            )



        if easyocr_val is not None and easyocr_val != "" and easyocr_val != "None" and easyocr_val != " ":
            create_article_entry(
            newspaper_issue_id=1288,
            page_number=parser[3],
            title='',
            content="",
            author='',
            date=None,
            directory=parser[4],
            ocr_text=easyocr_val,
            llm_text='',
            ocr_method="easyocr",
            llm_method='none',
            font='',
            coordinates=parser[1],
            segmentation_method='Layout_Parser'
            )

        if apple_text is not None and apple_text != "" and apple_text != "None" and apple_text != " ":
            create_article_entry(
            newspaper_issue_id=1288,
            page_number=parser[3],
            title='',
            content="",
            author='',
            date=None,
            directory=parser[4],
            ocr_text=apple_text,
            llm_text='',
            ocr_method="apple_vision_ocr",
            llm_method='none',
            font='',
            coordinates=parser[1],
            segmentation_method='Layout_Parser'
            )

        if docling_text is not None and docling_text != "" and docling_text != "None" and docling_text != " ":
            create_article_entry(
            newspaper_issue_id=1288,
            page_number=parser[3],
            title='',
            content="",
            author='',
            date=None,
            directory=parser[4],
            ocr_text=docling_text,
            llm_text='',
            ocr_method="docling_ocr",
            llm_method='none',
            font='',
            coordinates=parser[1],
            segmentation_method='Layout_Parser'
            )
        
        if docling_default_pipeline is not None and docling_default_pipeline != "" and docling_default_pipeline != "None" and docling_default_pipeline != "":
            create_article_entry(
            newspaper_issue_id=1288,
            page_number=parser[3],
            title='',
            content="",
            author='',
            date=None,
            directory=parser[4],
            ocr_text=docling_default_pipeline,
            llm_text='',
            ocr_method="docling_ocr_default_pipeline",
            llm_method='none',
            font='',
            coordinates=parser[1],
            segmentation_method='Layout_Parser'
            )

        if abbyy_ocr_val is not None and abbyy_ocr_val != "" and abbyy_ocr_val != "None" and abbyy_ocr_val != " ":
            create_article_entry(
            newspaper_issue_id=1288,
            page_number=parser[3],
            title='',
            content="",
            author='',
            date=None,
            directory=parser[4],
            ocr_text=abbyy_ocr_val,
            llm_text='',
            ocr_method="abbyy_ocr",
            llm_method='none',
            font='',
            coordinates=parser[1],
            segmentation_method='Layout_Parser'
            )




        # mark the article as processed
        mark_article_as_processed(parser[0])

def display_article(article):
    try:
        conn = sql_setup.sqlConnection()
        cursor = conn.cursor()
        cursor.execute(f"select * from newspaper_issues where id = {article[2]}")
        newspaper_issue = cursor.fetchone()
        dir = str(os.getcwd()) + str(newspaper_issue[5])
        pdf_files = list(filter(lambda f: f.lower().endswith('.pdf'), os.listdir(dir)))
        pdf_files.sort()
        pdf_file = pdf_files[0]
        pdf_path = os.path.join(dir, pdf_file)

        images = pdf2image.convert_from_path(pdf_path)
        images = images[article[3] - 1]

        # parse the coordinates as a list of lists, each with two floats
        coordinates = article[1].strip('[[').strip(']]').split('], [')


        coordinates = [list(map(float, coord.split(','))) for coord in coordinates]
        images = images.crop((coordinates[0][0], coordinates[0][1], coordinates[2][0], coordinates[2][1]))

        return images
    except Exception as e:
        article_directory = str('/Users/stergioskonstantinidis/Documents/GitHub/HistoricData/data/competition-dataset-public/PNG/') + str(article[4].split('/')[-1])
        image_ = Image.open(article_directory)
        coordinates = article[1].strip('[[').strip(']]').split('], [')


        coordinates = [list(map(float, coord.split(','))) for coord in coordinates]
        images = image_.crop((coordinates[0][0], coordinates[0][1], coordinates[2][0], coordinates[2][1]))

        return images


while True:
    article = get_next_article()
    if not article:
        print("No more articles to process.")
        break

    images = display_article(article)

    


    # Perform OCR using different methods
    if True:
        pytesseract_val = pytesseract_ocr(images)
        if pytesseract_val == '' or pytesseract_val == 'None' or pytesseract_val == ' ':
            pytesseract_val = None
    

    try:
        easyocr_val = easyocr_ocr(images)
        if easyocr_val == '' or easyocr_val == 'None' or easyocr_val == ' ':
            easyocr_val = None
    except Exception as e:
        print(f"Error occurred in easyocr OCR: {e}")
        easyocr_val = None

    try:
        apple_text = apple_ocr_fun(images)
        if apple_text == 'Unable to perform OCR with Apple OCR. Ensure you are using an Apple device with macOS 13 or later.':
            apple_text = None
    except Exception as e:
        print(f"Error occurred in Apple OCR: {e}")
        apple_text = None

    try:
        docling_text = None # docling_ocr(images)
        if docling_text == '' or docling_text == 'None' or docling_text == ' ':
            docling_text = None
    except Exception as e:
        print(f"Error occurred in docling OCR: {e}")
        docling_text = None

    try:
        docling_default_pipeline_text = docling_default_pipeline(images)
        if docling_default_pipeline_text == '' or docling_default_pipeline_text == 'None' or docling_default_pipeline_text == ' ':
            docling_default_pipeline_text = None
    except Exception as e:
        print(f"Error occurred in Docling default pipeline OCR: {e}")
        docling_default_pipeline_text = None

    try:
        conn = sql_setup.sqlConnection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM newspaper_issues WHERE id = %s", (article[2],))
        newspaper_issue = cursor.fetchone()
        directory = str(os.getcwd()) + str(newspaper_issue[5])

        abby_ocr_val = abby_fine_reader(directory=directory.replace(os.getcwd(), ""), page=article[3], coordinates=article[1])
    except Exception as e:
        print(f"Error occurred in Abbyy OCR: {e}")
        abby_ocr_val = None

    # Submit the ground truth if more than 3 OCR methods are available and do not return None or empty strings
    #count non-empty OCR methods
    ocr_methods = [pytesseract_val, easyocr_val, apple_text, docling_text, docling_default_pipeline_text, abby_ocr_val]
    if sum(1 for method in ocr_methods if method) > 2:
        print("Submitting ground truth...")
        
    
        submit_groundtruth(pytesseract_val, easyocr_val, apple_text, docling_text, docling_default_pipeline_text, abby_ocr_val)
    else:
        print("Not enough valid OCR methods to submit ground truth. Skipping this article.")
        mark_article_as_processed(article[0])
    try:
        cursor.close()
    except Exception as e:
        print(f"Error occurred while closing cursor: {e}")
    try:
        conn.close()
    except Exception as e:
        print(f"Error occurred while closing connection: {e}")