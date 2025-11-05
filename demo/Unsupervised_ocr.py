import os
import sys
from ocr_tools import pytesseract_ocr, easyocr_ocr, apple_ocr_fun, docling_ocr, docling_default_pipeline, abby_fine_reader
import sql_setup

from PIL import Image




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