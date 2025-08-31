import os
import PIL
import pytesseract
import easyocr
from apple_ocr.ocr import OCR
import pandas as pd
import docling
from huggingface_hub import snapshot_download
import pdf2image
from docling.datamodel.pipeline_options import RapidOcrOptions, PdfPipelineOptions
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    InputFormat,
    ImageFormatOption,
    PdfFormatOption,
)
def visualize_article(dir, coordinates, page):
    """ Used for abby fine reader, to visualize the article and get the coordinates of the text."""
    pdf_files = list(filter(lambda f: f.endswith('.pdf'), os.listdir(dir)))
    pdf_files.sort()
    pdf_file = pdf_files[0]
    pdf_path = os.path.join(dir, pdf_file)
    images = pdf2image.convert_from_path(pdf_path)
    images = images[int(page) - 1]
    image_width, image_height = images.size

    images = images.crop((coordinates[0][0], coordinates[0][1], coordinates[2][0], coordinates[2][1]))
    return images, image_width, image_height

from functools import reduce

# Ensure French language data is available for pytesseract

def pytesseract_ocr(image, lang='fra'):
    """
    Perform OCR on the given image using Tesseract.

    :param image: the image blob.
    :param lang: Language for OCR (default is French).
    :return: Extracted text from the image.
    """
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        print('error', e)
        return None
    
def easyocr_ocr(image, lang='fr'):
    """
    Perform OCR on the given image using EasyOCR.

    :param image: the image blob.
    :param lang: Language for OCR (default is French).
    :return: Extracted text from the image.
    """
    image.save('test.png')
    reader = easyocr.Reader(lang_list=[lang])  # Initialize EasyOCR reader with specified language

    result = reader.readtext('test.png', detail=0, paragraph=True)

    return reduce(lambda x, y: x + ' ' + y, result) if result else None

def apple_ocr_fun(image, lang='fran'):
    """ Perform OCR on the given image using Apple OCR.
    :param image: the image blob.
    :param lang: Language for OCR (default is French).
    :return: Extracted text from the image."""
    # Check os
    import platform
    if platform.system() != 'Darwin':
        raise EnvironmentError("Apple OCR is only supported on macOS 13 or later.")

    try:
        ocr = OCR(image=image)
        apple_ocr_text = ocr.recognize()
        text = pd.DataFrame(apple_ocr_text)['Content'].to_list()
        text = ' '.join(text)
    except:
        text = 'Unable to perform OCR with Apple OCR. Ensure you are using an Apple device with macOS 13 or later.'

    return text


def docling_ocr(image, lang='fra'):
    """
    Perform OCR on the given image using Docling.

    :param image: the image blob.
    :param lang: Language for OCR (default is French).
    :return: Extracted text from the image.
    """
    # Download RappidOCR models from HuggingFace
    
    download_path = snapshot_download(repo_id="SWHL/RapidOCR")

    # Setup RapidOcrOptions for french detection
    det_model_path = os.path.join(
        download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "PP-OCRv3", "en_PP-OCRv3_rec_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
    )
    ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
    )

    pipeline_options = PdfPipelineOptions(
        ocr_options=ocr_options,
    )

    
    # Convert the document
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )

    # Convert the image to pdf
    image.save('temp_image.pdf', format='PDF')
    conversion_result: ConversionResult = converter.convert(source='temp_image.pdf')
    doc = conversion_result.document
    md = doc.export_to_markdown()
    os.remove('temp_image.pdf')  # Clean up the temporary file
    md = md.replace('**', '')  # Remove bold formatting for simplicity
    md = md.replace('`', '')  # Remove code formatting for simplicity
    md = md.replace('\n', ' ')  # Replace newlines with spaces for better readability
    md = md.strip()  # Remove leading and trailing whitespace
    md = md.replace('<!-- image -->', ' ')  # Replace image placeholders with spaces
    md = md.replace('#', ' ')  # Replace hashtags with spaces
    for i in range(1000):
        md = md.replace('  ', ' ')  # Replace double spaces with single space

    return md

def docling_default_pipeline(image, lang='fra'): #unused
    """
    Perform OCR on the given image using Docling's default pipeline.

    :param image: the image blob.
    :param lang: Language for OCR (default is French).
    :return: Extracted text from the image.
    """
    # Convert the image to pdf
    image.save('temp_image.pdf', format='PDF')
    
    converter = DocumentConverter()
    result = converter.convert(source='temp_image.pdf')

    doc = result.document
    md = doc.export_to_markdown()


    md = md.replace('**', '')  # Remove bold formatting for simplicity
    md = md.replace('`', '')  # Remove code formatting for simplicity
    md = md.replace('\n', ' ')  # Replace newlines with spaces for better readability
    md = md.replace('_', ' ')  # Replace underscores with spaces for better readability
    md = md.strip()  # Remove leading and trailing whitespace
    md = md.replace('<!-- image -->', ' ')  # Replace image placeholders with spaces
    md = md.replace('#', ' ')  # Replace hashtags with spaces
    for i in range(1000):
        md = md.replace('  ', ' ')  # Replace double spaces with single space

    try:
        os.remove('temp_image.pdf')  # Clean up the temporary file
    except:
        pass

    return md

def docling_md_pipeline(image, lang='fra'):
    """
    Perform OCR on the given image using Docling's markdown pipeline.

    :param image: the image blob.
    :param lang: Language for OCR (default is French).
    :return: Extracted text from the image.
    """
    # Convert the image to pdf
    image.save('temp_image.pdf', format='PDF')
    
    converter = DocumentConverter()
    result = converter.convert(source='temp_image.pdf')

    doc = result.document
    md = doc.export_to_markdown()

    os.remove('temp_image.pdf')  # Clean up the temporary file
    
    return md


def abby_fine_reader(image=None, lang='fra', directory=None, coordinates=None, page=None):
    """
    Perform OCR on the given image using ABBYY FineReader.
    This function is designed to work with images that have been previously processed by ABBYY OCR, where we have an xml file with coordinates of the text.
    :param image: Optional the image blob.
    :param dir: Required directory containing the XML file.
    :param coordinates: Required coordinates of the image in the PDF. Coordinates must be (lower bound, upper bound, left bound, right bound).
    :param lang: Language for OCR (default is French).
    :return: Extracted text from the image.
    """
    if image is not None:
        raise NotImplementedError("This function is designed to work with images that have been previously processed by ABBYY OCR, where we have an xml file with coordinates of the text.")
    else:
        chars = []
        directory = str(os.getcwd()) + str(directory)
        coordinates = coordinates.strip('[[').strip(']]').split('], [')
        coordinates = [list(map(float, coord.split(','))) for coord in coordinates]
        image, image_width, image_height = visualize_article(directory, coordinates, page)


        xml_files = list(filter(lambda f: f.endswith('.xml'), os.listdir(directory)))


        xml_files.sort()
        xml_file = xml_files[int(page) - 1]  
        xml_path = os.path.join(directory, xml_file)

        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        
        # split xml content by lines
        xml_lines_ = xml_content.split('\n')
        try:
            pdf_width = float(xml_lines_[0].split('width="')[1].split('"')[0])  # get the width of the pdf
            pdf_height = float(xml_lines_[0].split('height="')[1].split('"')[0])  # get the height of the pdf
        except:
            for i in range(100):
                try:
                    pdf_width = float(xml_lines_[i].split('WIDTH="')[1].split('"')[0])  # get the width of the pdf
                    pdf_height = float(xml_lines_[i].split('HEIGHT="')[1].split('"')[0])  # get the height of the pdf
                    print("success")
                    break
                except:
                    continue


        #keep lines that start with <charParams
        xml_lines = [line for line in xml_lines_ if line.startswith('<charParams')]


        for line in xml_lines:
            l = int(line.split('l="')[1].split('"')[0]) * (image_width / pdf_width)  # left coordinate
            t = int(line.split('t="')[1].split('"')[0]) * (image_height / pdf_height) # top coordinate
            r = int(line.split('r="')[1].split('"')[0]) * (image_width / pdf_width)  # right coordinate
            b = int(line.split('b="')[1].split('"')[0]) * (image_height / pdf_height) # bottom coordinate
            character = line.split('">')[1].split('</charParams')[0]


            if ((float(l)+float(r))/2) >= coordinates[0][0] and ((float(t) + float(b))/2) >= coordinates[0][1] and ((float(r) + float(l))/2) <= coordinates[2][0] and ((float(b) + float(t))/2) <= coordinates[2][1]:
                chars.append(character)
        string_of_chars = ''.join(chars).replace('&apos;', "'").replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        if string_of_chars and string_of_chars != ' ' and string_of_chars != '\n' and string_of_chars != '':
            return string_of_chars
        print("fallback")
        # Fallback to parsing the XML file if the above fails
        print("page:", page, "coordinates:", coordinates)
        print("Parsing XML file for text extraction... xml_files:", xml_path)
        # process the following string format if the above fails <String ID="S37" CONTENT="ville," WC="0.870" CC="5 2 0 0 0 0" STYLE="superscript" HEIGHT="52" WIDTH="117" HPOS="305" VPOS="3021"/>
        chars = []
        xml_lines = [line for line in xml_lines_ if line.startswith('<String')]
        for line in xml_lines:
            l = int(line.split('HPOS="')[1].split('"')[0]) * (image_width / pdf_width)  # left coordinate
            t = int(line.split('VPOS="')[1].split('"')[0]) * (image_height / pdf_height)  # top coordinate
            r = l + int(line.split('WIDTH="')[1].split('"')[0]) * (image_width / pdf_width)  # right coordinate
            b = t + int(line.split('HEIGHT="')[1].split('"')[0]) * (image_height / pdf_height)  # bottom coordinate
            
            if ((float(l) + float(r)) / 2) >= coordinates[0][0] and ((float(t) + float(b)) / 2) >= coordinates[0][1] and ((float(r) + float(l)) / 2) <= coordinates[2][0] and ((float(b) + float(t)) / 2) <= coordinates[2][1]:
                
                content = line.split('CONTENT="')[1].split('"')[0] + ' '
                chars.append(content)
        string_of_chars = ''.join(chars).replace('&apos;', "'").replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        if string_of_chars and string_of_chars != ' ' and string_of_chars != '\n' and string_of_chars != '':
            return string_of_chars

if __name__ == "__main__":
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
        cursor.execute("SELECT MAX(id) FROM articles")
        id = cursor.fetchone()[0] + 1
        cursor.execute("""
            INSERT INTO articles (id, newspaper_issue_id, page_number, title, content, author, date, directory, ocr_text, llm_text, ocr_method, llm_method, manual_correction_text, font, coordinates, segmentation_method)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (id, newspaper_issue_id, page_number, title, content, author, date, directory, ocr_text, llm_text, ocr_method, llm_method, manual_correction_text, font, str(coordinates), segmentation_method))
        conn.commit()
        cursor.close()
        conn.close()




if __name__ == "__main__":

    # Example usage
    import os
    import sys
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 
    import pdf2image
    import PIL
    import sql_setup
    conn = sql_setup.sqlConnection()
    cursor = conn.cursor()
    cursor.execute("SELECT directory, coordinates, page_number, newspaper_issue_id, title, content, author, date, manual_correction_text FROM articles")
    articles = cursor.fetchall()
    for article in articles:
        text = abby_fine_reader(directory=article[0], coordinates=article[1], page=article[2])
        print(text)

        create_article_entry(
            newspaper_issue_id=article[3],
            page_number=article[2],
            title=article[4],
            content=article[5],
            author=article[6],
            date=article[7],
            directory=article[0],
            ocr_text=text,
            llm_text='',
            ocr_method='abby_fine_reader',
            llm_method='none',
            manual_correction_text=article[8],
            font='',
            coordinates=article[1],
            segmentation_method='Layout_Parser'
        )


        
