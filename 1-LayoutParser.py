import sys
import os
import inspect
from pdf2image import convert_from_path
import PIL
import PIL.ImageDraw
import os
import inspect
import json
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import sql_setup
import pdf2image
import torch
torch.cuda.empty_cache()
# set up environment
conn = sql_setup.sqlConnection()
cursor = conn.cursor()

# download config and weights files
import urllib.request
urllib.request.urlretrieve("https://www.dropbox.com/s/yc92x97k50abynt/config.yml?dl=1", "config.yml")
urllib.request.urlretrieve("https://www.dropbox.com/s/h7th27jfv19rxiy/model_final.pth?dl=1", "model.pth")

# load model using local files
import layoutparser as lp
model = lp.models.Detectron2LayoutModel(config_path ='config.yml',
                                 model_path="model.pth",
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

def generate_entries(model=model, cursor=cursor):

    cursor.execute("SELECT directory, id, date FROM newspaper_issues where id not in (select newspaper_issue_id from Layout_Parser_for_RAG)") 
    files = cursor.fetchall()

    print("Files to process:", files)
    for index, directory in enumerate(files[0::]):
        print(f"Processing file {index + 1}/{len(files)}: {directory}")

        file = list(filter(lambda x: str.lower(x).endswith('.pdf'), os.listdir(str(os.getcwd()) + directory[0])))
        file = str(os.getcwd()) + str(directory[0]) + "/" + str(file[0])




        images = pdf2image.convert_from_path(file, dpi=300)
        for page, document in enumerate(images):
            cursor.execute("SELECT * FROM Layout_Parser_for_RAG WHERE newspaper_issue_id = %s AND page = %s", (directory[1], page + 1))
            if cursor.fetchone() is not None:
                print(f"Skipping page {page + 1} of {directory[1]} as it has already been processed.")
                continue

            layout = model.detect(document)

            img = convert_from_path(file)[page]



            width = document.width
            height = document.height

            for cell in layout:

                # Here you can add code to visualize the cell in the source document
                # For example, using a PDF viewer or an image viewer to highlight the cell
                draw = PIL.ImageDraw.Draw(img)




                img_width, img_height = img.size
                coordinates = [
                    (cell.block.x_1 * img_width/width, cell.block.y_1  * img_height/height),
                    (cell.block.x_2 * img_width/width, cell.block.y_1 * img_height/height),
                    (cell.block.x_2 * img_width/width, cell.block.y_2 * img_height/height),
                    (cell.block.x_1 * img_width/width, cell.block.y_2 * img_height/height),
                ]
                print(coordinates)
                draw.polygon(coordinates, outline="red", width=2)
                conn = sql_setup.sqlConnection()
                cursor = conn.cursor()
                cursor.execute(
                "INSERT INTO Layout_Parser_for_RAG (newspaper_issue_id, page, coordinates) VALUES (%s, %s, %s)",
                ( directory[1], page + 1, json.dumps(coordinates))
                )
            print(f"Processed page {page + 1} of {len(images)} with {len(layout)} detected cells.")



generate_entries()