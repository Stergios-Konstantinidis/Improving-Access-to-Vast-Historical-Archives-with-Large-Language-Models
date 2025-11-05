import os
import streamlit as st
import PIL
import PIL.ImageDraw
import os
import json
import pandas as pd

import torch
torch.cuda.empty_cache()
# set up environment

global images
images = []

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

def generate_entries(model=model, directory="/icdar2023_dataset/PNG/"):
    # Fetch newspaper issues to process

    files = list(filter(lambda x: x.lower().endswith('.png'), os.listdir(os.getcwd() +directory)))[0:10]
    df = pd.DataFrame({'file_path': [os.path.join(directory, f) for f in files]})

    print("Files to process:", files)
    for index, directory in df.iterrows():
        print(f"Processing file {index + 1}/{len(files)}: {directory}")

        file = str(os.getcwd()) + directory['file_path']


        image = PIL.Image.open(file)
        imagr = [image]

        layout_parser_df = pd.DataFrame(columns=['file_path', 'page', 'coordinates'])

        for page, document in enumerate(imagr):
            layout = model.detect(document)

            

            img = document.copy()

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

                # add entry to layout parser dataframe
                layout_parser_df.loc[len(layout_parser_df)] = [
                    file,
                    page + 1,
                    {
                        'type': cell.type,
                        'x_1': cell.block.x_1,
                        'y_1': cell.block.y_1,
                        'x_2': cell.block.x_2,
                        'y_2': cell.block.y_2,
                    }
                ]
        try:
            images.append(img)
        except:
            images = [img]

    
    layout_parser_df.to_csv(f"{file}_layout_parser_output.csv", index=False)

    return layout_parser_df, images
