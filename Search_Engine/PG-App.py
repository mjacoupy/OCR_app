# -*- coding: utf-8 -*-
# @Author: mjacoupy
# @Date:   2021-09-29 11:02:47
# @Last Modified by:   mjacoupy
# @Last Modified time: 2021-09-30 11:27:22


# #######################################################################################################################
#                                              # === LIBRAIRIES === #
# #######################################################################################################################
import streamlit as st
import s3fs
from PIL import Image
import numpy as np
import pytesseract
import boto3
import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import pathlib

# #######################################################################################################################
#                                              # === S3 AWS === #
# #######################################################################################################################
fs = s3fs.S3FileSystem(anon=False)
bucket_name = "ocrplus-app-mja"
bucket_name_txt = "ocrplus-app-mja-txt"


s3 = boto3.resource('s3')
my_bucket = s3.Bucket(bucket_name)
my_bucket2 = s3.Bucket(bucket_name_txt)

docs = []
for file in my_bucket.objects.all():
    docs.append(file.key)

# docs.remove('text_files/')

# #######################################################################################################################
#                                              # === FUNCTIONS === #
# #######################################################################################################################
@st.cache(ttl=600)
def read_file(filename):
    """..."""

    infile = fs.open(filename, "rb")
    pil_image = Image.open(infile).convert('RGB')
    open_cv_image = np.array(pil_image)
    image = open_cv_image[:, :, ::-1].copy()
    return image


def clean_text(ext_text):
    """Clean extracted text obtain by pytesseract.

    :param ext_text: text of each box
    :type ext_text: str
    :returns: cleaned text
    :rtype: list
    """
    splits = ext_text.splitlines()
    storage_list = []

    for iWord in splits:
        if iWord not in ['', ' ']:
            storage_list.append(iWord)

    return storage_list


def extract_content_to_txt(image):
    """Extract raw text from page.

    :returns: raw text
    :rtype: str
    """
    ext_text = pytesseract.image_to_string(image, lang='fra')
    text = clean_text(ext_text)
    str_text = ' '.join(text)

    return str(str_text)



def side_bar():
    """..."""
    end = '<p style="font-family:Avenir; font-weight:bold; color:#FCBA28; font-size:12px; ">©2021 Positive Thinking Company et/ou ses affiliés. Tous droits réservés. Produit par le PTC Tech Lab.</p>'
    st.sidebar.markdown("""---""")
    st.sidebar.image(image1, width=50)
    st.sidebar.markdown(end, unsafe_allow_html=True)

# #######################################################################################################################
#                                              # === APPEARANCE === #
# #######################################################################################################################
st.title("OCRplus - ARDIAN")
st.markdown("""---""")

image1 = Image.open("app_logos/PTCtechLab.png")
image2 = Image.open("app_logos/PTC.png")
st.sidebar.image(image2, width=200)

analysis = st.sidebar.selectbox('', ['[1] Image Processing', '[2] Indexation', '[3] Search Engine'])
# #######################################################################################################################
#                                              # === PROCESS NEW FILE === #
# #######################################################################################################################

if analysis == "[1] Image Processing":
    st.header('Image Processing')

    side_bar()

    docs_all = docs.copy()
    docs_all.append('All')

    select = st.selectbox('Which document', docs_all)
    button = st.button('OCR analysis')

    if select == 'All' and button:
        for doc in docs:
            select_path = bucket_name+"/"+doc
            image = read_file(select_path)
            str_text = extract_content_to_txt(image)
            out_file = str(doc)+'.txt'
            s3.Object(bucket_name_txt, out_file).put(Body=str_text)

    elif select != 'All' and button:
        select_path = bucket_name+"/"+select
        image = read_file(select_path)

        st.image(image, caption=select)
        str_text = extract_content_to_txt(image)
        st.markdown(str_text)
        out_file = str(select)+'.txt'
        s3.Object(bucket_name_txt, out_file).put(Body=str_text)

        # export_path = os.path.join(os.path.abspath(os.getcwd()), "ocr_exports", "se_txt")

        # out_txt_file = export_path + out_file

        # with open(out_txt_file, "w") as text_file:
        #     text_file.write(str_text)
# #######################################################################################################################
#                                              # === INDEXER === #
# #######################################################################################################################
if analysis == "[2] Indexation":
    st.header('Indexation')

    side_bar()

    folder_path = os.path.join(os.path.abspath(os.getcwd()), "ocr_exports", "se_txt")

    txt = st.text_input('root', folder_path)
    button = st.button('Run')

    if button:

        my_bar = st.progress(0)
        schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
        if not os.path.exists("se_indexdir"):
            os.mkdir("se_indexdir")

        # Creating a index writer to add document as per schema
        ix = create_in("se_indexdir", schema)
        writer = ix.writer()

        filepaths = [os.path.join(txt, i) for i in os.listdir(txt)]

        st.markdown(filepaths)

        path = pathlib.PurePath(txt)
        last = path.name
        for path, percent in zip(filepaths, range(len(filepaths))):

            val = (percent+1) / len(filepaths)
            my_bar.progress(val)

            # Do not select empty document
            try:
                fp = open(path, 'r')
                text = fp.read()
                writer.add_document(title=path.split(last+"/")[1], path=path, content=text, textdata=text)
                fp.close()
            except UnicodeDecodeError:
                pass

        writer.commit()

