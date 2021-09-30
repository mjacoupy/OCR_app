# -*- coding: utf-8 -*-
# @Author: mjacoupy
# @Date:   2021-09-29 11:02:47
# @Last Modified by:   mjacoupy
# @Last Modified time: 2021-09-30 09:26:02

import streamlit as st
import s3fs
from PIL import Image
import numpy as np
import pytesseract
import boto3

fs = s3fs.S3FileSystem(anon=False)
bucket_name = "ocrplus-app-mja"
# content = "ocrplus-ptc/Page_6.jpeg"
# content = "ocrplus-app-mja/03_ARDIAN_P4.jpeg"

s3 = boto3.resource('s3')
my_bucket = s3.Bucket(bucket_name)

docs = []
for file in my_bucket.objects.all():
    docs.append(file.key)


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

    return str_text




    # -------------------

st.markdown(docs)

select = st.selectbox('Which document', docs)

select_path = bucket_name+"/"+select

if select:
    image = read_file(select_path)

    st.image(image, caption="first test")

    button = st.button('OCR analysis')

    if button:

        str_text = extract_content_to_txt(image)
        st.markdown(str_text)

        out_file = "text.txt"

        with open(out_file, "w") as text_file:
            text_file.write(str_text)

        with open(out_file, "rb") as f:
            s3.upload_fileobj(f, bucket_name, "test")

        # with open("FILE_NAME", "rb") as f:
        #     s3.upload_fileobj(f, "BUCKET_NAME", "OBJECT_NAME")




        # import boto3

        # some_binary_data = b'Here we have some data'
        # more_binary_data = b'Here we have some more data'

        # # Method 1: Object.put()
        # s3 = boto3.resource('s3')
        # object = s3.Object('my_bucket_name', 'my/key/including/filename.txt')
        # object.put(Body=some_binary_data)

        # # Method 2: Client.put_object()
        # client = boto3.client('s3')
        # client.put_object(Body=more_binary_data, Bucket='my_bucket_name', Key='my/key/including/anotherfilename.txt')

