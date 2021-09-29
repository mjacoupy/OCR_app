# -*- coding: utf-8 -*-
# @Author: mjacoupy
# @Date:   2021-09-29 11:02:47
# @Last Modified by:   mjacoupy
# @Last Modified time: 2021-09-29 17:47:35

import streamlit as st
import s3fs
from PIL import Image
import numpy as np
import pytesseract

fs = s3fs.S3FileSystem(anon=False)
# content = "ocrplus-ptc/Page_6.jpeg"
content = "ocrplus-app-mja/03_ARDIAN_P4.jpeg"


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


image = read_file(content)

st.image(image, caption="first test")

button = st.button('OCR analysis')

if button:
    str_text = extract_content_to_txt(image)
    st.markdown(str_text)

