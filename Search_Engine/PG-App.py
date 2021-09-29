# -*- coding: utf-8 -*-
# @Author: mjacoupy
# @Date:   2021-09-29 11:02:47
# @Last Modified by:   mjacoupy
# @Last Modified time: 2021-09-29 11:43:56

import streamlit as st
import s3fs
from PIL import Image
import cv2
import numpy as np

fs = s3fs.S3FileSystem(anon=False)


@st.cache(ttl=600)
def read_file(filename):
    """..."""

    infile = fs.open(filename, "rb")
    image = Image.open(infile)
    # image = cv2.imdecode(np.asarray(bytearray(infile)), cv2.IMREAD_COLOR)

    return image

#content = "ocrplus-ptc/ARDIAN_Comptes sociaux2019_p4.pdf"
content = "ocrplus-ptc/Page_6.jpeg"

image = read_file(content)

st.image(image, caption="first test")


infile = fs.open(content, "rb")
st.text(infile)
st.text(infile.type())
