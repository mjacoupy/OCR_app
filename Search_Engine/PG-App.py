# -*- coding: utf-8 -*-
# @Author: mjacoupy
# @Date:   2021-09-29 11:02:47
# @Last Modified by:   mjacoupy
# @Last Modified time: 2021-09-29 11:47:05

import streamlit as st
import s3fs
from PIL import Image
import cv2
import numpy as np

fs = s3fs.S3FileSystem(anon=False)
content = "ocrplus-ptc/Page_6.jpeg"

@st.cache(ttl=600)
def read_file(filename):
    """..."""

    infile = fs.open(filename, "rb")
    pil_image = Image.open(infile).convert('RGB')
    open_cv_image = numpy.array(pil_image)
    image = open_cv_image[:, :, ::-1].copy()
    return image



image = read_file(content)

st.image(image, caption="first test")

st.text(image)

