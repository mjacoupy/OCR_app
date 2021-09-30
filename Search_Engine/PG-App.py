# -*- coding: utf-8 -*-
# @Author: mjacoupy
# @Date:   2021-09-29 11:02:47
# @Last Modified by:   mjacoupy
# @Last Modified time: 2021-09-30 14:37:41


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
import cv2
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import pandas as pd
from SearchEngine_app import SearchEngine
import re

# #######################################################################################################################
#                                              # === S3 AWS === #
# #######################################################################################################################
fs = s3fs.S3FileSystem(anon=False)
bucket_name = "ocrplus-app-mja"
bucket_name_txt = "ocrplus-app-mja-txt"


s3 = boto3.resource('s3')
my_bucket = s3.Bucket(bucket_name)
my_bucket2 = s3.Bucket(bucket_name_txt)


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


def my_split(s, seps):
    """..."""
    res = [s]
    for sep in seps:
        s, res = res, []
        for seq in s:
                res += seq.split(sep)
    return res
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

    ##########################################################################################################################
    data = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg"])
    button = st.button("Process")

    if data is not None and button:
        image = Image.open(data)
        pil_image = Image.open(data).convert('RGB')
        open_cv_image = np.array(pil_image)
        image = open_cv_image[:, :, ::-1].copy()
        st.image(image, caption='Selected document')

        image_string = cv2.imencode('.jpg', image)[1].tostring()
        s3.Bucket(bucket_name).put_object(Key="test_new_image.jpg", Body=image_string, ACL='public-read')

    ##########################################################################################################################
    docs = []
    for file in my_bucket.objects.all():
        docs.append(file.key)


    docs_all = docs.copy()
    docs_all.append('All')

    l = len(docs_all)-1
    select = st.selectbox('Which document', docs_all, index=l)
    button = st.button('OCR analysis')

    if select == 'All' and button:
        st.markdown(docs)
        for doc in docs:
            select_path = bucket_name+"/"+doc
            st.markdown(select_path)
            image = read_file(select_path)
            name = doc.split('.')[0]
            str_text = extract_content_to_txt(image)
            out_file = str(name)+'.txt'
            s3.Object(bucket_name_txt, out_file).put(Body=str_text)

    elif select != 'All' and button:
        select_path = bucket_name+"/"+select
        image = read_file(select_path)
        name = select.split('.')[0]
        st.image(image, caption=name)
        str_text = extract_content_to_txt(image)
        out_file = str(name)+'.txt'
        s3.Object(bucket_name_txt, out_file).put(Body=str_text)

##########################################################################
#                                              # === INDEXER === #
# #######################################################################################################################
if analysis == "[2] Indexation":
    st.header('Indexation')

    side_bar()

    button = st.button('Run')

    if button:

        my_bar = st.progress(0)
        schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
        if not os.path.exists("se_indexdir"):
            os.mkdir("se_indexdir")

        # Creating a index writer to add document as per schema
        ix = create_in("se_indexdir", schema)
        writer = ix.writer()

        # filepaths = [os.path.join(txt, i) for i in os.listdir(txt)]

        filepaths = []
        for file in my_bucket2.objects.all():
            filepaths.append(file.key)

        st.markdown(filepaths)

        for name, percent in zip(filepaths, range(len(filepaths))):

            val = (percent+1) / len(filepaths)
            my_bar.progress(val)

            # Do not select empty document
            try:
                select_path = bucket_name_txt+"/"+name
                fp = fs.open(select_path, "rb")
                text = fp.read().decode('utf-8', 'ignore')
                # st.markdown(select_path)
                # st.markdown(name)
                # st.markdown(text)
                writer.add_document(title=name, path=select_path, content=text, textdata=text)
                fp.close()
            except UnicodeDecodeError:
                pass

        writer.commit()


# #######################################################################################################################
#                                              # === SEARCH ENGINE === #
# #######################################################################################################################
if analysis == "[3] Search Engine":

    # Create the Search Engine
    SE = SearchEngine()

    # Initialize variable
    languages = ['french', 'english', 'spanish', 'italian', 'german']
    is_response = False
    lang = None
    kw = doc = positive = score = None

# ##################################### == SIDE BAR == ##########################################################
    # Create header and subheader
    st.sidebar.markdown('1. Write your text')
    st.sidebar.markdown('2. Select the number of response')
    st.sidebar.markdown('3. Select in which language the keys word will be translated')
    st.sidebar.markdown('4. Choose the parameters you want to display')
    st.sidebar.markdown('5. Click on search')
    side_bar()

# ##################################### == PART 1 == ##########################################################
    st.header('Search Engine')
    st.subheader('Part1 - Search')

    # Ask user for parameters
    user_input = st.text_input("Request")
    response = st.radio('Number of response', ('Best', 'Most Relevant', 'All'), index=1)



    lang = st.multiselect('Which language', ['french', 'english', 'spanish', 'italian', 'german'], default=['french', 'english', 'spanish', 'italian', 'german'])

    st.write('Specific Parameters')
    kw = st.checkbox('Key words selected')
    doc = st.checkbox('Number of document in the database')
    positive = st.checkbox('Number of results')
    score = st.checkbox('Score of documents')
    all_of_them = st.checkbox('All', value=True)

    search_button = st.button("Search")

    # Create the variables
    if all_of_them:
        kw = doc = positive = score = 1

    # Launch the reaserch
    if user_input and lang and search_button:
        tmp = SE.analyze_in_different_language(user_input)

        # Count the number of positive document for each language
        doc_found = []
        for ilang in lang:
            try:
                doc_found.append(tmp[ilang]['Documents containing key words'])
            except TypeError:
                doc_found.append(0)

        # Prepare the data for the dataframe
        title_list = []
        score_list = []
        preview_list = []
        lang_list = []

        idx_list = list(range(1, sum(doc_found)+1))
        for iCpt, ilang in enumerate(lang):
            if doc_found[iCpt] != 0:
                for n in range(doc_found[iCpt]):
                    title_list.append(tmp[ilang][n]['Document'][:-4])
                    score_list.append(float(tmp[ilang][n]['Score']))
                    lang_list.append(ilang)

        # Create and clean the dataframe
        df = pd.DataFrame(title_list, columns=['Documents'])
        df['Language'] = lang_list
        df['Score'] = score_list
        df = df.sort_values(by=['Score'], ascending=False)
        df = df.drop_duplicates(subset=['Documents'])
        df.index = np.arange(1, len(df) + 1)

        # Print the informations if the are asked
        if kw:
            for ilang in lang:
                try:
                    st.markdown("Key Words selected in **"+ilang+"**: **"+str(tmp[ilang]['Key Words'])+"**")
                except TypeError:
                    pass
        if doc:
            argmax = np.argmax(doc_found)
            try:
                st.markdown("The Database contain: **"+str(tmp[lang[argmax]]['Documents to analyze'])+"** documents")
            except TypeError:
                pass
        if positive:
            st.markdown("**"+str(len(df))+"** results found in the Database")

        # Crop the daframe depending on the paramaters selected
        if response == "Best":
            df = df.iloc[:1]
        elif response == "Most Relevant":
            df = df.loc[df['Score'] > 3]
            st.markdown("**"+str(len(df))+"** relevant results found in the Database")
            if len(df) > 20:
                df = df.iloc[:20]

        if score:
            pass
        else:
            df = df.drop(['Score'], axis=1)

        # Print the Dataframe
        st.write(df)


        st.markdown("""---""")

 # ##################################### == PART 2 == ##########################################################
        st.subheader('Part2 - Show')

        # Create list of document and language for the previews
        tmp_doc_list = [doc for doc in df['Documents']]
        tmp_lang = [l for l in df['Language']]

        if 0 < len(tmp_doc_list):

            # find localisation of key word in the text and create the previews

            for (iCpt, txt), lang in zip(enumerate(tmp_doc_list[:20]), tmp_lang[:20]):
                sel_txt = txt+'.txt'
                sel_png = txt+'.png'
                for i in range(20):
                    try:
                        if sel_txt == tmp[lang][i]['Document']:
                            content = tmp[lang][i]["Content"]
                    except KeyError:
                        pass

                split_content = my_split(content, [" ", "'"])

                regex = re.compile('[,\.!?]')
                content_simplifyed = regex.sub('', content)
                simplifyed_split_content = my_split(content_simplifyed, [" ", "'"])

                final_content = ""
                for word, simplifyed_word in zip(split_content, simplifyed_split_content):
                    keywords = my_split(tmp[lang]['Key Words'], [" ", "'"])
                    if simplifyed_word.lower() in keywords:
                        modif_word = "**"+word+"** "
                    else:
                        modif_word = word+" "
                    final_content += modif_word

                lenght = []
                for iCpt2, letter in enumerate(final_content):
                    if letter == '*':
                        lenght.append(iCpt2)

                try:
                    final_content = '[...] '+str(final_content[lenght[0]:])
                    final_content = str(final_content[:300])+'[...]'
                except IndexError:
                    pass

                # print the selected previews
                st.markdown("___"+str(iCpt+1)+". "+txt+"___")
                st.markdown(final_content)
                try:
                    with st.expander("See original page"):
                        img = Image.open(os.path.join("ocr_exports", "se_png", sel_png))
                        st.image(img)
                except AttributeError:
                    pass
                st.markdown("""---""")
