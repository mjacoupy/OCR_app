#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:15:19 2021

@author: maximejacoupy
"""

# #######################################################################################################################
#                                              # === LIBRAIRIES === #
# #######################################################################################################################
import streamlit as st
from PIL import Image
import numpy as np
from PyPDF2 import PdfFileReader
from SearchEngine_app import SearchEngine
from OCRPlus_app import OCRPlus
import pandas as pd
import re
import os


from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import pathlib


# #######################################################################################################################
#                                              # === S3 AWS === #
# #######################################################################################################################

local = True
if local is False:
    import boto3
    from io import BytesIO
    bucket_name = "ocrplus-ptc"
    item_name = "ARDIAN_Comptes sociaux2019_p4.pdf"

    # content = read_file("ocrplus-ptc/ARDIAN - Comptes sociaux 2019.pdf")
    # content = read_file("ocrplus-ptc/Page_6.jpeg")
    # content = read_file("ocrplus-ptc/ARDIAN_Comptes sociaux2019_p4.pdf")
    # content = "ocrplus-ptc/ARDIAN_Comptes sociaux2019_p4.pdf"
    content = "ocrplus-ptc/ARDIAN - Comptes sociaux 2019.pdf"

# #######################################################################################################################
#                                              # === FUNCTIONS === #
# #######################################################################################################################


def my_split(s, seps):
    """..."""
    res = [s]
    for sep in seps:
        s, res = res, []
        for seq in s:
                res += seq.split(sep)
    return res


def side_bar():
    """..."""
    end = '<p style="font-family:Avenir; font-weight:bold; color:#FCBA28; font-size:12px; ">©2021 Positive Thinking Company et/ou ses affiliés. Tous droits réservés. Produit par le PTC Tech Lab.</p>'
    st.sidebar.markdown("""---""")
    st.sidebar.image(image1, width=50)
    st.sidebar.markdown(end, unsafe_allow_html=True)
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #

# #######################################################################################################################
#                                              # === PROCESS NEW FILE === #
# #######################################################################################################################
st.title("OCRplus - ARDIAN")
st.markdown("""---""")

image1 = Image.open("app_logos/PTCtechLab.png")
image2 = Image.open("app_logos/PTC.png")
st.sidebar.image(image2, width=200)


analysis = st.sidebar.selectbox('', ['[1] Image Processing', '[2] Indexation', '[3] Search Engine'])

if analysis == "[1] Image Processing":
    st.header('Image Processing')

    st.sidebar.markdown('Be sure to have a se_png folder')
    st.sidebar.markdown('Be sure to have a se_txt folder')

    side_bar()


    box_save = st.radio('Save export ?', ('Yes', 'No'))
    if box_save == 'Yes':
        box_save = True
    elif box_save == 'No':
        box_save = False

    save = st.radio('Export format (choose 1)', ('box contents', 'box relations', 'whole structure', 'text', 'image', 'all formats'), index=5)
    if save == 'box contents':
        save = 'bc'
    elif save == 'box relations':
        save = 'br'
    elif save == 'whole structure':
        save = 'ws'
    elif save == 'text':
        save = 'txt'
    elif save == 'image':
        save = 'png'
    elif save == 'all formats':
        save = 'all'

    if local:
        folder_path = os.path.join(os.path.abspath(os.getcwd()), "ocr_doc_to_process")
        ocrplus = OCRPlus(path=folder_path, neo4j_location="local")
        docs = os.listdir(ocrplus.path_to_documents)
        try:
            docs.remove('.DS_Store')
        except ValueError:
            pass
    else:
        docs = [content]


    st.markdown("**"+str(len(docs))+"** documents will be analyzed")
    button_t = st.button('Analyze documents')

    if button_t:
        lenghts = []
        for iCpt, iDoc in enumerate(docs):

            if local:
                doc_path = os.path.join(folder_path, iDoc)
                pdf = PdfFileReader(open(doc_path, 'rb'))
                lenght = pdf.getNumPages()
                st.markdown('Document **'+str(iDoc)+"** contains **"+str(lenght)+"** page(s)")
                l = list(range(1, lenght+1))
                lenghts.append(l)

            ######

            else:

                s3 = boto3.resource('s3')
                obj = s3.Object(bucket_name, item_name)
                fs = obj.get()['Body'].read()
                pdf = PdfFileReader(BytesIO(fs))

                lenght = pdf.getNumPages()
                st.markdown('Document **'+str(iDoc)+"** contains **"+str(lenght)+"** page(s)")
                l = list(range(1, lenght+1))
                lenghts.append(l)


        ocrplus.set_pages(lenghts)
        ocrplus.process_documents(save_data=box_save, save_type=[save])

        for iDoc in docs:
            os.remove(os.path.join(folder_path, iDoc))

        st.markdown("**Done**")
        st.balloons()

# #######################################################################################################################
#                                              # === INDEXER === #
# #######################################################################################################################
if analysis == "[2] Indexation":
    st.header('Indexation')

    side_bar()

    folder_path = os.path.join(os.path.abspath(os.getcwd()), "ocr_exports", "se_txt")

    txt = st.text_input('root', folder_path)
    button = st.button('Run')
    if txt and button:

        my_bar = st.progress(0)
        schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
        if not os.path.exists("se_indexdir"):
            os.mkdir("se_indexdir")

        # Creating a index writer to add document as per schema
        ix = create_in("se_indexdir", schema)
        writer = ix.writer()

        filepaths = [os.path.join(txt, i) for i in os.listdir(txt)]

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
