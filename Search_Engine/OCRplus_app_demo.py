# -*- coding: utf-8 -*-
# @Author: mjacoupy
# @Date:   2021-09-29 11:02:47
# @Last Modified by:   mjacoupy
# @Last Modified time: 2021-11-02 16:37:18


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
import pandas as pd
from SearchEngine_app import SearchEngine
import re
import cv2
from pdf2image import convert_from_bytes
from io import BytesIO
import matplotlib.image as mpimg


# #######################################################################################################################
#                                              # === S3 AWS CONNEXION === #
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

    # st.sidebar.markdown("1. **Import :** Ajout d'un nouveau document à analyser. Cela peut etre une image (png ou jpeg) ou un document PDF")
    # st.sidebar.markdown("2. **Processing :** Analyse OCR du ou des fichiers sélectionnés et création de l'indexer")
    # st.sidebar.markdown('2. **Moteur de recherche**')

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
st.title("OCR+")
st.header("Demonstration environment")
st.markdown("This OCR+ module has been developped by PTCTech Lab. It aims at interpreting various types of documents, such as bills, tickets, official reports, or financial documents.")
st.markdown("This demo is a concrete example of how the OCR+ module works in a various setting of cases. However, it needs to be adapted and tweaked to your context so as to be performing at its full capacity.")

st.markdown("""---""")

image1 = Image.open("app_logos/PTCtechLab.png")
image2 = Image.open("app_logos/PTC.png")
st.sidebar.image(image2, width=200)

analysis = st.sidebar.selectbox('', ['Search Engine', 'Import'])

# #######################################################################################################################
#                                              # === IMPORT NEW FILE === #
# #######################################################################################################################
if analysis == "Import":
    st.header("Import of a new document")

    side_bar()
    
    data = st.file_uploader("", type=["png", "jpg", "jpeg", "pdf"])
    if data:
        name = str(data.name)

    if data is not None and "pdf" in str(data.type):
        images = convert_from_bytes(data.read())
        text = 'Page number between **1** and **'+str(len(images))+"**"
        col1, col2 = st.columns([8, 2])
        with col1:
            int_val = st.number_input(text, min_value=1, max_value=len(images), value=1)
        with col2:
            button1 = st.button("Import")
            
        img = np.array(images[int_val-1])
        scale_percent = 20
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        st.image(resized)


        if data is not None and "pdf" in str(data.type) and button1:
            page = images[int_val-1]
            img = np.array(page)

            export_path = os.path.join(os.path.abspath(os.getcwd()), "ocr_doc_to_process/")
            out_file = export_path + str(name) + "-" + str(int_val) + ".png"
            cv2.imwrite(out_file, img)

            str_text = extract_content_to_txt(img)
            out_file = str(name)+'.txt'
            s3.Object(bucket_name_txt, out_file).put(Body=str_text)
        
            my_bar = st.progress(0)
            schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
            if not os.path.exists("se_indexdir"):
                os.mkdir("se_indexdir")
    
            # Creating a index writer to add document as per schema
            ix = create_in("se_indexdir", schema)
            writer = ix.writer()
    
            filepaths = []
            for file in my_bucket2.objects.all():
                filepaths.append(file.key)
    
            for name, percent in zip(filepaths, range(len(filepaths))):
    
                val = (percent+1) / len(filepaths)
                my_bar.progress(val)
    
                # Do not select empty document
                try:
                    select_path = bucket_name_txt+"/"+name
                    fp = fs.open(select_path, "rb")
                    text = fp.read().decode('utf-8', 'ignore')
                    writer.add_document(title=name, path=select_path, content=text, textdata=text)
                    fp.close()
                except UnicodeDecodeError:
                    pass
    
            writer.commit()  

            

    elif data is not None and "pdf" not in str(data.type):

        image = Image.open(data)
        pil_image = Image.open(data).convert('RGB')
        open_cv_image = np.array(pil_image)
        image2 = open_cv_image[:, :, ::-1].copy()

        scale_percent = 20
        width = int(image2.shape[1] * scale_percent / 100)
        height = int(image2.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)

        st.image(resized, caption='Selected document')

        button = st.button("Import")


        if data is not None and button:

            export_path = os.path.join(os.path.abspath(os.getcwd()), "ocr_doc_to_process/")
            out_file = export_path + str(name) + ".png"
            cv2.imwrite(out_file, image2)
            
            str_text = extract_content_to_txt(image2)
            out_file = str(name)+'.txt'
            s3.Object(bucket_name_txt, out_file).put(Body=str_text)
            
            my_bar = st.progress(0)
            schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
            if not os.path.exists("se_indexdir"):
                os.mkdir("se_indexdir")
    
            # Creating a index writer to add document as per schema
            ix = create_in("se_indexdir", schema)
            writer = ix.writer()
    
            filepaths = []
            for file in my_bucket2.objects.all():
                filepaths.append(file.key)
    
            for name, percent in zip(filepaths, range(len(filepaths))):
    
                val = (percent+1) / len(filepaths)
                my_bar.progress(val)
    
                # Do not select empty document
                try:
                    select_path = bucket_name_txt+"/"+name
                    fp = fs.open(select_path, "rb")
                    text = fp.read().decode('utf-8', 'ignore')
                    writer.add_document(title=name, path=select_path, content=text, textdata=text)
                    fp.close()
                except UnicodeDecodeError:
                    pass
    
            writer.commit()  
        


# #######################################################################################################################
#                                              # === PROCESS NEW FILE(S) === #
# #######################################################################################################################
if analysis == "Processing":
    st.header('Analyse du document')

    side_bar()

    docs_s3 = []
    for file in my_bucket.objects.all():
        docs_s3.append(file.key)

    # try:
    import_path = os.path.join(os.path.abspath(os.getcwd()), "ocr_doc_to_process/")
    docs_repo = os.listdir(import_path)

    final_list = docs_s3 + docs_repo

    docs_all = final_list.copy()
    # docs_all.append('All')


    l = len(docs_all)-1
    select = st.selectbox('Sélection du document', docs_all, index=l)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        all_image = st.checkbox('Tous')
    with col2:
        last_image = st.checkbox('Dernier')
    with col5:    
        button = st.button('Analyse OCR')


    if all_image and button:

        for doc in docs_s3:
            select_path = bucket_name+"/"+doc
            st.markdown(doc)
            image = read_file(select_path)
            name = doc.split('.')[0]
            str_text = extract_content_to_txt(image)
            out_file = str(name)+'.txt'
            s3.Object(bucket_name_txt, out_file).put(Body=str_text)

        for doc in docs_repo:
            select_path = import_path + doc
            st.markdown(doc)
            image = cv2.imread(select_path)
            name = doc.split('.')[0]
            str_text = extract_content_to_txt(image)
            out_file = str(name)+'.txt'
            s3.Object(bucket_name_txt, out_file).put(Body=str_text)

        # my_bar = st.progress(0)
        # schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
        # if not os.path.exists("se_indexdir"):
        #     os.mkdir("se_indexdir")

        # # Creating a index writer to add document as per schema
        # ix = create_in("se_indexdir", schema)
        # writer = ix.writer()

        # filepaths = []
        # for file in my_bucket2.objects.all():
        #     filepaths.append(file.key)

        # for name, percent in zip(filepaths, range(len(filepaths))):

        #     val = (percent+1) / len(filepaths)
        #     my_bar.progress(val)

        #     # Do not select empty document
        #     try:
        #         select_path = bucket_name_txt+"/"+name
        #         fp = fs.open(select_path, "rb")
        #         text = fp.read().decode('utf-8', 'ignore')
        #         writer.add_document(title=name, path=select_path, content=text, textdata=text)
        #         fp.close()
        #     except UnicodeDecodeError:
        #         pass

        # writer.commit()            
        ix =  ("se_indexdir")
        writer = ix.writer()
        try:
            select_path = bucket_name_txt+"/"+name
            fp = fs.open(select_path, "rb")
            text = fp.read().decode('utf-8', 'ignore')
            writer.add_document(title=name, path=select_path, content=text, textdata=text)
            fp.close()
        except UnicodeDecodeError:
            pass
        writer.commit() 
            

            ############
        if button:
    
            my_bar = st.progress(0)
            schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
            if not os.path.exists("se_indexdir"):
                os.mkdir("se_indexdir")
    
            # Creating a index writer to add document as per schema
            ix = create_in("se_indexdir", schema)
            writer = ix.writer()
    
            filepaths = []
            for file in my_bucket2.objects.all():
                filepaths.append(file.key)
    
            for name, percent in zip(filepaths, range(len(filepaths))):
    
                val = (percent+1) / len(filepaths)
                my_bar.progress(val)
    
                # Do not select empty document
                try:
                    select_path = bucket_name_txt+"/"+name
                    fp = fs.open(select_path, "rb")
                    text = fp.read().decode('utf-8', 'ignore')
                    writer.add_document(title=name, path=select_path, content=text, textdata=text)
                    fp.close()
                except UnicodeDecodeError:
                    pass
    
            writer.commit()


    elif select is not None and all_image is not None and last_image is not None and button:
        if select in docs_s3:
            select_path = bucket_name+"/"+select
            st.markdown(select)
            image = read_file(select_path)
            name = select.split('.')[0]
            st.image(image, caption=name)
            str_text = extract_content_to_txt(image)
            out_file = str(name)+'.txt'
            s3.Object(bucket_name_txt, out_file).put(Body=str_text)

        elif select in docs_repo:
            select_path = import_path + select
            st.markdown(select)
            image = cv2.imread(select_path)
            name = select.split('.')[0]
            str_text = extract_content_to_txt(image)
            out_file = str(name)+'.txt'
            s3.Object(bucket_name_txt, out_file).put(Body=str_text)
            
            ############
        if button:
    
            my_bar = st.progress(0)
            schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
            if not os.path.exists("se_indexdir"):
                os.mkdir("se_indexdir")
    
            # Creating a index writer to add document as per schema
            ix = create_in("se_indexdir", schema)
            writer = ix.writer()
    
            filepaths = []
            for file in my_bucket2.objects.all():
                filepaths.append(file.key)
    
            for name, percent in zip(filepaths, range(len(filepaths))):
    
                val = (percent+1) / len(filepaths)
                my_bar.progress(val)
    
                # Do not select empty document
                try:
                    select_path = bucket_name_txt+"/"+name
                    fp = fs.open(select_path, "rb")
                    text = fp.read().decode('utf-8', 'ignore')
                    writer.add_document(title=name, path=select_path, content=text, textdata=text)
                    fp.close()
                except UnicodeDecodeError:
                    pass
    
            writer.commit()  
            
    elif last_image and button:
        pass

#########################################################################################################################
#                                              # === INDEXER === #
# #######################################################################################################################
if analysis == "Indexation":
    st.header('Indexation')

    side_bar()

    button = st.button('Create')

    if button:

        my_bar = st.progress(0)
        schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
        if not os.path.exists("se_indexdir"):
            os.mkdir("se_indexdir")

        # Creating a index writer to add document as per schema
        ix = create_in("se_indexdir", schema)
        writer = ix.writer()

        filepaths = []
        for file in my_bucket2.objects.all():
            filepaths.append(file.key)

        for name, percent in zip(filepaths, range(len(filepaths))):

            val = (percent+1) / len(filepaths)
            my_bar.progress(val)

            # Do not select empty document
            try:
                select_path = bucket_name_txt+"/"+name
                fp = fs.open(select_path, "rb")
                text = fp.read().decode('utf-8', 'ignore')
                writer.add_document(title=name, path=select_path, content=text, textdata=text)
                fp.close()
            except UnicodeDecodeError:
                pass

        writer.commit()


# #######################################################################################################################
#                                              # === SEARCH ENGINE === #
# #######################################################################################################################
if analysis == "Search Engine":
    
    # Create the Search Engine
    try:
        SE = SearchEngine()
    except ValueError:

        my_bar = st.progress(0)
        schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, textdata=TEXT(stored=True))
        if not os.path.exists("se_indexdir"):
            os.mkdir("se_indexdir")

        # Creating a index writer to add document as per schema
        ix = create_in("se_indexdir", schema)
        writer = ix.writer()

        filepaths = []
        for file in my_bucket2.objects.all():
            filepaths.append(file.key)
        
            
        filepaths = filepaths[:10]
        for name, percent in zip(filepaths, range(len(filepaths))):

            val = (percent+1) / len(filepaths)
            my_bar.progress(val)

            # Do not select empty document
            try:
                select_path = bucket_name_txt+"/"+name
                fp = fs.open(select_path, "rb")
                text = fp.read().decode('utf-8', 'ignore')
                writer.add_document(title=name, path=select_path, content=text.lower(), textdata=text.lower())
                fp.close()
            except UnicodeDecodeError:
                pass

        writer.commit()    
        st.success("The indexer has been created")


    # Initialize variable
    languages = ['french', 'english', 'spanish', 'italian', 'german']
    is_response = False
    lang = None
    kw = doc = positive = score = None

    side_bar()

# ##################################### == PART 1 == ##########################################################
    st.header('Search Engine')
    st.subheader('Part 1 - Search')

    # Ask user for parameters
    user_input = st.text_input("Request")
    # response = st.radio('Nombre de réponses', ('La meilleure', 'Les plus pertinentes', 'Toutes'), index=1)
    response = 'All'


    lang = st.multiselect('Selected language of the documents', ['french', 'english', 'spanish', 'italian', 'german'], default=['french', 'english', 'spanish', 'italian', 'german'])

    # st.write('Paramètres a afficher')
    # kw = st.checkbox('Mots clés sélectionnés')
    # doc = st.checkbox('Nombre de document dans la base de donnée')
    # positive = st.checkbox('Nombre de résultats positifs obtenus')
    # score = st.checkbox('Score du document')
    # all_of_them = st.checkbox('Tout afficher', value=True)
    doc = True
    all_of_them = True
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
        # if kw:
        #     for ilang in lang:                      
        #             st.markdown("Les mots clés sélectionnés en **"+str(L)+"** : **"+str(tmp[ilang]['Key Words'])+"**")
        if doc:
            argmax = np.argmax(doc_found)
            try:
                st.markdown("The database contains: **"+str(tmp[lang[argmax]]['Documents to analyze'])+"** documents")
            except TypeError:
                pass
        if positive:
            st.markdown("**"+str(len(df))+"** positive results have been found in the database")

        # Crop the daframe depending on the paramaters selected
        if response == "Best":
            df = df.iloc[:1]
        elif response == "Most Pertinent":
            df = df.loc[df['Score'] > 3]
            #st.markdown("**"+str(len(df))+"** résultats positifs ont été trouvé dans la base de données")
            if len(df) > 20:
                df = df.iloc[:20]

        if score:
            pass
        else:
            df = df.drop(['Score'], axis=1)

        # Print the Dataframe
        st.dataframe(df)


        st.markdown("""---""")

 # ##################################### == PART 2 == ##########################################################
        st.subheader('Part 2 - Display')

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

                with st.expander("See original page"):
                    try:
                        bucket = my_bucket
                        image_object = bucket.Object(txt[:-9]+'.png')
                        image = mpimg.imread(BytesIO(image_object.get()['Body'].read()), 'png')
                        
                        st.image(image)
                    except (AttributeError, TypeError) as e:
                        st.warning("Erreur - l'image n'existe pas")
                        


#########################################################################################################################
#########################################################################################################################

#                                              # === END OF FILE === #

#########################################################################################################################
# #######################################################################################################################
