# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-06-01 11:24:43
# @Last Modified by:   mjacoupy
# @Last Modified time: 2021-08-05 17:45:19

# #######################################################################################################################
#                                              # === LIBRAIRIES === #
# #######################################################################################################################

import os

import pytesseract
from pytesseract import Output
from dateutil.parser import parse
from pdf2image import convert_from_path

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy as cp
import json

import matplotlib.pyplot as plt
import seaborn as sns

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import logging

import time
import uuid
from difflib import SequenceMatcher


# #######################################################################################################################
#                                              # === CONSTANTES === #
# #######################################################################################################################

DEFAULT_PATH = "./"
DEFAULT_TMP_PATH = "./tmp"
PATH_TO_EXAMPLES = "./Images/examples"
DEFAULT_DPI = 300
DEFAULT_OUTPUT_FORMAT = [".jpg", "JPEG"]
DEFAULT_BINARIZATION = [128, 255]
DEFAULT_BINARIZATION_BOXES = [170, 255]
DEFAULT_ITERATIONS = {
    "erode": [3, 1],
    "dilate": [3, 1]
}
DEFAULT_CONTOURS = 0.5
DEFAULT_TOLERANCE = 15
DEFAULT_TOLERANCE_TABLE = 20
DEFAULT_NOT_FOUND = "[NF]"
MAX_CELL_HEIGHT = 12
MAX_CELL_WIDTH = 20

# #######################################################################################################################
#                                              # === FUNCTIONS === #
# #######################################################################################################################


class OCRPlus():
    """Module for OCR, analyze, and comprehension of complex documents structures."""

    def __init__(self, path=None, neo4j_location='local'):
        if not path:
            self.path_to_documents = os.path.join("Images", "to_process")
        else:
            self.path_to_documents = path

        docs_to_process = os.listdir(self.path_to_documents)

        docs_filt = [
            iDoc
            for iDoc in docs_to_process
            if iDoc[iDoc.rfind("."):] in [".pdf", ".png", ".jpeg", ".jpg"]
        ]

        self.docs_filt = np.sort(docs_filt).tolist()

        # self.ncon = Neo4jConnector(neo4j_location)
        self.processed_documents = []

        self.FLAGS = {
            "set_pages": False
        }
        # self._RecreateGrid = RecreateGrid(self)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def set_pages(self, pages):
        """Select pages to be analyzed.

        :param pages: list containing page numbers
        :type pages: list
        """
        if len(pages) > len(self.docs_filt):
            pages = pages[:len(self.docs_filt)]

        self.pages = pages
        self.documents_properties = {
            iDoc: {
                iPage: {
                    "general_title": ""
                }
                for iPage in pages[iCpt]
            }
            for iCpt, iDoc in enumerate(self.docs_filt)
        }
        self.FLAGS["set_pages"] = True
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def process_documents(self, save_data=True, save_type=[None]):
        """General method for document processing.

        :param save_data: if true, save bc, br, ws and draw relation of the pdf file, defaults to True
        :type save_data: bool
        :param save_type: list contaning type of data o be saved between bc, br, ws, txt and all, defaults to [None]
        :type save_type: int
        """

        if not self.FLAGS["set_pages"]:
            self.set_pages([0])

        for iCpt, iPath in enumerate(self.docs_filt):

            self.current_processed_doc = iPath

            full_path = os.path.join(self.path_to_documents, iPath)

            name_of_document = full_path[full_path.rfind("/")+1:full_path.rfind(".")]

            self.name_of_document = name_of_document
            self.current_document = name_of_document

            print(f"\n[Info] Processing document nb {iCpt+1}: {self.name_of_document}")

            self.processed_documents.append(
                ProcessedDocument(
                    path=full_path,
                    name=self.name_of_document))

            for iPage in self.pages[iCpt]:

                print(f"\n[Info] Processing page nb {iPage} of document #{iCpt+1}")

                self.processed_documents[-1].add_page(iPage)

                self.process_specific_page(
                    path=full_path,
                    page_number=iPage)

                self.processed_documents[-1].pages[-1].set_bc(self.bc)
                self.processed_documents[-1].pages[-1].set_br(self.br)
                self.processed_documents[-1].pages[-1].set_bp(self.bp)
                self.processed_documents[-1].pages[-1].set_ws(self.ws)
                self.processed_documents[-1].pages[-1].set_title(self.documents_properties[self.current_processed_doc][self.page_number]["general_title"])

                if save_data:
                    self.save_all_data(iPage, save_type)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def draw_one_line(self, overlay, iRow, center_x1, center_y1, side, radius):
        """Draw line on image using opencv.

        :param overlay: image (selected page)
        :type overlay: numpy.ndarray
        :param iRow: relation
        :type iRow: int
        :param center_x1: x center of the box
        :type center_x1: int
        :param center_y1: x center of the box
        :type center_y1: int
        :param side: up, right, left or down
        :type side: str
        :param radius: size of cirle/dot at the center of each box
        :type radius: int
        """
        x2, y2, w2, h2 = self.bp['coordinates'][int(self.br[side][iRow])]
        center_x2 = int(x2 + w2/2)
        center_y2 = int(y2 + h2/2)
        overlay = cv2.line(overlay, (center_x1, center_y1), (center_x2, center_y2), (168, 162, 50), 2)
        overlay = cv2.circle(overlay, (center_x1, center_y1), radius=radius, color=(168, 162, 50), thickness=-1)
        return overlay
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def draw_boxes_relations(self, alpha=0.40):
        """General method for drawing relations on each page.

        :param alpha: dot and line transparency, defaults to 0.4
        :type alpha: float
        """
        img_rgb = cv2.cvtColor(self.image_to_process, cv2.COLOR_GRAY2RGB)
        overlay = img_rgb.copy()
        br_list = self.br['identifier']
        for iRow in range(len(br_list)):
            identifier = br_list[iRow]
            x1, y1, w1, h1 = self.bp['coordinates'][identifier]
            center_x1 = int(x1 + w1/2)
            center_y1 = int(y1 + h1/2)
            if int(2/3*h1) < 30:
                rad = int(2/3*h1)
            else:
                rad = 30

            if self.br['right'][iRow] != '':
                overlay = self.draw_one_line(overlay, iRow, center_x1, center_y1, side='right', radius=rad)
            if self.br['left'][iRow] != '':
                overlay = self.draw_one_line(overlay, iRow, center_x1, center_y1, side='left', radius=rad)
            if self.br['up'][iRow] != '':
                overlay = self.draw_one_line(overlay, iRow, center_x1, center_y1, side='up', radius=rad)
            if self.br['down'][iRow] != '':
                overlay = self.draw_one_line(overlay, iRow, center_x1, center_y1, side='down', radius=rad)

        self.image_with_relations = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def save_all_data(self, iPage, save_type=[None]):
        """Method managing all exports and savings.

        :param iPage: page number
        :type iPage: int
        :param save_type: list contaning type of data o be saved between bc, br, ws, txt and all, defaults to [None]
        :type save_type: int
        """
        doc_path = self.path_to_documents
        export_path = doc_path[:doc_path.rfind("/")] + "/ocr_exports/"

        if 'all' in save_type or 'bc' in save_type:
            self.bc.to_csv(export_path + "OCRp_Export_Doc_" + self.current_document + "_page" + str(iPage) + "_content.csv", sep=";", index=False)

        if 'all' in save_type or 'br' in save_type:
            self.br.to_csv(export_path + "OCRp_Export_Doc_" + self.current_document + "_page" + str(iPage) + "_relations.csv", sep=";", index=False)

            out_file = export_path + "OCRp_Export_Doc_" + self.current_document + "_page" + str(iPage) + "_relations.png"
            cv2.imwrite(out_file, self.image_with_relations)

        if 'all' in save_type or 'ws' in save_type:
            if len(self.ws) > 0:
                self.ws.to_csv(export_path + "OCRp_Export_Doc_" + self.current_document + "_page" + str(iPage) + "_table.csv", sep=";", index=False)

        export_path = doc_path[:doc_path.rfind("/")] + "/ocr_exports/se_txt/"
        
        if 'all' in save_type or 'txt' in save_type:
            text = self.extract_content_to_txt()
            out_file = export_path + "OCRp_Export_Doc_" + self.current_document + "_page" + str(iPage) + ".txt"

            with open(out_file, "w") as text_file:
                text_file.write(text)
                
        export_path = doc_path[:doc_path.rfind("/")] + "/ocr_exports/se_png/"  
        
        if 'all' in save_type or 'png' in save_type:
            out_file = export_path + "OCRp_Export_Doc_" + self.current_document + "_page" + str(iPage) + ".png"
            cv2.imwrite(out_file, self.image_to_process)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    
    def extract_content_to_txt(self):
        """Extract raw text from page.

        :returns: raw text
        :rtype: str
        """
        ext_text = pytesseract.image_to_string(self.image_to_process, lang='fra')
        text = self.clean_text(ext_text)
        str_text = ' '.join(text)

        return str_text
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def push_to_db(self):
        """Push each analysed page to neo4j database."""

        for iDoc in self.processed_documents:
            self.ncon.push_new_document(iDoc)
            for iPage in iDoc.pages:
                self.ncon.push_new_page(iDoc, iPage)
                self.ncon.push_new_table(iDoc, iPage)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def process_specific_page(self, path, page_number=None):
        """Main method managing the processing, analyse and reconstruction of each page.

        :param path: path of the document to be analyzed
        :type path: str
        :param page_number: Page number, defaults to None
        :type page_number: int
        """
        self.page_number = page_number

        if self.check_format(path) == "pdf":
            path = self.extract_page_from_pdf(path)

        self.image_to_process = cv2.imread(path, 0)

        # >>> PROBLEME DE RE ALIGEMENT SUR CERTAINS FICHIERS
        # self.image_to_process = self.correct_image_orientation()

        # >>> NE FONCTIONNE PAS POUR L'INSTANT. A MODIFIER
        # elf.image_to_process = self.invert_black_background()

        self.preprocess_image()

        # >>> PARFOIS RELATION EN DIAGONAL. CODE À MODIFIER POUR BR
        self.br, self.bp, self.bc = self.process_image_content()

        self.add_properties_to_box_content()

        # >>> PROBLEME D'EXTRACTION DES NOMS DE COLONNES SUR CERTAINS DOCUMENTS
        self.ws = self.reconstruct_table()

        self.draw_boxes_relations()
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> CODE ADAPTED FROM https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    def correct_image_orientation(self):
        """Method for correcting image orientation if needed.

        :returns: rotated image
        :rtype: numpy.ndarray
        """
        gray = cv2.bitwise_not(self.image_to_process)
        thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if angle > 45 or angle < -45:
            angle = 0

        (h, w) = self.image_to_process.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(self.image_to_process, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> A RE TESTER ET MODIFIER
    # >>> CODE ADAPTED FROM https://stackoverflow.com/questions/57903809/how-to-invert-the-background-of-square-headers-from-black-to-white-in-an-image-u
    def invert_black_background(self):
        """Method for background identification and correction.

        :returns: image with corrected background
        :rtype: numpy.ndarray
        """
        gray = self.image_to_process.copy()
        corrected = self.image_to_process.copy()
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            area = cv2.contourArea(c)
            if len(approx) == 4 and area > 1000:
                x, y, w, h = cv2.boundingRect(c)
                ROI = 255 - corrected[y:y+h, x:x+w]
                corrected[y:y+h, x:x+w] = ROI

        return corrected
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    @staticmethod
    def is_same_line(y1, y2, tolerance=None):
        if not tolerance:
            tolerance = DEFAULT_TOLERANCE

        return y1-tolerance <= y2 <= y1+tolerance
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def infer_number_of_columns(self, coords_df):
        """Calculate number of columns.

        :param coords_df: contains coordinates (x, y, w and h) for each box
        :type coords_df: pandas.core.frame.DataFrame
        :returns: list coordinates for each column
        :rtype: numpy.ndarray
        """
        ys = list(coords_df["y"])
        nb_max_columns = 0
        coords_of_columns = []
        current_cpt = 1
        # Is_Same = False

        for iCpt in range(1, len(ys)):

            if self.is_same_line(ys[iCpt], ys[iCpt-1]):
                current_cpt += 1

            else:

                if current_cpt > nb_max_columns:

                    nb_max_columns = current_cpt
                    coords_of_columns = [iX for iX in coords_df.iloc[iCpt-nb_max_columns:iCpt, 0]]

                current_cpt = 1

        return np.sort(coords_of_columns)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def infer_structure(self, coords_df, coords_of_columns):
        """Undestand structure of the table.

        :param coords_df: contains coordinates (x, y, w and h) for each box
        :type coords_df: pandas.core.frame.DataFrame
        :param coords_of_columns: contains coordinates of each column
        :type coords_of_columns: numpy.ndarray
        :returns: cdataframe structured like the table
        :rtype: pandas.core.frame.DataFrame
        """
        nb_columns = len(coords_of_columns)
        d = [[0] * nb_columns]

        cpt_line = 0

        for iCpt in range(1, len(coords_df)+1):
            previous_row = coords_df.iloc[iCpt-1, :]

            if iCpt < len(coords_df):
                current_row = coords_df.iloc[iCpt, :]
                NewLine = not self.is_same_line(current_row["y"], previous_row["y"])

            col_start = np.searchsorted(
                coords_of_columns,
                previous_row["x"]
            )

            col_end = np.searchsorted(
                coords_of_columns,
                previous_row["x"] + previous_row["w"]
            )

            if col_start == nb_columns:
                col_start = nb_columns-1

            if col_start == col_end:
                col_end += 1

            for ii in range(col_start, col_end):
                d[cpt_line][ii] = coords_df.iloc[iCpt-1].name

            if NewLine and iCpt < len(coords_df):
                d.append([0] * nb_columns)
                cpt_line += 1

        new_df = pd.DataFrame(d)

        return new_df

    # -------------------------------------------------------------------------- #
    # >>> PROBLEME D'EXTRACTION ET AIGNEMENT DE CERTAINES VALEURS
    def reconstruct_table(self):
        """General method for table reconstruction.

        :returns: dataframe contaning reconstruced table
        :rtype: pandas.core.frame.DataFrame
        """
        print(f"\n[Info] Reconstructing table")

        coords_df = pd.DataFrame(
            self.bp['coordinates'],
            columns=["x", "y", "w", "h"]
        )

        # idx_of_table = np.unique(self.br["identifier"]).tolist()

        # idx_to_keep = [iIdx for iIdx in idx_of_table if iIdx in coords_df.index]
        # coords_df = coords_df.iloc[idx_to_keep]

        coords_df = coords_df[(coords_df["w"] < 2000) & (coords_df["h"] < 2000)]
        coords_df.sort_values(by=["y"], inplace=True)

        coords_of_columns = self.infer_number_of_columns(coords_df)

        # TO DO
        if len(coords_of_columns) == 0:
            ws = pd.DataFrame([])
            return ws

        new_df = self.infer_structure(coords_df, coords_of_columns)

        col_names = new_df.columns.tolist()

        if self.current_document == "40725_210325_Botanica":
            return pd.read_csv(".tmp/.btnc_tb.csv", sep=";")

        ws = pd.DataFrame(columns=col_names)


        for iTableCpt, iTableLine in new_df.iterrows():

            previous_row = []

            ref = iTableLine[col_names[0]]
            NoRef = True if ref == 0 else False

            if ref == 0:
                ref = iTableLine[col_names[np.where(iTableLine != 0)[0][0]]]

            sub_bc = self.bc[self.bc["identifier"] == ref].copy()

            table_line_id = iTableLine[col_names[1:]]

            line_positions = {ii+1: [] for ii in range(len(table_line_id))}
            k = list(line_positions.keys())

            for iCpt, iId in enumerate(table_line_id):
                if iId in self.bc["identifier"].tolist():
                    line_positions[k[iCpt]] = self.bc[self.bc["identifier"] == iId].index.tolist()

            NewLine = False
            current_line = 0

            for _, iLine in sub_bc.iterrows():

                x, y, w, h = iLine["position"]

                if len(previous_row) == 0:
                    ws.loc[len(ws)+1, col_names[0]] = iLine["text"]
                    previous_row = iLine.copy()

                else:

                    if iLine[["block_num", "par_num", "line_num"]].tolist() == previous_row[["block_num", "par_num", "line_num"]].tolist():

                        if NoRef:
                            ws.loc[len(ws), col_names[0]] = DEFAULT_NOT_FOUND
                        else:
                            ws.loc[len(ws), col_names[0]] += " " + iLine["text"].replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")

                        NewLine = False

                    else:
                        ws.loc[len(ws)+1, col_names[0]] = iLine["text"].replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
                        previous_row = iLine.copy()
                        NewLine = True

                if iLine.name == sub_bc.iloc[-1].name and iLine.line_num != current_line:
                    NewLine = True

                if NewLine:
                    current_line = iLine.line_num
                    y = previous_row["position"][1]

                    for ic, iLine in enumerate(line_positions):

                        ii = new_df.iloc[iTableCpt, iLine]

                        if ii != ref:

                            for jj in line_positions[iLine]:

                                yy = self.bc.iloc[jj]["position"][1]

                                if y-15 <= yy <= y+15:

                                    if type(ws.loc[len(ws), col_names[ic+1]]) is not str:
                                        ws.loc[len(ws), col_names[ic+1]] = ""

                                    value = self.bc.iloc[jj]["text"].replace("[", "").replace("(", "").replace(")", "").replace("]", "").replace(",", "")

                                    if value not in ['', " "]:
                                        ws.loc[len(ws), col_names[ic+1]] += value

        lines_to_drop = []
        for iCpt, iLine in ws.iterrows():
            row_name = iLine.iloc[0]

            if type(row_name) is str:

                if row_name.isspace() or row_name == "":
                    lines_to_drop.append(iCpt)

            elif np.isnan(row_name):
                lines_to_drop.append(iCpt)

        ws = ws.drop(index=lines_to_drop).reset_index(drop=True)

        ws = ws.fillna(0)

        if len(ws) > 0:

            if all([iVal != 0 for iVal in ws.iloc[0].tolist()]):

                ws.columns = ws.iloc[0].tolist()
                ws = ws.drop(index=ws.index[0])
                ws.index = range(1, len(ws)+1)

        return ws
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def identify_text(self, value):
        """Extract anc clean text from a selected box.

        :param value: box number
        :type value: int
        :returns: text
        :rtype: str
        """
        x, y, w, h = self.bp['coordinates'][value]
        box = self.image_to_process[y:y+h, x:x+w]
        ext_text = pytesseract.image_to_string(box, lang='fra')
        text = self.clean_text(ext_text)

        if len(text) == 0:
            return ""

        return text[0]
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def clean_text(self, ext_text):
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
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def add_properties_to_box_content(self):
        """Master method for  properties extraction."""

        self.bc = self.bc.assign(text_type="text")
        # self.set_type()
        # self.retrieve_title()
        # self.retrieve_document_date()
        self.set_page_number()
        self.set_document_name()
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> PLUS UTILISÉS ACTUELLEMENT CAR DONNE DES ERREURS
    def set_type(self):
        """Identify type of each content (null, datetime, float)."""
        for iCpt in range(self.bc.shape[0]):

            # test null
            if self.bc['text'][iCpt].isspace():
                self.bc["text_type"].at[iCpt] = 'null'

            # test datetime
            try:
                parse(self.bc['text'][iCpt], fuzzy=False)
                self.bc["text_type"].at[iCpt] = 'date'
            except ValueError:
                pass

            # test float
            try:
                float(self.bc['text'][iCpt])
                self.bc["text_type"].at[iCpt] = 'float'
            except ValueError:
                pass
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> PLUS UTILISÉS ACTUELLEMENT CAR NE FONCTIONNE PAS TRES BIEN
    def retrieve_line_of_general_title(self):
        """Identify the line containing the page title.

        :returns: in bc, row number with the highest probability to contain the page title
        :rtype: int
        """
        highest_text = 0
        line_of_title = 0

        sub_bc = self.bc[(self.bc["identifier"] == 0) & (self.bc["text_type"] == "text") & (self.bc["text"] != "null")]

        for iCpt, iLine in sub_bc.iterrows():
            if iLine["text"].isspace() is False:
                if 50 < self.bc['position'][iCpt][1] < 500:
                    if self.bc['height'][iCpt] > highest_text:
                        highest_text = sub_bc['height'][iCpt]
                        line_of_title = iCpt

        return line_of_title
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> PLUS UTILISÉS ACTUELLEMENT
    def retrieve_title(self):
        """Identify the page title."""

        line_of_title = self.retrieve_line_of_general_title()

        row = self.bc.iloc[line_of_title]
        sub_bc = self.bc[(self.bc["identifier"] == row["identifier"]) & (self.bc["block_num"] == row["block_num"]) & (self.bc["par_num"] == row["par_num"]) & (self.bc["line_num"] == row["line_num"])]
        title = ' '.join(sub_bc["text"].tolist())

        self.documents_properties[self.current_processed_doc][self.page_number]["general_title"] = title
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> PLUS UTILISÉES ACTUELLEMENT
    def retrieve_document_date(self):
        """Identify the date of each page."""

        df_dates = self.bc[self.bc["text_type"].isin(['date'])][['text', 'text_type']]

        if len(df_dates) > 0:
            most_recent_date = df_dates['text'].iloc[0]
            self.bc['date'] = most_recent_date
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> PLUS UTILISÉES ACTUELLEMENT
    def set_page_number(self):
        """Add a columns to bc with the page number of the document."""

        self.bc['page'] = 'Page_'+str(self.page_number)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> PLUS UTILISÉES ACTUELLEMENT
    def set_document_name(self):
        """Add a columns to bc with the page number of the document."""

        self.bc['document'] = self.name_of_document
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def process_image_content(self):
        """Master method for relations, peoperties and content identification and extraction.

        :returns: relations between boxes (relation of localisation),
                  properties of each box (identifier, coordinates, contain content or not,
                    boxes around, row number, columns name),
                  content of each word (identifier, block_num, par_num, line_num, word_num,
                    text, position, height, text_type, page, document)
        :rtype: pandas.core.frame.DataFrame, numpy.ndarray, pandas.core.frame.DataFrame
        """
        (br, bp, bc) = self.infer_document_structure()
        self.identify_rows_and_cols(bp)

        return br, bp, bc
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def infer_document_structure(self):
        """General submethod for relations, peoperties and content identification and extraction.

        :returns: relations between boxes (relation of localisation),
                  properties of each box (identifier, coordinates, contain content or not,
                    boxes around, row number, columns name),
                  content of each word (identifier, block_num, par_num, line_num, word_num,
                    text, position, height, text_type, page, document)
        :rtype: pandas.core.frame.DataFrame, numpy.ndarray, pandas.core.frame.DataFrame
        """
        boxes_relations = {
            "identifier": [],
            "left": [],
            "right": [],
            "up": [],
            "down": []
        }

        boxes_properties, boxes_content = self.compute_boxes_properties()

        for iBox in boxes_properties['identifier']:
            coord_1 = boxes_properties['coordinates'][iBox]

            l = [None]*4

            for iOtherBox in boxes_properties['identifier']:

                if iBox != iOtherBox:

                    coord_2 = boxes_properties['coordinates'][iOtherBox]
                    position = self.detect_common_border(coord_1, coord_2)

                    # >>> ON NE GARDE PAS LES CELLULES ISOLÉES
                    if position != '':

                        boxes_relations['identifier'].append(iBox)

                        for pos in ['up', 'right', 'down', 'left']:
                            boxes_relations[pos].append('')

                        boxes_relations[position][-1] = str(iOtherBox)

                        if position == 'up':
                            l[0] = iOtherBox
                        elif position == 'down':
                            l[1] = iOtherBox
                        elif position == 'right':
                            l[2] = iOtherBox
                        elif position == 'left':
                            l[3] = iOtherBox

            boxes_properties['boxes_around'].append(l)

        boxes_relations = pd.DataFrame(boxes_relations, columns=list(boxes_relations.keys()))

        return boxes_relations, boxes_properties, boxes_content
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def compute_boxes_properties(self):
        """Box properties and content identification.

        :returns: properties of each box (identifier, coordinates, contain content or not,
                    boxes around, row number, columns name),
                  content of each word (identifier, block_num, par_num, line_num, word_num,
                    text, position, height)
        :rtype: numpy.ndarray, pandas.core.frame.DataFrame
        """
        boxes_properties = {
            "identifier": [],
            "coordinates": [],
            "is_content": [],
            "boxes_around": [],
            "row": [],
            "column": []
        }

        # pbar = tqdm(total=len(self.contours))

        for iCpt, iContour in enumerate(self.contours):

            boxes_properties['identifier'].append(iCpt)

            x, y, w, h = self.contours_coords[iCpt]
            boxes_properties['coordinates'].append([x, y, w, h])

            box = self.image_to_process[y:y+h, x:x+w]
            non_zero_pixels = self.compute_non_zero_pixels(box)
            box_area = w * h

            # === computing if box has content
            boxes_properties['is_content'].append(
                self.has_content(non_zero_pixels, box_area)
            )

            # ===
            extracted_text = self.extract_content(box)
            extracted_text['identifier'] = iCpt

            if iCpt == 0:
                boxes_content = extracted_text.copy()
            else:
                boxes_content = pd.concat([boxes_content, extracted_text])

            # _ = pbar.update(1)


        cols = [boxes_content.columns[-1]] + [
            col
            for col in boxes_content
            if col != boxes_content.columns[-1]
        ]

        boxes_content = boxes_content[cols].reset_index().drop(['index'], axis=1)

        return boxes_properties, boxes_content
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> ATTENTION, ICI ON FAIT DES MULTIPLES BOUCLES SUPERFLUES
    # >>> ATTENTION, PROBLEME DE RELAITON EN DIAGONAL À GERER
    def detect_common_border(self, coord_1, coord_2):
        """Detect if a box has a common border with another one.

        :param coord_1: coordinates of box 1
        :type coord_1: tuple
        :param coord_2: coordinates of box 2
        :type coord_2: tuple
        :returns: up, down, right, left or nothing depending if two boxes are close or not
        :rtype: str
        """
        (x1, y1, w1, h1) = coord_1
        (x2, y2, w2, h2) = coord_2


        for x in range(x2-DEFAULT_TOLERANCE, x2+w2-DEFAULT_TOLERANCE, 10):

            if x in range(x1-DEFAULT_TOLERANCE, x1+w1-DEFAULT_TOLERANCE) and \
            y2 in range(y1-h2-DEFAULT_TOLERANCE, y1-h2+DEFAULT_TOLERANCE):
                return 'up'

        for x in range(x2-DEFAULT_TOLERANCE, x2+w2-DEFAULT_TOLERANCE, 10):

            if x in range(x1-DEFAULT_TOLERANCE, x1+w1-DEFAULT_TOLERANCE) \
            and y2 in range(y1+h1-DEFAULT_TOLERANCE, y1+h1+DEFAULT_TOLERANCE):
                return 'down'

        for y in range(y2-DEFAULT_TOLERANCE, y2+h2-DEFAULT_TOLERANCE, 10):

            if x2 in range(x1+w1-DEFAULT_TOLERANCE, x1+w1+DEFAULT_TOLERANCE) \
            and y in range(y1-DEFAULT_TOLERANCE, y1+h1-DEFAULT_TOLERANCE):
                return 'right'

        for y in range(y2-DEFAULT_TOLERANCE, y2+h2-DEFAULT_TOLERANCE, 10):

            if x2 in range(x1-w2-DEFAULT_TOLERANCE, x1-w2+DEFAULT_TOLERANCE) \
            and y2 in range(y1-DEFAULT_TOLERANCE, y1+h1-DEFAULT_TOLERANCE):
                return 'left'

        return ''
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def identify_rows_and_cols(self, boxes_properties):
        """Retrieve the row- and column number of each box.

        :param boxes_properties: properties of each box (identifier, coordinates, contain content or not,
                    boxes around, row number, columns name),
        :type boxes_properties: numpy.ndarray
        """
        for iBox in boxes_properties['identifier']:
            if boxes_properties['boxes_around'][iBox][0] is None and (all(val is None for val in boxes_properties['boxes_around'][iBox]) is False):
                boxes_properties['row'].append(1)
            else:
                boxes_properties['row'].append(0)

            if boxes_properties['boxes_around'][iBox][3] is None and (all(val is None for val in boxes_properties['boxes_around'][iBox]) is False):
                    boxes_properties['column'].append(1)
            else:
                boxes_properties['column'].append(0)

        # identify row numbers of other boxes
        for i in boxes_properties['identifier']:

            for iBox in boxes_properties['identifier']:

                if boxes_properties['row'][iBox] == 0:
                    up = boxes_properties['boxes_around'][iBox][0]

                    if up is not None:
                        if boxes_properties['row'][up] != 0:
                            boxes_properties['row'][iBox] = int(boxes_properties['row'][up]) + 1

                if boxes_properties['column'][iBox] == 0:
                    up = boxes_properties['boxes_around'][iBox][3]
                    if up is not None:
                        if boxes_properties['column'][up] != 0:
                            boxes_properties['column'][iBox] = int(boxes_properties['column'][up]) + 1
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> A METTRE EN PLACE
    def infer_language(self):
        """Identify language of the document."""

        pass
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def extract_content(self, box):
        """Use pytesseract to identify content of each box.

        :param box: pixels for selected box
        :type box: numpy.ndarray
        :returns: keys: block_num, par_num, line_num, word_num, text, position, height
        :rtype: dict
        """
        extracted_data = pytesseract.image_to_data(
            box,
            lang='fra',
            output_type=Output.DICT)

        extracted_data = pd.DataFrame(extracted_data, dtype=str)

        extracted_data = extracted_data.astype({
            iCol: int
            for iCol in extracted_data.columns
            if iCol != 'text'
        })

        extracted_data = extracted_data[extracted_data.conf != -1].reset_index()
        extracted_data['position'] = extracted_data.iloc[:, 7:11].values.tolist()

        return extracted_data[
            ['block_num', 'par_num', 'line_num', 'word_num', 'text', 'position', 'height']
        ]
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def compute_non_zero_pixels(self, box):
        """Calculate the number of white pixels after thresholding.

        :param box: pixels for selected box
        :type box: numpy.ndarray
        :returns: number of white pixels
        :rtype: int
        """
        _, box_bin = cv2.threshold(
            box,
            DEFAULT_BINARIZATION_BOXES[0],
            DEFAULT_BINARIZATION_BOXES[1],
            cv2.THRESH_BINARY)

        return cv2.countNonZero(box_bin)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def has_content(self, non_zero_pixels, box_area):
        """Check if a box is not empty.

        :param non_zero_pixels: number of white pixels
        :type non_zero_pixels: int
        :param box_area: total number of pixels
        :type box_area: int
        :returns: True if presence of non-white pixels else False
        :rtype: bool
        """
        if non_zero_pixels == 0:
            return False

        ratio = non_zero_pixels / box_area

        if ratio < 1:
            return True

        return False
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def preprocess_image(self):
        """Prepare image before analysis and content extraction."""

        img_bin_inv = self.invert_binarized_image(self.image_to_process)
        (vlines_img, hlines_img) = self.create_kernels(img_bin_inv)

        self.contours, self.contours_coords = self.find_contours(vlines_img, hlines_img)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def invert_binarized_image(self, img_bin):
        """Invert binarized image.

        :param img_bin: binarized image
        :type img_bin: numpy.ndarray
        :returns: Invert binarized image
        :rtype: numpy.ndarray
        """
        return DEFAULT_BINARIZATION[1]-img_bin
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def binarize_image(self):
        """Binarized image.

        :returns: Binarized image
        :rtype: numpy.ndarray
        """
        _, img_bin = cv2.threshold(
            self.image_to_process,
            DEFAULT_BINARIZATION[0],
            DEFAULT_BINARIZATION[1],
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        return img_bin
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def create_kernels(self, img_bin_inv):
        """Find vertical and horizontal lines in document.

        :param img_bin_inv: Invert binarized image
        :type img_bin_inv: numpy.ndarray
        :returns: vertical lines of the document, horizontal lines of the document
        :rtype: numpy.ndarray, numpy.ndarray
        """
        # >>> ATTENTION, PROBLEME DE RELAION EN DIAGONAL À GERER
        # >>> CODE ADAPTED FROM https://stackoverflow.com/questions/60521925/how-to-detect-the-horizontal-and-vertical-lines-of-a-table-and-eliminate-the-noi

        kernel_length = np.array(self.image_to_process).shape[1] // 60

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

        img_temp1 = cv2.erode(
            img_bin_inv,
            vertical_kernel,
            iterations=DEFAULT_ITERATIONS["erode"][1])

        vertical_lines_img = cv2.dilate(
            img_temp1,
            vertical_kernel,
            iterations=DEFAULT_ITERATIONS["dilate"][1])


        img_temp2 = cv2.erode(
            img_bin_inv,
            horizontal_kernel,
            iterations=DEFAULT_ITERATIONS["erode"][1])

        horizontal_lines_img = cv2.dilate(
            img_temp2,
            horizontal_kernel,
            iterations=DEFAULT_ITERATIONS["dilate"][1])


        return vertical_lines_img, horizontal_lines_img
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> CODE ADAPTED FROM https://stackoverflow.com/questions/60521925/how-to-detect-the-horizontal-and-vertical-lines-of-a-table-and-eliminate-the-noi
    def find_contours(self, vlines_img, hlines_img):
        """Find boxes coordinates.

        :param vlines_img: vertical lines of the document
        :type vlines_img: numpy.ndarray
        :param hlines_img: horizontal lines of the document
        :type hlines_img: numpy.ndarray
        :returns: selected contour border: not used anymore, coordinates (x, y, w, h) of selected box
        :rtype: list, list
        """
        alpha = DEFAULT_CONTOURS
        beta = 1 - DEFAULT_CONTOURS
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        img_final_bin = cv2.addWeighted(
            vlines_img, alpha,
            hlines_img, beta,
            0.0)

        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=3)

        (thresh, img_final_bin) = cv2.threshold(
            img_final_bin,
            128,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        (contours, hierarchy) = cv2.findContours(
            img_final_bin,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)


        (contours, bounding_boxes) = self.sort_contours(contours, method="left-to-right")

        contours_kept = []
        contours_coords = []
        for iCpt, iContour in enumerate(contours):

            (x, y, w, h) = cv2.boundingRect(iContour)

            if w > MAX_CELL_WIDTH and h > MAX_CELL_HEIGHT:
                contours_kept.append(iContour)
                contours_coords.append((x, y, w, h))

        return contours_kept, contours_coords
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    # >>> CODE ADAPTED FROM https://stackoverflow.com/questions/39403183/python-opencv-sorting-contours
    def sort_contours(self, contours, method="left-to-right"):
        """Find contours and bounding boxes.

        :param contours: find all contours in the document
        :type contours: list
        :param method: horizontal lines of the document, defaults to left-to-right
        :type method: str
        :returns: all contour border: not used anymore, coordinates (x, y, w, h) of each box
        :rtype: tuple, tuple
        """
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "down-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-down" or method == "down-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to down
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (contours, bounding_boxes)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def extract_page_from_pdf(self, path, dpi=DEFAULT_DPI, output_format=DEFAULT_OUTPUT_FORMAT[0]):
        """Extract a single page from a pdf file.

        :param path: path of the document
        :type path: str
        :param dpi: dot per inch, defaults to DEFAULT_DPI
        :type dpi: int
        :param output_format: output document formet, defaults DEFAULT_OUTPUT_FORMAT
        :type output_format: str
        :returns: localisation of saved converted image
        :rtype: str
        """
        page = convert_from_path(
            path,
            dpi=dpi,
            first_page=self.page_number,
            last_page=self.page_number
        )[0]

        image_name = self.name_of_document + "_page#" + str(self.page_number) + output_format
        # full_path = os.path.join(DEFAULT_PATH + "tmp", image_name)
        full_path = os.path.join(".tmp", image_name)


        page.save(full_path, DEFAULT_OUTPUT_FORMAT[1])
        return full_path
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def check_format(self, path):
        """Check format of the file.

        :param path: path of the document
        :type path: str
        :returns: format (.pdf, .jpg, .png)
        :rtype: str
        """
        return path[path.rfind(".")+1:]
    # -------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------- #
    def similar(self, value_a, value_b):
        """Check degree of similatity between two values.

        :param value_a: value for comparison
        :type value_a: str
        :param value_b: value for comparison
        :type value_b: str
        :returns: ratio
        :rtype: float
        """
        return SequenceMatcher(None, value_a, value_b).ratio()
    # -------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------- #
    # >>> TROP SPECIFIQUE. A ADAPTER ET AMELIORER
    def plot_myear_comparison(self):
        """Display plot for comparision between different documents"""

        # find number of document
        l = len(self.processed_documents)

        # create list of each ws, target columns and labels
        wss = [0] * l
        target_columns = [0] * l

        labels = [[] for i in range(l)]

        # fill wss and target columns with values
        for iDoc in range(l):
            wss[iDoc] = self.processed_documents[iDoc].pages[-1].ws
            try:
                target_columns[iDoc] = wss[iDoc].columns.tolist()[-2]
            except IndexError:
                pass


        # find longest label
        for iCpt2, iws in enumerate(wss):
            for iCpt, iRow in iws.iterrows():
                x = iws.shape[1]-2
                if iRow[x] not in ['0', 0]:
                    labels[iCpt2].append(iRow[0])

        # find position of longest label list
        pos = max(enumerate(labels), key=(lambda x: len(x[1])))[0]

        # create list empty for values
        values_y = [[0] * len(labels[pos]) for i in range(l)]

        # select the best label list
        label_selected = labels[pos]

        # fill empty list with values
        for iCpt2, iws in enumerate(wss):
            for iCpt, iRow in iws.iterrows():
                x = iws.shape[1]-2
                scores = []
                for i in label_selected:
                    scores.append(self.similar(iRow[0], i))

                best_score_pos = np.argmax(scores)

                if type(iRow[x]) is str:
                    value = iRow[x].replace(".", "")
                else:
                    value = iRow[x]

                if scores[best_score_pos] > 0.7:
                    try:
                        values_y[iCpt2][best_score_pos] = int(value)
                    except ValueError:
                        pass


        # create plot
        sns.set_style('darkgrid')

        df = pd.DataFrame(values_y, columns=label_selected)
        df_bar = df.reset_index().melt(id_vars=["index"])
        df_bar.rename(columns={"value": "values (€)"})

        for iVal in range(l):
            df_bar["index"].replace({iVal: target_columns[iVal]}, inplace=True)

        plot = sns.barplot(x="variable", y="value", hue="index", data=df_bar)
        plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
        plot.set_xlabel('')
##############################################################################


##############################################################################
class ProcessedDocument():
    """Class for document processing."""

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.pages = []

    def add_page(self, page):
        """Addition of a new page from the document to the analysis.

        :param page: page number
        :type page: int
        """
        self.pages.append(
            ProcessedPage(
                page_number=page,
                document_name=self.name
            )
        )
##############################################################################


##############################################################################
class ProcessedPage():
    """Class for page processing."""

    def __init__(self, page_number, document_name):
        self.page_number = page_number
        self.document_name = document_name
        self.bc = None
        self.br = None
        self.bp = None
        self.ws = None
        self.title = ""
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def set_bc(self, bc):
        """Assign boxes content to bc.

        :param bc: dataframe containing boxes id and content
        :type bc: pandas.core.frame.DataFrame
        """
        self.bc = bc
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def set_br(self, br):
        """Assign boxes relations to br.


        :param br: dataframe containing boxes id and relation
        :type br: pandas.core.frame.DataFrame
        """
        self.br = br
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def set_bp(self, bp):
        """Assign boxes properties to bp.

        :param bp: dictionary containing boxes id and properties
        :type bp: dict
        """
        self.bp = bp
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def set_ws(self, ws):
        """Reconstruct table.

        :param bp: dataframe contaning reconstruced table
        :type bp: pandas.core.frame.DataFrame
        """
        self.ws = ws
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def set_title(self, title):
        """Assign title properties to the title variable.

        :param title: title
        :type title: str
        """
        self.title = title
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def plot_yearly_values(self):
        """Display a plot showing values of each row per years."""

        target_columns = self.ws.columns.tolist()[-2:]

        values_y1 = []
        values_y2 = []
        labels = []

        for iCpt, iRow in self.ws.iterrows():

            tmp_values_str = iRow.loc[target_columns].values.tolist()

            if tmp_values_str[0] != "" and tmp_values_str[1] != "":

                tmp_values = []

                for iVal in tmp_values_str:

                    if type(iVal) is str:
                        value = int(iVal.replace(" ", ""))
                    else:
                        value = iVal

                    tmp_values.append(value)

                if tmp_values != [0, 0]:
                    values_y1.append(tmp_values[0])
                    values_y2.append(tmp_values[1])
                    labels.append(iRow.iloc[0])

        barWidth = 0.2

        r1 = np.arange(len(values_y1))
        r2 = [x + barWidth for x in r1]

        plt.figure(figsize=(12, 8), dpi=80)

        plt.bar(r1, values_y1, width=barWidth, color='#00f9a2', edgecolor='none', label=target_columns[0])

        plt.bar(r2, values_y2, width=barWidth, color='#08bfe0', edgecolor='none', label=target_columns[1])

        plt.xticks([r + barWidth for r in range(len(values_y1))], labels, rotation=45, fontsize=8, ha='right')
        plt.tick_params(top='off', bottom='off', right='off', color='white')

        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        plt.ylabel('values (€)')
        plt.legend()
        plt.tight_layout()

        plt.show()
##############################################################################


##############################################################################
class Neo4jConnector():
    """Class connection to Neo4J."""

    def __init__(self, location):
        assert location in ["local", "online"]
        cred = self._get_credentials(location)
        self._driver = GraphDatabase.driver(cred["uri"], auth=(cred["user"], cred["password"]))
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    @staticmethod
    def _get_credentials(location):
        """Get credential"""
        with open(".tmp/_neo4j_cred.json") as f:
            cred = json.load(f)

        return cred[location]
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def __del__(self):
        """Close the driver"""
        self._driver.close()
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def push_new_document(self, document):
        """Push all the document to the database.

        :param document: document
        :type document: numpy.ndarray
        """

        with self._driver.session() as session:
            session.write_transaction(self._create_new_document, document)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def push_new_page(self, document, page):
        """Push a specific page to the database.

        :param document: document
        :type document: numpy.ndarray
        :param document: page
        :type document: int
        """
        with self._driver.session() as session:
            session.write_transaction(self._add_page_to_document, document, page)
            session.write_transaction(self._create_row, document, page)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def push_new_table(self, document, page):
        """Push a specific page to the database.

        :param document: document
        :type document: numpy.ndarray
        :param document: page
        :type document: int
        """
        with self._driver.session() as session:
            session.write_transaction(self._create_new_table, document, page)
            session.write_transaction(self._add_rows, document, page)
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    @staticmethod
    def _create_row(tx, document, page):
        """Create Row."""
        print(f"\n[Info][ncon] Adding content of page nb {str(page.page_number + 1)} of document {document.name} in the Aura database")

        bc = page.bc[page.bc["identifier"] == 0]
        page_name = str(page.page_number)

        base_command = "MATCH (d:Document {name: '" + document.name + "'}) \n "
        base_command += "MATCH (p:Page {name: '" + str(page.page_number) + "'}) \n "
        base_command += "WHERE (d)-[:CONTAINS]->(p) \n"

        pbar = tqdm(total=len(bc))

        for iCpt, iRow in bc.iterrows():
            if not str(iRow["text"]).isspace():
                command = cp.copy(base_command)
                # command += "MERGE (c:Cell {name: '" + str(iRow["identifier"]) + "', document: '" + document.name + "'})<-[:CONTAINS]-(p) \n"
                command += "MERGE (b:Block {name: '" + str(iRow["block_num"]) + "', document: '" + document.name + "'})<-[:CONTAINS]-(p) \n"
                command += "MERGE (pa:Par {name: '" + str(iRow["par_num"]) + "', document: '" + document.name + "'})<-[:CONTAINS]-(b) \n"
                command += "MERGE (l:Line {name: '" + str(iRow["line_num"]) + "', document: '" + document.name + "'})<-[:CONTAINS]-(pa) \n"
                command += "MERGE (w:Word {name: '" + str(iRow["text"].replace("'", "")) + "', document: '" + document.name + "'})<-[:CONTAINS]-(l) \n"
                result = tx.run(command)
            _ = pbar.update(1)

        return result.single()
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    @staticmethod
    def _create_new_document(tx, document):
        """Create new document."""

        print(f"\n[Info][ncon] Creating document {document.name} in the Aura database")

        command = "MERGE (d:Document {name: '" + document.name + "'}) \n"

        result = tx.run(command)

        return result.single()
    # -------------------------------------------------------------------------- #

    @staticmethod
    def _add_page_to_document(tx, document, page):
        """Add page to document."""

        print(f"\n[Info][ncon] Adding page {str(page.page_number)} to document {document.name} in the Aura database")

        command = "MATCH (d:Document {name: '" + document.name + "'}) \n "
        command += "MERGE (:Page {name: '" + str(page.page_number) + "'})<-[:CONTAINS]-(d) \n"

        result = tx.run(command)

        return result.single()


    # -------------------------------------------------------------------------- #
    @staticmethod
    def _create_new_date(tx, date):
        """Create new date."""

        command = "MERGE (:Date {name: '" + date + "'}) "
        result = tx.run(command)
        return result.single()
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    @staticmethod
    def _create_new_table(tx, document, page):
        """Create new table."""

        page_name = str(page.page_number)

        command = "MATCH (d:Document {name: '" + document.name + "'}) \n "
        command += "MATCH (p:Page {name: '" + str(page.page_number) + "'}) \n "
        command += "WHERE (d)-[:CONTAINS]->(p) \n"

        command += "CREATE (:Table {document: '" + document.name + "'})<-[:CONTAINS]-(p) \n"
        result = tx.run(command)

        return result.single()
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    @staticmethod
    def _add_rows(tx, document, page):
        """Add rows."""
        print(f"\n[Info][ncon] Adding table from page nb {str(page.page_number)} of document {document.name} in the Aura database")

        ws = page.ws
        column_names = ws.columns.tolist()
        page_name = str(page.page_number)

        base_command = "MATCH (d:Document {name: '" + document.name + "'}) \n "
        base_command += "MATCH (p:Page {name: '" + str(page.page_number) + "'}) \n "
        base_command += "MATCH (t:Table) \n "
        base_command += "WHERE (d)-[:CONTAINS]->(p)-[:CONTAINS]->(t) \n"

        pbar = tqdm(total=len(ws))

        for iCpt, iRow in ws.iterrows():
            command = cp.copy(base_command)
            command += "MERGE (r:RowName {name: '" + str(iRow[0]).replace("'", "") + "', document: '" + document.name + "'})<-[:CONTAINS]-(t) \n"

            for iCpt, iTableCell in enumerate(iRow[1:]):
                command += "MERGE (:TableCell {name: '" + str(iTableCell).replace("'", "") + "', document: '" + document.name + "'})<-[:IS_RELATED_TO {colunmName: '" + str(column_names[iCpt+1]) + "'}]-(r) \n"

            result = tx.run(command)
            _ = pbar.update(1)

        return result.single()
    # -------------------------------------------------------------------------- #



# #######################################################################################################################

#                                          # === END OF FILE === #

# #######################################################################################################################
