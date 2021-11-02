#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:54:53 2021

@author: maximejacoupy
"""
# from https://appliedmachinelearning.blog/2018/07/31/developing-a-fast-indexing-and-full-text-search-engine-with-whoosh-a-pure-python-library/


# #######################################################################################################################
#                                              # === LIBRAIRIES === #
# #######################################################################################################################

from rake_nltk import Rake
from whoosh.qparser import QueryParser
from whoosh.index import open_dir
from deep_translator import GoogleTranslator
import seaborn as sns
from whoosh.lang.morph_fr import variations

# #######################################################################################################################
#                                              # === FUNCTIONS === #
# #######################################################################################################################


class SearchEngine():
    """This class allow a processing of text data into Whoosh database format and create a Search Engine."""

    def __init__(self):
        self.ix = self.retrieve_indexer()
        self.idx_lenght = self.retrieve_indexer_lenght()
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def retrieve_indexer(self):
        ix = open_dir("se_indexdir")
        return ix
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def retrieve_indexer_lenght(self):
        idx_lenght = self.ix.doc_count_all()
        return idx_lenght
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def search_in_database_document(self, important_text, content=False):
        """General method for the search engine.

        :param important_text: list contining the most important word in the sentence
        :type important_text: list
        :param max_response_number: number of text being kept. select first text with high number of hint
        :type max_response_number: int
        :param content: if the content of the document has to be expo, defaults to False
        :type content: bool
        :raises IndexError: if no hint
        :returns: contain 'Key Words', 'Documents to analyze', 'Documents containing key words'
                    and the content for each document
        :rtype: dict
        """

        # for type of scoring : https://github.com/jerem/Whoosh/blob/master/src/whoosh/scoring.py
        searcher = self.ix.searcher() # or weighting=scoring.TF_IDF() or weighting=scoring.Frequency
        query_str = ', '.join(important_text)
        query = QueryParser("content", self.ix.schema).parse(query_str)
        results = searcher.search(query, limit=None)

        n = 0
        result_dict = {}
        result_dict['Key Words'] = query_str
        result_dict['Documents to analyze'] = self.idx_lenght
        result_dict['Documents containing key words'] = len(results)
        for i in range(self.idx_lenght):
            result_dict[i] = {}

        try:
            for i in range(self.idx_lenght):
                result_dict[i]['Document'] = results[i]['title']
                result_dict[i]['Score'] = str(round(results[i].score, 3))
                if content is True:
                    result_dict[i]['Content'] = results[i]['textdata']
                n += 1
        except IndexError:
            pass
        return result_dict
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def extract_important_words(self, text, language='english'):
        """Find important word in a sentence.

        :param text: sentence written in the search cell
        :type text: str
        :param language: language used for Rake analysis, defaults to English
        :type language: str
        :returns: list contining the most important word in the sentence
        :rtype: list
        """
        r = Rake(language=language)

        r.extract_keywords_from_text(text)

        extracted_text = r.get_ranked_phrases_with_scores()
        important_text = []
        for iExtract in extracted_text:
            important_text.append(iExtract[1])
        return important_text
    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def analyze_in_different_language(self, text):
        """Do the search in different languages (here french, english, spanish, italian, german)

        :param text: sentence written in the search cell
        :type text: str
        :returns: dictionary contining the result dict for each language
        :rtype: dict
        """
        full_results = {}
        for iLanguage, iLanguage_short in zip(['french', 'english', 'spanish', 'italian', 'german'], ['fr', 'en', 'es', 'it', 'de']):

            translated = GoogleTranslator(source='auto', target=iLanguage_short).translate(text)

            if iLanguage == 'english':
                important_text_tmp = self.extract_important_words(translated, language=iLanguage)
                important_text = []
                for word in important_text_tmp:
                    try:
                        t = list(variations(word))
                    except:
                        t = []
                    important_text.extend(t)         

            else:
                important_text = self.extract_important_words(translated, language=iLanguage)
                
            if self.search_in_database_document(important_text, content=True)[0] != {}:
                full_results[iLanguage] = self.search_in_database_document(important_text, content=True)
            else:
                full_results[iLanguage] = None       
        return full_results

    # -------------------------------------------------------------------------- #


    # -------------------------------------------------------------------------- #
    def draw_plot(self, full_results):
        values = []
        languages = ['french', 'english', 'spanish', 'italian', 'german']
        for iLanguage in languages:
            try:
                tmp_val = full_results[iLanguage]['Documents containing key words']
            except TypeError:
                tmp_val = 0
            values.append(tmp_val)

        sns.barplot(languages, values)
        
# #######################################################################################################################

#                                          # === END OF FILE === #

# #######################################################################################################################
