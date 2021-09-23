# streamlit-apps

## Search Engine

### Prepare the repository
1. Create a virtual environment 
2. pip install -r requirements.txt

### Appearance settings
1. ~/.streamlit/ (mac) or %userprofile%/.streamlit/(windows)
2. nano config.tolm
3. change in the [theme] part (ad remove the #):
      1. base="light"
      2. primaryColor="#25b3c2"
      3. secondaryBackgroundColor="#fcf7ef"
      4. textColor="#250044"


### Launch the app
1. streamlit run App.py

### Prepare the parameters
1. Write your text (word(s) or sentence)
2. Choose how many response do you want to display (by default Most relevant --> reponses with with a minimum score of 3 and up to 20 responses
4. Select in which language the keys word will be translated (default: english, french, spanish, italian, german)
5. Choose the parameters you want to display:
      1. Key Words: the selected key word based on your research
      2. Number of document in the database
      3. Number of result: Number of document containing the key words
      4. Score: if you want to display the score of each response in the dataframe

5. Click on search 
