# OCR_app

## Définition
L'application est un moteur de recherche permettant de trouver la ou les pages de divers documents contenant les informations recherchées. Elle est divisée en deux onglets :
 - Une partie "Import d'un nouveau document" 
 - Une partie "Moteur de recherche"
 
## Description

### Partie "Import d'un nouveau document" 
Cette étape de l'application permet d'importer divers types de documents (PNG, PDF, JPG, JPEG). Pour les formats PDF, il est possible d'importer une page ou le document complet. 
Deux formats sont importés :
- l'image sélectionnée (JPG, JPEG, PNG ou la page sélectionnée si PDF) qui va ensuite être stockée sur un bucket AWS S3.
- le texte "brut" sous format .txt qui va ensuite être stocké sur un autre bucket AWS S3. 
> Les noms des deux buckets sont à définir dans le fichier .py

### Partie "Moteur de recherche"
Cette étape permet de faire une recherche en 107 langages différents dans des documents en 5 langages différents (Français, Anglais, Espagnol, Italien et Allemand). 
> La seule étape de "sémantique" existante est une sélection des mots clés pertinents basée sur la bibliothèque "Rake-nltk".
 
Sont affichés une fois la recherche lancée :
- la quantité de document dans la base de données.
- le nombre de documents prédits comme "positif".
- un tableau contenant le nom de ces documents, la langue dans laquelle ils sont ainsi qu'un score de prédiction. 
> Ce score est basé sur la fréquence d'apparition du ou des mots de la requête sur le nombre total de mot unique.
- un extrait "brut" des documents trouvés (300 caractères en commençant par le premier mot clé trouvé).
- la possibilité d'afficher la page ou le document initial contenant l'information.

## Contenu du Git

### Le Git doit absolument contenir les éléments suivants :
- le dossier se_indexdir pour la création et le stockage de l'indexer.
- le dossier Search_Engine contenant :
  - le fichier OCRplus_app_demo.py (l'application).
  - le fichier SearchEngine_app.py (pour la partie moteur de recherche.
- un fichier package.txt contenant : 
  ```ruby
  libgl1-mesa-glx
  libglib2.0-0
  poppler-utils
  tesseract-ocr
  tesseract-ocr-fra
  poppler-utils
  ```
- un fichier requirements.txt contenant :
  ```ruby
  boto3==1.17.106
  deep-translator==1.5.4
  opencv-python==4.5.3.56
  pdf2image==1.16.0
  pytesseract==0.3.8
  rake-nltk==1.0.6
  s3fs==2021.10.1
  seaborn==0.11.2
  Whoosh==2.7.4
  ```
  
### Il peut également contenir :
 - Le dossier app_logos à la racine contenant les logos à afficher 
 - Un dossier .streamlit contenant un fichier .config.toml avec le contenu suivant :  
    ```ruby

    [theme]

    # The preset Streamlit theme that your custom theme inherits from. One of "light" or "dark".
    base =

    # Primary accent color for interactive elements.
    primaryColor =

    # Background color for the main content area.
    #backgroundColor =

    # Background color used for the sidebar and most interactive widgets.
    secondaryBackgroundColor =

    # Color used for almost all text.
    textColor =

    # Font family for all text in the app, except code blocks. One of "sans serif", "serif", or "monospace".
    #font =
    ```
    > A compléter avec les couleurs choisies 

## Déploiement sur Streamlit

### 1. Création d'un compte
Tout d'abord il faut créer un compte sur <https://share.streamlit.io/>
  > la version "*Community*" autorise 3 applications mais qui doivent se trouver sur un *repository* Git publique.
  >
  > La version "*Teams*" coute 250$ mensuel et permet le déploiement de 10 applications privées. 


### 2. Ajout d'une nouvelle App
- Cliquer sur *New app*
- Sélectionner le *repository* git 
- Sélectionner la branche
- Sélectionner l'application à déployer 
- Cliquer sur *deploy*

### 3. Une fois l'application déployée
- Revenir sur <https://share.streamlit.io/> et se connecter
- Cliquer sur les ... verticaux à droite du nom de l'app 
- Cliquer sur *settings*
- Cliquer sur *secrets*
- Remplir les champs suivants :
    - AWS_ACCESS_KEY_ID = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    - AWS_SECRET_ACCESS_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    > Le lien vers AWS se fait via les bibliothèques boto3 et s3fs. La création de ces deux IDs est nécessaire au bon fonctionnement de l'application
