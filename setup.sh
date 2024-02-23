#!/bin/bash

# Installation de Streamlit
pip install streamlit

# Création du répertoire ~/.streamlit s'il n'existe pas
mkdir -p ~/.streamlit

# Création du fichier de configuration pour Streamlit
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \$PORT\n\
" > ~/.streamlit/config.toml

# Installation des dépendances Python à partir du fichier requirements.txt
pip install -r requirements.txt