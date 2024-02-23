import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
import pickle
import shap
import numpy as np
import requests
from pathlib import Path

# Chargement du modèle
df = pd.read_csv('./test_preprocess.csv')
# Charger le modèle
model = pickle.load(open("./best_model.pickle", "rb"))
# Initialiser l'explainer SHAP
explainer = shap.TreeExplainer(model)

def show_score_gauge(probability, threshold=0.475):
    """
    Affiche une jauge de score colorée basée sur la probabilité.
    """
    fig, ax = plt.subplots(figsize=(6, 0.5))
    cmap = mcolors.ListedColormap(['green', 'yellow', 'red'])
    norm = mcolors.BoundaryNorm([0, threshold-0.1, threshold, 1], cmap.N)
    cb2 = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    plt.axvline(probability, color='blue', linestyle='--')
    ax.set_xticks([0, threshold, 1])
    ax.set_xticklabels(['0', f'Seuil: {threshold}', '1'])
    ax.set_title('Probabilité de défaut')
    st.pyplot(fig)

def plot_shap_summary(shap_values, X):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    st.pyplot(plt.gcf())
    plt.clf()
    
def display_client_info(df, client_id):
    """
    Affiche les informations descriptives du client, limitées aux ratios financiers,
    informations sur l'emploi, et statistiques familiales.
    """
    st.subheader("Informations du Client")

    cols_to_display_part1 = {
        'SK_ID_CURR': 'ID Client',
        'INCOME_TO_CREDIT_RATIO': 'Ratio Revenu/Crédit',
        'INCOME_TO_ANNUITY_RATIO': 'Ratio Revenu/Annuité',
        'CREDIT_TO_ANNUITY_RATIO': 'Ratio Crédit/Annuité',
        'AMT_GOODS_PRICE': 'Prix des biens'
    }

    cols_to_display_part2 = {
        'INCOME_TO_ANNUITY_RATIO_BY_AGE': 'Ratio Revenu/Annuité par Âge',
        'CREDIT_TO_ANNUITY_RATIO_BY_AGE': 'Ratio Crédit/Annuité par Âge',
        'PROPORTION_LIFE_EMPLOYED': 'Proportion de la Vie Employée'
    }

    cols_to_display_part3 = {
        'CODE_GENDER': 'Genre',
        'DAYS_BIRTH': 'Âge (en jours)',
        'INCOME_TO_FAMILYSIZE_RATIO': 'Ratio Revenu/Taille de la Famille',
        'ANNUITY_INCOME_PERC': 'Part de l’Annuité dans le Revenu'
    }

    cols_to_display_part4 = {
    'EXT_SOURCE_3': 'Source Externe 3',
    'EXT_SOURCE_2': 'Source Externe 2',
    'EXT_SOURCE_1': 'Source Externe 1'
    }

    # Filtrage et affichage pour la première partie
    client_info_part1 = df.loc[df['SK_ID_CURR'] == client_id, cols_to_display_part1.keys()]
    client_info_part1 = client_info_part1.rename(columns=cols_to_display_part1)
    st.write(client_info_part1)
    st.write("")

    # Filtrage et affichage pour la deuxième partie
    client_info_part2 = df.loc[df['SK_ID_CURR'] == client_id, cols_to_display_part2.keys()]
    client_info_part2 = client_info_part2.rename(columns=cols_to_display_part2)
    st.write(client_info_part2)
    st.write("")

    # Filtrage et affichage pour la troisième partie
    client_info_part3 = df.loc[df['SK_ID_CURR'] == client_id, cols_to_display_part3.keys()]
    client_info_part3 = client_info_part3.rename(columns=cols_to_display_part3)
    st.write(client_info_part3)

    # Filtrage et affichage pour la quatrième partie
    client_info_part4 = df.loc[df['SK_ID_CURR'] == client_id, cols_to_display_part4.keys()]
    client_info_part4 = client_info_part4.rename(columns=cols_to_display_part4)
    st.write(client_info_part4)
    
def compare_client_to_others(df, client_id, feature):
    """
    Compare un client à d'autres clients basé sur une feature spécifique.
    """
    st.subheader(f"Comparaison de {feature}")
    client_value = df.loc[df['SK_ID_CURR'] == client_id, feature].values[0]
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, stat="density", linewidth=0, ax=ax)
    plt.axvline(client_value, color='r', linestyle='--')
    plt.title(f"Distribution de {feature} avec la valeur du client")
    
    st.pyplot(fig)

image_path = "./images/image_7.png"
st.image(image_path, width=200)

st.title('Dashboard Scoring Client')

client_id = st.number_input("Entrez l'ID du client", value=100002, step=1)

if st.button('Prédire le score du client'):
    api_url = "http://127.0.0.1:8000/predict/"
    #api_url = "https://apicredit-bfd7efc2dfcb.herokuapp.com/predict/"
    response = requests.post(api_url, data={'id_client': client_id})
    if response.status_code == 200:
        prediction_data = response.json()
        prediction_text = prediction_data.get("prediction_text", "Erreur dans l'API")
        probability = prediction_data.get("Score", 0)
        explanation = prediction_data.get("Explanation")
        
         # Détermination de la couleur en fonction du texte de prédiction
        if "NON Accordé" in prediction_text:
            color = "red"
            image_path = "./images/denied.png"
            st.write(explanation)
        elif "Accordé" in prediction_text:
            color = "green"
            image_path = "./images/approved.png"
        else:
            color = "#FFDB58"
            image_path = "./images/warning.png"
        
        # Utilisation de colonnes pour afficher le texte et l'image côte à côte
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Affichage de l'image/logo
            st.markdown(f"<h4 style='color: {color}; margin-top:10px;'>{prediction_text}</h4>", unsafe_allow_html=True)

        with col2:
            # Affichage du texte de prédiction
            st.image(image_path, width=50)

    # Col3 est utilisé comme espace vide pour pousser col1 et col2 plus proches l'un de l'autre

        st.write(f"Explication: {explanation}")
        # Affichage de la probabilité de défaut
        st.write(f"Probabilité de défaut : {probability:.2f}")
        show_score_gauge(probability)
        st.subheader("Jauge de Score de Crédit")
        st.text("""
        La jauge indique la probabilité de défaut de remboursement du crédit. Une flèche vers le rouge signifie un risque élevé 
        de non-remboursement, tandis qu'une flèche vers le vert indique un risque faible. Le seuil détermine la décision d'accord 
        du prêt.
        """)
        # Sélection des données du client pour le calcul des valeurs SHAP
        X_client = df[df['SK_ID_CURR'] == client_id].drop(['SK_ID_CURR'], axis=1)
        shap_values = explainer.shap_values(X_client)[1]
        st.subheader("Résumé de l'impact des caractéristiques (SHAP Values)")
        st.text("""
        Ce graphique montre comment chaque caractéristique influence la prédiction du modèle vers un prêt accordé ou refusé. 
        Les valeurs SHAP positives (rouges) poussent la prédiction vers un prêt non accordé, tandis que les négatives (bleues) 
        indiquent une influence vers un prêt accordé. La longueur des barres reflète l'importance de l'impact.
        """)
        plot_shap_summary(shap_values, X_client)
    else:
        st.error(f"Erreur lors de la requête à l'API: {response.status_code}")
        st.text(response.text)

display_client_info(df, client_id)

# Colonnes spécifiques à inclure dans la selectbox
cols_descriptive_names = {
    'INCOME_TO_CREDIT_RATIO': 'Ratio Revenu/Crédit',
    'INCOME_TO_ANNUITY_RATIO': 'Ratio Revenu/Annuité',
    'CREDIT_TO_ANNUITY_RATIO': 'Ratio Crédit/Annuité',
    'INCOME_TO_ANNUITY_RATIO_BY_AGE': 'Ratio Revenu/Annuité par Âge',
    'CREDIT_TO_ANNUITY_RATIO_BY_AGE': 'Ratio Crédit/Annuité par Âge',
    'PROPORTION_LIFE_EMPLOYED': 'Proportion de la Vie Employée',
    'DAYS_BIRTH': 'Âge (en jours)',
    'INCOME_TO_FAMILYSIZE_RATIO': 'Ratio Revenu/Taille de la Famille',
    'AMT_GOODS_PRICE': 'Prix des biens',
    'EXT_SOURCE_3': 'Source Externe 3',
    'EXT_SOURCE_2': 'Source Externe 2',
    'EXT_SOURCE_1': 'Source Externe 1',
    'ANNUITY_INCOME_PERC': 'Part de l’Annuité dans le Revenu',
    'CODE_GENDER': 'Genre'
}

# Selectbox pour afficher uniquement les colonnes spécifiées
feature_choice_descriptive = st.selectbox("Choisir une feature pour la comparaison", options=list(cols_descriptive_names.values()))
feature_choice = [key for key, value in cols_descriptive_names.items() if value == feature_choice_descriptive][0]

if st.button('Comparer avec les autres clients'):
    compare_client_to_others(df, client_id, feature_choice)
    
# cd Desktop/Python/projet_8
# streamlit run Caramanno_Julien_1_dashboard_022024.py