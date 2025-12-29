# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Inclusion Financière", layout="centered")


@st.cache_resource  # Pour éviter de recharger le modèle à chaque clic
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler


model, scaler = load_assets()

st.title("Prédiction d'Accès Bancaire")
st.write("Saisissez les informations pour prédire si un individu possède un compte.")

# Création du formulaire
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Pays (Code)", [0, 1, 2, 3])
        year = st.number_input("Année", value=2018)
        location = st.selectbox("Zone (0: Rural, 1: Urbain)", [0, 1])
        cellphone = st.selectbox("Accès Mobile (0: Non, 1: Oui)", [0, 1])
        household = st.slider("Taille du foyer", 1, 20, 3)

    with col2:
        age = st.number_input("Âge", 16, 100, 30)
        gender = st.selectbox("Genre (0: Femme, 1: Homme)", [0, 1])
        rel = st.selectbox("Lien Chef (0-5)", range(6))
        marital = st.selectbox("Statut Marital (0-4)", range(5))
        education = st.selectbox("Éducation (0-6)", range(7))
        job = st.selectbox("Métier (0-9)", range(10))

    submit = st.form_submit_button("Lancer la prédiction")

if submit:
    # 1. Créer un DataFrame avec les noms de colonnes exacts
    data = pd.DataFrame([[country, year, location, cellphone, household, age, gender, rel, marital, education, job]],
                        columns=['country', 'year', 'location_type', 'cellphone_access', 'household_size',
                                 'age_of_respondent', 'gender_of_respondent', 'relationship_with_head',
                                 'marital_status', 'education_level', 'job_type'])

    # 2. Appliquer le Scaler sur les colonnes numériques (comme dans votre entraînement)
    numerical_cols = ['year', 'household_size', 'age_of_respondent']
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    # 3. Prédiction
    pred = model.predict(data)[0]

    if pred == 1:
        st.success("Résultat : Cet individu possède probablement un compte bancaire.")
    else:
        st.warning("Résultat : Cet individu ne possède probablement pas de compte.")