import os
import streamlit as st
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import PyPDF2
from docx import Document
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

# --- Force UTF-8 pour les graphiques (évite les erreurs avec les accents) ---
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# --- Configuration de la page ---
st.set_page_config(
    page_title="EVA - Évaluation des résumés",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='color:#4b77ff;'>EVA - Outil d'évaluation des résumés juridiques</h1>", unsafe_allow_html=True)
st.write("Téléchargez deux fichiers (PDF ou DOCX) : un document original et son résumé, puis lancez l'évaluation.")

# --- Sidebar OpenAI ---
st.sidebar.header("⚙️ Configuration OpenAI")
openai_api_key = st.sidebar.text_input("Clé API OpenAI", type="password")
model_string = st.sidebar.text_input(
    "Modèle OpenAI",
    value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
)

# Vérification de la clé API
if not openai_api_key:
    st.warning("⚠️ Veuillez entrer votre clé OpenAI pour évaluer le résumé.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_MODEL"] = model_string

# --- Fonctions pour extraire le texte ---
def extraire_texte_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    textes = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(textes)

def extraire_texte_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def nettoyer_texte(texte):
    return ' '.join(texte.split())

# --- Uploads côte à côte ---
col1, col2 = st.columns(2)
with col1:
    fichier1 = st.file_uploader("📄 Document original", type=["pdf", "docx"])
with col2:
    fichier2 = st.file_uploader("📝 Résumé", type=["pdf", "docx"])

# --- Evaluation ---
if fichier1 and fichier2 and openai_api_key:
    # Extraction
    texte1 = extraire_texte_pdf(fichier1) if fichier1.type == "application/pdf" else extraire_texte_docx(fichier1)
    texte2 = extraire_texte_pdf(fichier2) if fichier2.type == "application/pdf" else extraire_texte_docx(fichier2)

    texte1_nettoye = nettoyer_texte(texte1)
    texte2_nettoye = nettoyer_texte(texte2)

    # Affichage côte à côte
    st.subheader("📑 Textes comparés")
    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Texte original", texte1_nettoye, height=250)
    with col2:
        st.text_area("Résumé", texte2_nettoye, height=250)

    if st.button("🚀 Évaluer le résumé"):
        try:
            # Définition des métriques
            metrics = [
                GEval(
                    name="Concision",
                    criteria="Le résumé est-il concis et représente tous les éléments clés de la source ? Réponds strictement en français.",
                    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                ),
                GEval(
                    name="Couverture",
                    criteria="Le résumé couvre-t-il toutes les informations juridiques importantes du texte source ? Réponds strictement en français.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                ),
                GEval(
                    name="Exactitude",
                    criteria="Le résumé représente-t-il fidèlement le contenu du texte source sans erreurs factuelles ? Réponds strictement en français.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                ),
                GEval(
                    name="Citations",
                    criteria="S'il y a des articles de loi ou jurisprudences cités dans le texte source, sont-ils identiques dans le résumé ? Réponds strictement en français.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                )
            ]

            # Cas de test
            cas_de_test = [LLMTestCase(input=texte1_nettoye, actual_output=texte2_nettoye)]

            # Évaluation
            resultat = evaluate(cas_de_test, metrics=metrics)

            # Traitement des résultats
            if hasattr(resultat, 'test_results') and resultat.test_results:
                test_result = resultat.test_results[0]
                if hasattr(test_result, 'metrics_data') and test_result.metrics_data:
                    donnees = []
                    for m in test_result.metrics_data:
                        donnees.append({
                            'Métrique': m.name,
                            'Score': getattr(m, 'score', 0),
                            'Seuil': getattr(m, 'threshold', 0.5),
                            'Succès': '✅ Succès' if getattr(m, 'success', False) else '❌ Échec',
                            'Raison': getattr(m, 'reason', 'Pas de justification fournie.')
                        })
                    df = pd.DataFrame(donnees)

                    # Résultats tabulaires
                    st.subheader("📊 Résultats d'évaluation")
                    st.dataframe(df, use_container_width=True)

                    # Graphiques côte à côte
                    col_g1, col_g2 = st.columns(2)

                    # Bar Chart
                    with col_g1:
                        plt.figure(figsize=(6, 5))
                        sns.barplot(data=df, x='Métrique', y='Score', hue='Succès', dodge=False)
                        plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1.5, label='Seuil')
                        plt.ylim(0, 1)
                        plt.title("Scores par métrique")
                        plt.legend()
                        st.pyplot(plt)

                    # Radar Chart
                    with col_g2:
                        categories = df['Métrique']
                        values = df['Score'].tolist()
                        values += values[:1]  # fermer le radar
                        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                        angles += angles[:1]

                        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                        ax.plot(angles, values, linewidth=2, linestyle='solid')
                        ax.fill(angles, values, alpha=0.3)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(categories)
                        ax.set_ylim(0, 1)
                        plt.title("Radar des scores par métrique")
                        st.pyplot(fig)

                    # Logs détaillés dans un accordéon fermé
                    with st.expander("📜 Logs détaillés de l'évaluation"):
                        for m in donnees:
                            st.markdown(f"**{m['Métrique']}** : {m['Raison']}")
                else:
                    st.error("⚠️ Aucune donnée de métrique générée.")
            else:
                st.error("⚠️ Aucun résultat disponible. Vérifie les fichiers.")

        except Exception as e:
            st.error(f"Erreur lors de l'évaluation : {e}")
