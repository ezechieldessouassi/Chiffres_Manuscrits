import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from streamlit_option_menu import option_menu

# Configurer la mise en page de l'application
st.set_page_config(layout="wide", page_icon=":smiley:", page_title="DigitRecognition")

# Charger le modèle une fois pour toutes
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("mon_modele_mnist.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

# Fonction pour afficher la barre de progression
def afficher_barre_de_progression(steps=5, wait=0.5):
    progress_bar = st.progress(0)
    status = st.empty()
    for i in range(steps):
        progress_bar.progress((i + 1) / steps)
        evolution = 100 * (i + 1) / steps
        status.text(f" 🔎 En cours...: {evolution:.0f}% ")
        time.sleep(wait)
    status.text("Analyse terminée ✅ !")

# Fonction pour le prétraitement de l'image
def preprocess_image(img: Image.Image, factor=5):
    img = img.convert('L')  # Convertir en niveaux de gris
    img = img.resize((28, 28))  # Redimensionner à 28x28 pixels
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)  # Améliorer le contraste
    img_array = np.array(img) / 255.0  # Normaliser entre 0 et 1
    img_array = 1 - img_array  # Inverser les couleurs si nécessaire
    img_flattened = img_array.reshape(1, 28 * 28)
    return img_flattened
# CSS pour améliorer l'apparence
st.markdown("""
    <style>
        /* Titre principal */
        h1 {
            font-family: 'Arial', sans-serif;
            color: #1f77b4;
            font-size: 40px;
            text-align: center;
            font-weight: bold;
        }
        
        /* Sous-titres */
        h2, h3 {
            font-family: 'Poppins', sans-serif;
            color: #ff6f61;
            font-size: 25px;
            font-weight: bold;
        }
        
        /* Texte général */
        p, li, span {
            font-family: 'Nunito SemiBold', sans-serif;
            color: #333;
            font-size: 13x;
            line-height: 1.6;
        }
        
        /* Texte dans la barre latérale */
        .sidebar .sidebar-content {
            font-family: 'Poppins', sans-serif;
            color: #4d4d4d;
        }

        /* Couleur des liens */
        a {
            color: #0073e6;
            text-decoration: none;
        }

        /* Ajouter un survol aux liens */
        a:hover {
            color: #ff6f61;
        }
        
        /* Barre de progression */
        .stProgress {
            height: 20px;
        }
    </style>
""", unsafe_allow_html=True)
# Page d'accueil
def page_accueil():
    st.image("dalle.jpg", use_container_width=True)
    st.markdown('<h1 style="color:#1f77b4;text-align:center;">Bienvenue dans l\'application DigitRecognition</h1>', unsafe_allow_html=True)
    st.info(
        "Cette application utilise un modèle de deep learning pour prédire le chiffre manuscrit présent sur une image. "
        "Elle est conçue pour offrir une interface simple et intuitive aux utilisateurs."
    )
    
    gif_path = "mnist0.gif"
    a, b, c = st.columns(3)
    b.image(gif_path, use_container_width=False)
    
    # Section 1 : Description de la base MNIST
    st.header("📚 1. Description de la base de données MNIST")
    st.markdown("""
    - **MNIST (Modified National Institute of Standards and Technology)** est un dataset de référence pour les algorithmes de reconnaissance d'images.
    - **Contenu** : 
  - 60 000 images pour l'entraînement.
  - 10 000 images pour le test.
  - Chaque image est en niveaux de gris, de taille **28x28 pixels**, et représente un chiffre manuscrit (0 à 9).
""")

    # Section 2 : Traitement matriciel
    st.header("🖼️ 2. Traitement matriciel des images")
    st.markdown("Les images sont représentées sous forme de matrices de dimension 28x28, où chaque valeur correspond à l'intensité d'un pixel (entre 0 et 255). Voici les étapes de traitement :")

    st.subheader("1. Normalisation")
    st.latex(r"x_{\text{norm}} = \frac{x}{255}")
    st.markdown("Chaque pixel est divisé par 255 pour que les valeurs soient comprises entre 0 et 1.")

    st.subheader("2. Aplatissement")
    st.markdown("Chaque image est transformée en un vecteur de dimension 784 = 28x28.")

    st.subheader("3. Inversion des couleurs (si nécessaire)")
    st.latex(r"x_{\text{inversé}} = 1 - x_{\text{norm}}")

    # Section 3 : Prétraitement
    st.header("✨ 3. Prétraitement des données")
    st.markdown("""
    - **Amélioration du contraste** : Utilisation de transformations pour augmenter le contraste des chiffres et les rendre plus visibles pour le modèle.
    - **Mise à l'échelle** : Les valeurs normalisées sont prêtes pour être traitées par le réseau de neurones.
    """)

    # Section 4 : Réseau de neurones
    st.header("🤖 4. Réseau de neurones utilisé")
    st.markdown("""
    - **Architecture** :
    - Une première couche dense de **128 neurones** avec activation **ReLU** :
    """)
    st.latex(r"f(x) = \max(0, x)")
    st.markdown("""
    - Deux couches cachées supplémentaires avec **64 et 32 neurones**, également avec ReLU.
    - Une couche de sortie avec **10 neurones**, utilisant l'activation **Softmax** pour produire les probabilités des classes :
    """)
    st.latex(r"\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}")
    st.markdown("où $z_i$ est la sortie brute (logit) pour la classe $i$.")

    st.subheader("Fonction de perte")
    st.markdown("La fonction utilisée est l'entropie croisée catégorique (Sparse Categorical Crossentropy) :")
    st.latex(r"\text{Loss} = -\frac{1}{N} \sum_{i=1}^N y_i \log(\hat{y}_i)")
    st.markdown("""
    où :
    - $y_i$ est la classe réelle.
    - $\hat{y}_i$ est la probabilité prédite.
    - $N$ est le nombre d'échantillons.
    """)

    # Section 5 : Visualisation des prédictions
    st.header("📊 5. Visualisation des prédictions")
    st.markdown("""
    - Pour chaque image donnée, le modèle produit une probabilité pour chaque chiffre (0 à 9). Le chiffre avec la probabilité maximale est celui prédit :
    """)
    st.latex(r"\text{Classe prédite} = \arg\max_{i} (\text{Softmax}(z_i))")
    st.markdown("""
    - L'interface vous permet de télécharger une image et d'obtenir :
      - Le chiffre prédit.
      - Les probabilités associées à chaque classe.
    """)

    st.markdown("Merci d'utiliser notre application ! 🎉 N'hésitez pas à explorer les fonctionnalités disponibles et à nous donner vos retours.")
 
    
    

# Page de prédiction
def page_upload_image():
    st.markdown('<h1 style="color:#1f77b4;text-align:center;">Prédiction des chiffres manuscrits</h1>', unsafe_allow_html=True)
    st.write("Chargez une image pour prédire le chiffre manuscrit qu'elle contient.")
    
    upload = st.file_uploader("Chargez une image (formats acceptés : PNG, JPEG, JPG)", type=["png", "jpeg", "jpg"])
    
    if upload is not None:
        # Lire et afficher l'image téléchargée
        image = Image.open(upload)
        c1, c2 = st.columns(2)
        c1.image(image, caption="Image chargée", use_container_width=False)
        
        # Prétraiter l'image et prédire
        img_traitee = preprocess_image(image)
        afficher_barre_de_progression(steps=10, wait=0.3)
        
        if model is not None:
            predictions = model.predict(img_traitee)
            classe = int(np.argmax(predictions))
            probabilites = [round(float(proba), 4) for proba in predictions.flatten()]
            
            # Afficher les résultats
            df_probabilites = pd.DataFrame({
                'Chiffre': list(range(len(probabilites))),
                'Probabilité': probabilites
            })
            c2.table(df_probabilites)
            c2.write(f"**Classe prédite :** {classe}")
        else:
            c2.error("Le modèle n'a pas pu être chargé.")
            
        st.warning("Les prédictions générées par cet outil ne sont pas toujours correctes. Vérifiez l'image avant utilisation.")

# Page Feedback
def page_feedback():
    st.markdown('<h1 style="color:#1f77b4;text-align:center;">Donnez-nous votre avis 😊</h1>', unsafe_allow_html=True)
    st.write("Nous aimerions savoir ce que vous pensez de notre application.")
    
    feedback = st.text_area("Laissez votre avis ici", placeholder="Écrivez votre feedback ici...")
    
    if st.button("Soumettre"):
        if feedback.strip():  # Vérifier que le champ n'est pas vide
            # Générer un nom de fichier basé sur l'heure actuelle
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"feedback_{timestamp}.txt"
            
            # Enregistrer le feedback dans un fichier
            with open(filename, "w", encoding="utf-8") as f:
                f.write(feedback)
            
            st.success("Merci pour votre retour ! 😊")
            st.write(f"Feedback enregistré. Fichier feedback : **{filename}**")
        else:
            st.warning("Veuillez écrire un feedback avant de soumettre.")
        
def page_contributeurs():
    st.markdown('<h1 style="color:#1f77b4;text-align:center;">Contributeurs 👥</h1>', unsafe_allow_html=True)
    st.write("Voici les contributeurs qui ont participé au développement de cette application :")
    st.markdown("""
    <ul style="list-style-type: none; padding: 0; font-size: 18px; color: #333;">
        <li style="margin-bottom: 20px;">
            <span style="color:#2ca02c; font-weight:bold;">DESSOUASSI Ezéchiel</span><br>
            Élève Ingénieur Statisticien Économiste, ENEAM Cotonou<br>
            <strong>Email :</strong> <a href="mailto:dessouassi.ezechiel@eneam.bj" style="color:#1f77b4;">dessouassi.ezechiel@eneam.bj</a>
        </li>
        <li style="margin-bottom: 20px;">
            <span style="color:#2ca02c; font-weight:bold;">SIMIYAKI Philippe</span><br>
            Élève Ingénieur Statisticien Économiste, ENEAM Cotonou<br>
            <strong>Email :</strong> <a href="mailto:simiyaki.philippe@eneam.bj" style="color:#1f77b4;">simiyaki.philippe@eneam.bj</a>
        </li>
    </ul>
    """, unsafe_allow_html=True)

# Menu principal
# Menu vertical dans la barre latérale
with st.sidebar:
    selected_page = option_menu(
        "Menu Principal",  # Titre du menu
        ["Accueil", "Predict on Picture (PoP)", "Feedback", "Contributeurs"],  # Noms des pages
        icons=['house', 'p-square-fill', 'envelope-arrow-up', 'people'],  # Icônes des pages
        menu_icon="cast",  # Icône du menu principal
        default_index=0,  # Page par défaut sélectionnée
    )

# Informations supplémentaires dans la barre latérale
st.sidebar.title("A propos")
st.sidebar.info(
    """
    Cette application est développée dans un but académique. 
    Il est réalisé dans le cadre du projet à rendre pour le cours de Machine Learning 1 enseigné aux Elèves Ingénieurs Statisticiens Economistes en deuxième année (ISE2).
    """
)


if selected_page == "Accueil":
    page_accueil()
elif selected_page == "Predict on Picture (PoP)":
    page_upload_image()
elif selected_page == "Feedback":
    page_feedback()
elif selected_page == "Contributeurs":
    page_contributeurs()

