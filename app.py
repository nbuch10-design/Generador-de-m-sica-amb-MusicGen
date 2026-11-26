import scipy
import torch
from transformers import pipeline
import streamlit as st
from datetime import datetime

# dispositiu: cuda o cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Creador de canciones", page_icon="♫")
st.title("Generador de música amb MusicGen ♫")
st.subheader(f"Genera petites peces musicals a partir de descripcions de text utilitzant facebook/musicgen-smal\nNoelia Buch {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
st.markdown('<p style="font-size:22px; font-weight:bold;">Descriu la música que vols generar en anglés</p>', unsafe_allow_html=True)
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #FF4B4B; /* Color de fondo (Rojo Streamlit) */
        color: white;              /* Color del texto */
        font-size: 24px;           /* Tamaño de la letra */
        border-radius: 10px;       /* Bordes redondeados */
        padding: 10px 20px;        /* Espaciado interno */
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource  # carrega el model una sola vegada per a tots els usuaris
def load_model():
    return pipeline("text-to-audio", model="facebook/musicgen-small", tokenizer="facebook/musicgen-small", device=-1)  # -1 = CPU

generador = load_model()

def generate_music(prompt):
    sortida= generador(prompt)
    sampling_rate = sortida["sampling_rate"]
    audio_data = sortida["audio"].T
    output_filename = 'output_music.wav'
    scipy.io.wavfile.write(output_filename, rate= sampling_rate, data= audio_data)
    return output_filename


text1 = st.text_area("", height=25)

boton1 = st.button("Generar")
if boton1:
    if not text1.strip():
        st.warning("Escribe un texto para generar una canción")
    else:
        with st.spinner("Generando canción..."):
            resultado = generate_music(text1)

        st.subheader("Canción")
        st.audio(resultado, format="audio/wav")