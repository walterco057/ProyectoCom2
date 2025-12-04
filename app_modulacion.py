import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq

import torch
import torch.nn as nn
import streamlit as st

# ROOT_DIR = carpeta ProyectoCom2 (donde está este script)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Ahora importamos usando el paquete src
from src.generate_signals import generate_baseband


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILENAME = "iq_cnn_model.pth"  # cambia si usas otro nombre
MODEL_PATH = os.path.join(ROOT_DIR, "data", MODEL_FILENAME)


# -------------------------------------------------------
# 2. Definición de la CNN (DEBE coincidir con la del entrenamiento)
#    Versión para 4 canales de entrada: [I, Q, |s|, fase]
# -------------------------------------------------------
class IQCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2),  # 4 canales de entrada
            nn.ReLU(),
            nn.MaxPool1d(2),      # 256 -> 128

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),      # 128 -> 64

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),      # 64 -> 32
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),            # 64 * 32 si Nsym=256 y 3 pool(2)
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------------------------------------
# 3. Utilidades para interpretar el nombre de la modulación
# -------------------------------------------------------
def parse_modulation_name(name: str):
    """
    Separa tipo y orden para nombres como:
    'ASK2', 'ASK4', 'BPSK', 'QPSK', 'PSK8', 'QAM16', 'QAM32', 'FSK2', etc.
    """
    n = str(name).upper()

    # Casos especiales
    if n == "BPSK":
        return "PSK", 2
    if n == "QPSK":
        return "PSK", 4

    # PSK genérico: 'PSK8', 'PSK16', ...
    if n.startswith("PSK"):
        suf = n[3:]
        try:
            order = int(suf)
        except ValueError:
            order = None
        return "PSK", order

    # ASK / FSK / QAM típicos: 'ASK2', 'QAM16', 'FSK2', etc.
    for t in ["ASK", "FSK", "QAM"]:
        if n.startswith(t):
            suf = n[len(t):]
            try:
                order = int(suf)
            except ValueError:
                order = None
            return t, order

    return "DESCONOCIDO", None


def describe_modulation(name: str) -> str:
    tipo, orden = parse_modulation_name(name)
    base = f"Modulación {tipo}"
    if orden is not None:
        base += f" de orden M={orden}"
    base += "."

    if tipo == "ASK":
        extra = (
            " La información se transmite variando niveles de amplitud en la parte real. "
            "Para M>2 hay varios niveles discretos; para M=2 (ASK2 unipolar) hay un nivel apagado/encendido."
        )
    elif tipo == "PSK":
        extra = (
            " La amplitud es aproximadamente constante y la información se codifica en la fase. "
            "A mayor orden M, hay más puntos equiespaciados en el círculo unitario."
        )
    elif tipo == "QAM":
        extra = (
            " La información se codifica tanto en amplitud como en fase. "
            "La constelación típica es una rejilla rectangular en el plano I/Q."
        )
    elif tipo == "FSK":
        extra = (
            " La información se representa mediante diferentes frecuencias portadoras."
        )
    else:
        extra = " Tipo de modulación no reconocido en el esquema actual."

    return base + extra


# -------------------------------------------------------
# 4. Cargar modelo y clases (cacheado para no recargar siempre)
# -------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    raw_classes = checkpoint["classes"]
    class_names = [str(c) for c in raw_classes]
    num_classes = len(class_names)

    model = IQCNN(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


# -------------------------------------------------------
# 5. Función para clasificar y devolver info + figuras
# -------------------------------------------------------
def classify_and_plot(model,
                      class_names,
                      s_complex: np.ndarray,
                      fs_sym: float | None = None):
    """
    s_complex: señal compleja de tamaño Nsym (como s_noisy de generate_baseband)
    """

    s_complex = np.asarray(s_complex)
    Nsym = s_complex.size

    # Construimos los 4 canales: I, Q, |s|, fase
    I = np.real(s_complex)
    Q = np.imag(s_complex)
    mag = np.abs(s_complex)
    phase = np.angle(s_complex)

    iq = np.stack([I, Q, mag, phase], axis=0).astype(np.float32)  # (4, Nsym)

    X_tensor = torch.from_numpy(iq).unsqueeze(0).to(device)       # (1, 4, Nsym)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_idx = pred_idx.item()
    conf = conf.item()
    pred_name = class_names[pred_idx]
    tipo, orden = parse_modulation_name(pred_name)
    desc = describe_modulation(pred_name)

    # ---- Figuras ----
    figs = {}

    # 1) Forma de onda (parte real)
    fig1, ax1 = plt.subplots(figsize=(6, 2))
    if fs_sym is not None:
        t = np.arange(Nsym) / fs_sym
        ax1.plot(t, I, marker='o', linewidth=0.8)
        ax1.set_xlabel("Tiempo (s)")
    else:
        ax1.plot(I, marker='o', linewidth=0.8)
        ax1.set_xlabel("n (símbolo)")
    ax1.set_title("Forma de onda (parte real)")
    ax1.grid(True)
    figs["waveform"] = fig1

    # 2) Constelación
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.scatter(I, Q, s=8)
    ax2.axhline(0, color='k', linewidth=0.3)
    ax2.axvline(0, color='k', linewidth=0.3)
    ax2.set_xlabel("I")
    ax2.set_ylabel("Q")
    ax2.set_title("Diagrama de constelación")
    ax2.grid(True, alpha=0.4)
    ax2.set_aspect('equal', 'box')
    figs["constellation"] = fig2

    # 3) Espectro (FFT)
    S = fftshift(fft(s_complex))
    N = len(S)
    S_mag = 20 * np.log10(np.abs(S) + 1e-12)

    if fs_sym is not None:
        freqs = fftshift(fftfreq(N, d=1/fs_sym))
        x_label = "Frecuencia (Hz)"
    else:
        freqs = fftshift(fftfreq(N, d=1.0))
        x_label = "Frecuencia normalizada"

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot(freqs, S_mag, linewidth=0.8)
    ax3.set_title("Espectro |S(f)| en dB")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Magnitud (dB)")
    ax3.grid(True, alpha=0.4)
    figs["spectrum"] = fig3

    return {
        "pred_name": pred_name,
        "tipo": tipo,
        "orden": orden,
        "conf": conf,
        "descripcion": desc,
        "figs": figs,
    }


# -------------------------------------------------------
# 6. Interfaz Streamlit
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Demo Clasificador de Modulación", layout="wide")

    st.title("Clasificación automática de esquemas de modulación")
    st.write(
        "Proyecto 2 – Aplicación de IA en Comunicaciones Digitales\n\n"
        "Selecciona un tipo de modulación, SNR y número de símbolos. "
        "La app generará una señal simulada, la clasificará con la CNN y mostrará "
        "las características principales (forma de onda, constelación, espectro)."
    )

    # Cargar modelo y clases
    with st.spinner("Cargando modelo..."):
        model, class_names = load_model_and_classes()

    # Panel lateral (controles)
    st.sidebar.header("Parámetros de la señal")

    # Lista de modulaciones disponibles
    mods_disponibles = class_names  # asumiendo que el modelo fue entrenado con los mismos nombres
    modulation = st.sidebar.selectbox("Modulación simulada", mods_disponibles, index=0)

    snr_db = st.sidebar.slider("SNR (dB)", min_value=0, max_value=30, value=10, step=1)
    Nsym = 256
    st.sidebar.write(f"Número de símbolos: **{Nsym}**")

    fs_sym = None  # si quieres un eje de tiempo real, puedes poner algo tipo 1e3

    if st.sidebar.button("Generar y clasificar"):
        with st.spinner("Generando señal y clasificando..."):
            # Generar señal en banda base
            out = generate_baseband(modulation, Nsym=Nsym, snr_db=snr_db)
            s = out["s_noisy"]

            # Clasificar y obtener figuras
            result = classify_and_plot(model, class_names, s, fs_sym=fs_sym)

        # Mostrar resultados
        st.subheader("Resultado del clasificador")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Predicción:** `{result['pred_name']}`")
            st.markdown(f"- Tipo: `{result['tipo']}`")
            st.markdown(f"- Orden: `{result['orden']}`")
        with col2:
            st.markdown(f"**Confianza:** `{result['conf']*100:.2f}%`")

        st.markdown("**Descripción breve de la modulación predicha:**")
        st.write(result["descripcion"])

        st.markdown("---")
        st.subheader("Características de la señal generada")

        # Mostrar figuras en tres columnas
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Forma de onda (I)")
            st.pyplot(result["figs"]["waveform"])
        with c2:
            st.markdown("### Constelación")
            st.pyplot(result["figs"]["constellation"])
        with c3:
            st.markdown("### Espectro")
            st.pyplot(result["figs"]["spectrum"])


if __name__ == "__main__":
    main()
