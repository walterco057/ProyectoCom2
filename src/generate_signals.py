import numpy as np

# =========================
# Utilidades generales
# =========================

def gen_random_bits(n_bits: int) -> np.ndarray:
    """Genera n_bits bits aleatorios (0/1)."""
    return np.random.randint(0, 2, size=n_bits)


def bits_to_ints(bits: np.ndarray, k: int) -> np.ndarray:
    """
    Convierte un array de bits (0/1) en enteros de k bits.
    bits: vector 1D de longitud múltiplo de k.
    """
    bits = np.asarray(bits).reshape(-1, k)
    weights = 2 ** np.arange(k - 1, -1, -1)  # [2^(k-1), ..., 1]
    return (bits * weights).sum(axis=1)


def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Agrega ruido AWGN a una señal (real o compleja)."""
    signal = np.asarray(signal)
    snr_linear = 10 ** (snr_db / 10.0)
    power_signal = np.mean(np.abs(signal) ** 2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )

    # si la señal original es real, devolvemos parte real
    if np.isrealobj(signal):
        return signal + noise.real
    return signal + noise


# =========================
# Moduladores
# =========================

def modulate_ask(bits: np.ndarray, M: int) -> np.ndarray:
    """
    Modulador M-ASK en banda base.
    Para M=2 usamos ASK unipolar (tipo OOK) -> niveles {0, A}
    Para M>2 usamos ASK simétrica alrededor de 0.
    """
    bits = np.asarray(bits)
    k = int(np.log2(M))
    assert bits.size % k == 0, "Número de bits no múltiplo de log2(M)"

    ints = bits_to_ints(bits, k)  # 0..M-1

    if M == 2:
        # 2-ASK unipolar: {0, 2} y luego normalizamos potencia promedio = 1
        levels = np.array([0.0, 2.0], dtype=np.float32)
    else:
        # M-ASK simétrica para M>2, como antes (p.ej. [-3, -1, 1, 3] para M=4)
        levels = np.arange(-(M - 1), M, 2)

    s = levels[ints].astype(np.complex64)
    # Normalizar potencia promedio a 1
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    return s

def modulate_fsk(bits: np.ndarray,
                 M: int,
                 df_norm: float = 0.05) -> np.ndarray:
    """
    Modulador M-FSK en banda base discreta.
    Representamos FSK como una señal de fase acumulada, donde cada símbolo m
    tiene una 'velocidad de giro' distinta (frecuencia instantánea diferente).

    bits: array de 0/1
    M: orden de FSK (2, 4, ...)
    df_norm: separación normalizada en radianes por muestra entre frecuencias adyacentes.
             Valores típicos: 0.02 - 0.1
    """
    bits = np.asarray(bits)
    k = int(np.log2(M))
    assert bits.size % k == 0, "Número de bits no múltiplo de log2(M)"

    ints = bits_to_ints(bits, k)  # 0..M-1
    Nsym = ints.size

    # Definimos M frecuencias angulares distintas alrededor de 0
    # (centradas en 0 para mantener simetría)
    # ej: para M=2 -> [-df_norm/2, +df_norm/2]
    #     para M=4 -> [-3df/2, -df/2, +df/2, +3df/2]
    freqs = (np.arange(M) - (M - 1) / 2.0) * (2.0 * np.pi * df_norm)

    # Para cada símbolo elegimos la frecuencia correspondiente
    omega = freqs[ints]              # (Nsym, )

    # Fase acumulada: θ[n] = θ[n-1] + ω[n]
    theta = np.cumsum(omega)

    # Señal compleja en banda base
    s = np.exp(1j * theta).astype(np.complex64)

    # Normalizamos potencia promedio a 1
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    return s

def modulate_psk(bits: np.ndarray, M: int) -> np.ndarray:
    """
    Modulador M-PSK en banda base (círculo unitario).
    Para M=2 es BPSK, M=4 es QPSK, etc.
    """
    bits = np.asarray(bits)
    k = int(np.log2(M))
    assert bits.size % k == 0, "Número de bits no múltiplo de log2(M)"

    ints = bits_to_ints(bits, k)  # 0..M-1
    phases = 2 * np.pi * ints / M
    s = np.exp(1j * phases).astype(np.complex64)
    # potencia promedio ~1
    return s


def modulate_qam_rect(bits: np.ndarray, Mx: int, My: int) -> np.ndarray:
    """
    Modulador QAM rectangular M = Mx * My.
    Ejemplos:
      - QAM16 -> Mx=4, My=4
      - QAM32 -> Mx=4, My=8
    """
    bits = np.asarray(bits)
    M = Mx * My
    k = int(np.log2(M))
    assert bits.size % k == 0, "Número de bits no múltiplo de log2(M)"

    ints = bits_to_ints(bits, k)  # 0..M-1
    row = ints // Mx
    col = ints % Mx

    I_levels = np.arange(-(Mx - 1), Mx, 2)
    Q_levels = np.arange(-(My - 1), My, 2)

    I = I_levels[col]
    Q = Q_levels[row]

    s = (I + 1j * Q).astype(np.complex64)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))  # normalizar potencia
    return s


# Wrappers por si quieres llamar BPSK/QPSK directo
def bpsk(bits: np.ndarray) -> np.ndarray:
    """BPSK como caso particular de 2-PSK."""
    return modulate_psk(bits, M=2)


def qpsk(bits: np.ndarray) -> np.ndarray:
    """QPSK como caso particular de 4-PSK."""
    return modulate_psk(bits, M=4)


# =========================
# Generador general
# =========================

def generate_baseband(modulation: str,
                      Nsym: int = 1000,
                      snr_db: float | None = None) -> dict:
    modulation = modulation.upper()

    if modulation == "ASK2":
        M = 2
        mod_fun = lambda b: modulate_ask(b, M)

    elif modulation == "ASK4":
        M = 4
        mod_fun = lambda b: modulate_ask(b, M)

    elif modulation == "BPSK":
        M = 2
        mod_fun = lambda b: modulate_psk(b, M)

    elif modulation == "QPSK":
        M = 4
        mod_fun = lambda b: modulate_psk(b, M)

    elif modulation == "PSK8":
        M = 8
        mod_fun = lambda b: modulate_psk(b, M)

    elif modulation == "QAM16":
        Mx, My = 4, 4
        M = Mx * My
        mod_fun = lambda b: modulate_qam_rect(b, Mx, My)

    elif modulation == "QAM32":
        Mx, My = 4, 8
        M = Mx * My
        mod_fun = lambda b: modulate_qam_rect(b, Mx, My)

    # 🔹 NUEVOS CASOS FSK 🔹
    elif modulation == "FSK2":
        M = 2
        mod_fun = lambda b: modulate_fsk(b, M, df_norm=0.05)

    elif modulation == "FSK4":
        M = 4
        mod_fun = lambda b: modulate_fsk(b, M, df_norm=0.04)

    else:
        raise ValueError(f"Modulación no soportada: {modulation}")

    # A partir de aquí igual que antes
    k = int(np.log2(M))
    n_bits = Nsym * k
    bits = gen_random_bits(n_bits)

    s_clean = mod_fun(bits)
    if snr_db is not None:
        s_noisy = awgn(s_clean, snr_db)
    else:
        s_noisy = s_clean.copy()

    return {
        "bits": bits,
        "s_clean": s_clean,
        "s_noisy": s_noisy,
    }


if __name__ == "__main__":
    # Prueba rápida
    out = generate_baseband("QPSK", Nsym=16, snr_db=10)
    print("bits:", out["bits"][:16])
    print("s_clean:", out["s_clean"][:8])
    print("s_noisy:", out["s_noisy"][:8])


