import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

from scipy import signal

# Parametros 7: f1 = 5.7 kHz, f2 = 6 kHz, δ2 = 0.1

# Noise function
def noise_audio(voice, a1, a2):
    n = np.arange(len(voice))  # Time axis
    noise = a1 * np.cos(0.76 * np.pi * n) + a2 * np.cos(0.8 * np.pi * n)
    noisy_voice = noise + voice
    return noisy_voice  # Return audio with noise


# Applying filters
def butterworth_filter(noisy_voice, cutoff, sample_rate, order):
    b, a = signal.butter(N=order, Wn=cutoff, fs=sample_rate, btype='low', analog=False)
    y = signal.lfilter(b, a, noisy_voice)
    return y


def chebyshev1_filter(noisy_voice, cutoff, sample_rate, order, ripple):
    b, a = signal.cheby1(order, ripple, cutoff, fs=sample_rate, btype='low', analog = False)
    y = signal.lfilter(b, a, noisy_voice)
    return y


def chebyshev2_filter(noisy_voice, cutoff, sample_rate, order, stop_attenuation):
    b, a = signal.cheby2(order, stop_attenuation, cutoff, fs=sample_rate, btype='low', analog=False)
    y = signal.lfilter(b, a, noisy_voice)
    return y


def eliptic_filter(noisy_voice, cutoff, sample_rate, order, stop_attenuation, ripple):
    b, a = signal.ellip(order, ripple, stop_attenuation, cutoff, fs=sample_rate, btype = 'low', analog = False)
    y = signal.lfilter(b, a, noisy_voice)
    return y


# Set-up
recording_time = 5  # seconds
sample_rate = 44100  # sample rate(Hz)
std_audio = "Audio padrao IIR"  # Audio gravado
noisy_audio = "Audio com Ruído IIR"  # Audio com ruido
butterworth_filtred_audio = "Audio filtrado com Butterworth"  # Butterworth
filtred_audio_chebyshev_I = "Audio filtrado com Chebyshev I"  # Chebyshev I
filtred_audio_chebyshev_II = "Audio filtrado com Chebyshev II"  # Chebyshev II
filtred_audio_eliptic = "Audio filtrado com Elíptico"  # Elíptico
a1 = 0.01
a2 = 0.01

# IIR filter settings
order = 4
cutoff = 1000  # Cut-Off Frequency (rad)
ripple = 3  # Ripple (dB)
stop_attenuation = 40  # Minimum attenuation within the rejection band (dB)

# Audio recording
print("Gravando..")
voice = sd.rec(int(recording_time * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

# Noisy audio
noisy_voice = noise_audio(voice.flatten(), 0.7, 0.7)

# Applying butteworth filter
btw_filtered_sign = butterworth_filter(noisy_voice, cutoff, sample_rate, order)

# Applying chebyshev I filter
cheb1_filtered_sign = chebyshev1_filter(noisy_voice, cutoff, sample_rate, order, ripple)

# Applying chebyshev II filter
cheb2_filtered_sign = chebyshev2_filter(noisy_voice, cutoff, sample_rate, order, stop_attenuation)

# Applying eliptic filter
elptc_filtered_sign = eliptic_filter(noisy_voice, cutoff, sample_rate, order, stop_attenuation, ripple)

# Save WAV file
file_name_wav1 = noisy_audio + ".wav"
sf.write(file_name_wav1, noisy_voice, sample_rate)

file_name_wav2 = std_audio + ".wav"
sf.write(file_name_wav2, voice.flatten(), sample_rate)

file_name_wav3 = butterworth_filtred_audio + ".wav"
sf.write(file_name_wav3, btw_filtered_sign, sample_rate)

file_name_wav4 = filtred_audio_chebyshev_I + ".wav"
sf.write(file_name_wav4, cheb1_filtered_sign, sample_rate)

file_name_wav5 = filtred_audio_chebyshev_II + ".wav"
sf.write(file_name_wav5, cheb2_filtered_sign, sample_rate)

file_name_wav6 = filtred_audio_eliptic + ".wav"
sf.write(file_name_wav6, elptc_filtered_sign, sample_rate)

print("""Gravado com sucesso. 
Arquivo salvo com sucesso.""")

# Charts plot
t = np.linspace(0, recording_time, num=len(voice))
plt.figure(figsize=(14, 12))

# Original audio file
plt.subplot(6, 1, 1)
plt.plot(t, voice.flatten(), color="purple")
plt.title("Sinal Original")
plt.xlabel("Tempo(s)")
plt.ylabel("Amplitude")
plt.ylim(-0.20, 0.20)

# Noisy audio
plt.subplot(6, 1, 2)
plt.plot(t, noisy_voice, color="lightgreen")
plt.title("Sinal com Ruído")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")

# Butterworth filtered audio
plt.subplot(6, 1, 3)
plt.plot(t, btw_filtered_sign, color="lightblue")
plt.title("Sinal Filtrado - Butterworth")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.ylim(-0.20, 0.20)

# Chebyshev I filtered audio
plt.subplot(6, 1, 4)
plt.plot(t, cheb1_filtered_sign, color="red")
plt.title("Sinal Filtrado - Chebyshev I")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.ylim(-0.20, 0.20)

# Chebyshev II filtered audio
plt.subplot(6, 1, 5)
plt.plot(t, cheb2_filtered_sign, color="blue")
plt.title("Sinal Filtrado - Chebyshev II")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.ylim(-0.20, 0.20)

# Eliptic filtered audio
plt.subplot(6, 1, 6)
plt.plot(t, elptc_filtered_sign, color="pink")
plt.title("Sinal Filtrado - Elíptico")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.ylim(-0.20, 0.20)

plt.tight_layout()
plt.savefig("plots_IIR.png")
plt.show()
