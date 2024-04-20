import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

sampFreq, sound = wavfile.read('noise_a3s.wav')

sound.dtype, sampFreq

sound = sound / 2.0**15

sound.shape

length_in_s = sound.shape[0] / sampFreq
print(length_in_s)

plt.subplot(2,1,1)
plt.plot(sound[:,0], 'r')
plt.xlabel("left channel, sample #")
plt.subplot(2,1,2)
plt.plot(sound[:,1], 'b')
plt.xlabel("right channel, sample #")
plt.tight_layout()
plt.show()

time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s

plt.subplot(2,1,1)
plt.plot(time, sound[:,0], 'r')
plt.xlabel("time, s [left channel]")
plt.ylabel("signal, relative units")
plt.subplot(2,1,2)
plt.plot(time, sound[:,1], 'b')
plt.xlabel("time, s [right channel]")
plt.ylabel("signal, relative units")
plt.tight_layout()
plt.show()

signal = sound[:,0]

plt.plot(time[6000:7000], signal[6000:7000])
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show()

fft_spectrum = np.fft.rfft(signal)
freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)

fft_spectrum

fft_spectrum_abs = np.abs(fft_spectrum)

plt.plot(freq, fft_spectrum_abs)
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()

plt.plot(freq[:3000], fft_spectrum_abs[:3000])
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()

plt.plot(freq[:500], fft_spectrum_abs[:500])
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.arrow(90, 5500, -20, 1000, width=2, head_width=8, head_length=200, fc='k', ec='k')
plt.arrow(200, 4000, 20, -1000, width=2, head_width=8, head_length=200, fc='g', ec='g')
plt.show()

for i,f in enumerate(fft_spectrum_abs):
    if f > 350: #looking at amplitudes of the spikes higher than 350 
        print('frequency = {} Hz with amplitude {} '.format(np.round(freq[i],1),  np.round(f)))

for i,f in enumerate(freq):
    if f < 62 and f > 58:# (1)
        fft_spectrum[i] = 0.0
    if f < 21 or f > 20000:# (2)
        fft_spectrum[i] = 0.0

plt.plot(freq[:3000], np.abs(fft_spectrum[:3000]))
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()

noiseless_signal = np.fft.irfft(fft_spectrum)

wavfile.write("data/noiseless_a3s.wav", sampFreq, noiseless_signal)
