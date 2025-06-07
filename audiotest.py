import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from glob import glob

import librosa
import librosa.display

a4 = 'audio/A4.wav'

a4, sr = librosa.load(a4)

N = len(a4)
print(N, sr)

ft = np.fft.fft(a4)
magnitude = np.abs(ft)
frequency = np.arange(N//2+1) * sr / N
# frequency = np.linspace(0, sr, N//2 + 1)

print(np.arange(N//2 + 1) * sr / N)

print(np.max(magnitude), np.argmax(magnitude))

plt.plot(frequency, magnitude[:N//2+1])

plt.show()
