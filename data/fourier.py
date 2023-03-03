import numpy as np

def fourier(freqs,times,data):

 fourspec = np.zeros(len(freqs),dtype=float)

 for i in np.arange(len(freqs)):
   sinterm = np.sum(np.sin(-2.*np.pi*freqs[i]*times)*data)
   costerm = np.sum(np.cos(-2.*np.pi*freqs[i]*times)*data)
   fourspec[i] = 2.*np.sqrt(sinterm**2+costerm**2)/len(data)

 return fourspec
