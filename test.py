import numpy as np
import matplotlib.pyplot as plt

cct1 = np.load("cct.npy")
cct2 = np.load("cct_one.npy")

cct1[cct1 > 5500] = 5500
cct1[cct1 < 4739.57958984375] = 4739.57958984375

cct2[cct2 > 6502.56591796875] = 6502.56591796875
cct2[cct2 < 4739.57958984375] = 4739.57958984375

plt.figure()
plt.imshow(cct1)
plt.colorbar()

plt.figure()
plt.imshow(cct2)
plt.colorbar()

plt.show()
