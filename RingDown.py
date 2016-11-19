import matplotlib.pyplot as plt
import numpy as np
import csv, glob
from scipy.optimize import curve_fit

list_files = (glob.glob('WA*1.CSV'))

data = []

for f in list_files:
    data_iter = csv.reader(open(f, 'r'), delimiter = ',', quotechar = '"')
    for i in range(2):
        data_iter.next()
    temp_data = [value for value in data_iter]      #if ends with a number
    #temp_data = [value[:-1] for value in data_iter]#if ends with a comma
    data.extend(temp_data)

data = np.asarray(data, dtype = float)

del(temp_data, list_files, value, f)

plt.subplot(111)
plt.clf()

plt.plot(data[data[:,0]>=0,0], data[data[:,0]>=0,1], label ='mes')

def func(t, tau, A, w , a, b):
    return A * np.exp(-t/tau) * np.sin((w+a*t)*t) + b

popt, pcov = curve_fit(func, data[data[:,0]>=0,0], data[data[:,0]>=0,1], p0 = [1e-5, 1e-2, 1e5, 1e10, 1e-4], maxfev=10000)
yfit = func(data[data[:,0]>=0,0], *popt)

plt.plot(data[data[:,0]>=0,0], yfit, label ='fit')

plt.xlabel('t')
plt.ylabel('Intensity')
plt.grid(which='both')
plt.legend()
plt.show()

tau = float(popt[0])
Q = np.pi*299792458*tau/(2.*140e-3)

#print(np.sqrt(np.diag(pcov)))
print(tau, Q)
