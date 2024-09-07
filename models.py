import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def rayleigh_cdf(t, a):
    return 1 - np.exp(-(pow(t, 2) / (2 * pow(a, 2))))

def weibull_cdf(t, a, b):
    return 1 - np.exp(-(pow(t / a, b)))

def logistic_cdf(t, a, b):
    return 1 / (1 + np.exp(-(t - a) / b))

def lyu_cdf(t, a, b): # def lyu_cdf(t, n, a, b):
    # return n * (1 - np.exp(-a * t)) / (1 + b * np.exp(-a * t))
    return 1 * (1 - np.exp(-a * t)) / (1 + b * np.exp(-a * t))

defects = [""]

with open("dataset.txt") as file:
    for line in file:
        x = line.rstrip()
        y = x.partition('\t')[0]
        defects.append(int(y))

total_defects = []
daily_defects = []
weekly_defects = []
weekly = 0
total = 0
for x in range(1,103):
    count = defects.count(x)
    weekly += count
    total += count
    daily_defects.append(count)
    total_defects.append(total)
    if(x % 7 == 0):
        weekly_defects.append(weekly)
        weekly = 0

xs = [x for x in range(len(total_defects))]
xsw = [x for x in range(len(weekly_defects))]

plt.plot(xs, total_defects)
plt.savefig('figures/total_defects.png')
plt.close()

plt.bar(xs, daily_defects)
plt.savefig('figures/daily_defects_bar.png')
plt.close()

fig, ax = plt.subplots()
ax.hist(daily_defects)
plt.savefig('figures/daily_defects_hist.png')
plt.close()

plt.bar(xsw, weekly_defects)
plt.savefig('figures/weekly_defects.png')
plt.close()

y_data = normalizeData(total_defects)

xdata = np.asarray(xs)
ydata = np.asarray(y_data)

plt.plot(xdata, ydata, 'b-', label='data')
popt, pcov = curve_fit(rayleigh_cdf, xdata, ydata, bounds=[(0.01),(np.inf)])
plt.plot(xdata, rayleigh_cdf(xdata, *popt), 'r-', label='fit: a=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('figures/fits/rayleigh.png')
plt.close()

plt.plot(xdata, ydata, 'b-', label='data')
popt, pcov = curve_fit(weibull_cdf, xdata, ydata, bounds=[(0.01, 0.01),(np.inf, np.inf)])
plt.plot(xdata, weibull_cdf(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('figures/fits/weibull.png')
plt.close()

plt.plot(xdata, ydata, 'b-', label='data')
popt, pcov = curve_fit(logistic_cdf, xdata, ydata, bounds=[(-np.inf, 0.01),(np.inf, np.inf)])
plt.plot(xdata, logistic_cdf(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('figures/fits/logistic.png')
plt.close()

plt.plot(xdata, ydata, 'b-', label='data')
popt, pcov = curve_fit(lyu_cdf, xdata, ydata)
plt.plot(xdata, lyu_cdf(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('figures/fits/lyu.png')
plt.close()
