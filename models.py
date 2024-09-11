import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
import numpy as np

def rayleigh_cdf(t, n, a):
    return n * (1 - np.exp(-(pow(t, 2) / (2 * pow(a, 2)))))

def weibull_cdf(t, n, a, b):
    return n * (1 - np.exp(-(pow(t / a, b))))

def logistic_cdf(t, n, a, b):
    return n * (1 / (1 + np.exp(-(t - a) / b)))

def lyu_cdf(t, n, a, b):
    return n * (1 - np.exp(-a * t)) / (1 + b * np.exp(-a * t))

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
plt.title('Cumulative Amount of Defects')
plt.xlabel('Days')
plt.ylabel('Defects')
plt.savefig('figures/total_defects.png')
plt.close()

plt.bar(xs, daily_defects)
plt.title('Daily Amount of Defects')
plt.xlabel('Days')
plt.ylabel('Defects')
plt.savefig('figures/daily_defects_bar.png')
plt.close()

plt.hist(daily_defects, bins='auto')
plt.title("Histogram of Daily Defects")
plt.xlabel('Defects')
plt.ylabel('Days')
plt.savefig('figures/daily_defects_hist.png')
plt.close()

plt.bar(xsw, weekly_defects)
plt.title('Weekly Amount of Defects')
plt.xlabel('Weeks')
plt.ylabel('Defects')
plt.savefig('figures/weekly_defects.png')
plt.close()

xdata = np.asarray(xs)
ydata = np.asarray(total_defects)

fit_x = np.arange(300)

popt, pcov, infodict, mesg, ier = curve_fit(rayleigh_cdf, xdata, ydata, bounds=[(0.001, 0.001), (np.inf, np.inf)], full_output=True)
fit_y = rayleigh_cdf(fit_x, *popt)
day_of_x_defects = fit_x[fit_y.searchsorted(0.98*popt[0], 'left')]
plt.plot(xdata, ydata, 'b-', label='data')
plt.axhline(y=0.98*popt[0], color='g', linestyle='-')
plt.axvline(x=day_of_x_defects, color='y', linestyle='-')
plt.text(day_of_x_defects + 12, popt[0]-130, f'{day_of_x_defects}', color='y', fontsize=12, ha='center', va='bottom')
plt.plot(fit_x, fit_y, 'r-', label='fit: n=%5.0f, a=%5.3f' % (round(popt[0]), popt[1]))
plt.title('Rayleigh model')
plt.xlabel('Days')
plt.ylabel('Cumulative sum of defects')
plt.legend()
plt.savefig('figures/fits/rayleigh.png')
plt.close()

residuals = infodict.get("fvec")
plt.plot(xdata,residuals, 'o')
plt.axhline(y=0, color='k', linestyle='--')
ax = plt.gca()
ax.set_ylim([-75, 75])
plt.title('Residuals Rayleigh')
plt.xlabel('Days')
plt.ylabel('Residual')
plt.savefig('figures/residuals/rayleigh_r.png')
plt.close()

ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

popt, pcov, infodict, mesg, ier = curve_fit(weibull_cdf, xdata, ydata, bounds=[(0.001, 0.001, 0.001),(np.inf, np.inf, np.inf)], full_output=True)
fit_y = weibull_cdf(fit_x, *popt)
day_of_x_defects = fit_x[fit_y.searchsorted(0.98*popt[0], 'left')]
plt.plot(xdata, ydata, 'b-', label='data')
plt.axhline(y=0.98*popt[0], color='g', linestyle='-')
plt.axvline(x=day_of_x_defects, color='y', linestyle='-')
plt.text(day_of_x_defects + 12, popt[0]-170, f'{day_of_x_defects}', color='y', fontsize=12, ha='center', va='bottom')
plt.plot(fit_x, fit_y, 'r-', label='fit: n=%5.0f, a=%5.3f, b=%5.3f' % (round(popt[0]), popt[1], popt[2]))
plt.title('Weibull model')
plt.xlabel('Days')
plt.ylabel('Cumulative sum of defects')
plt.legend()
plt.savefig('figures/fits/weibull.png')
plt.close()

residuals = infodict.get("fvec")
plt.plot(xdata,residuals, 'o')
plt.axhline(y=0, color='k', linestyle='--')
ax = plt.gca()
ax.set_ylim([-75, 75])
plt.title('Residuals Weibull')
plt.xlabel('Days')
plt.ylabel('Residual')
plt.savefig('figures/residuals/weibull_r.png')
plt.close()

ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

popt, pcov, infodict, mesg, ier = curve_fit(logistic_cdf, xdata, ydata, bounds=[(0.001, -np.inf, 0.001),(np.inf, np.inf, np.inf)], full_output=True)
fit_y = logistic_cdf(fit_x, *popt)
day_of_x_defects = fit_x[fit_y.searchsorted(0.98*popt[0], 'left')]
plt.plot(xdata, ydata, 'b-', label='data')
plt.axhline(y=0.98*popt[0], color='g', linestyle='-')
plt.axvline(x=day_of_x_defects, color='y', linestyle='-')
plt.text(day_of_x_defects + 12, popt[0]-130, f'{day_of_x_defects}', color='y', fontsize=12, ha='center', va='bottom')
plt.plot(fit_x, fit_y, 'r-', label='fit: n=%5.0f, a=%5.3f, b=%5.3f' % (round(popt[0]), popt[1], popt[2]))
plt.title('Logistic model')
plt.xlabel('Days')
plt.ylabel('Cumulative sum of defects')
plt.legend()
plt.savefig('figures/fits/logistic.png')
plt.close()

residuals = infodict.get("fvec")
plt.plot(xdata,residuals, 'o')
plt.axhline(y=0, color='k', linestyle='--')
ax = plt.gca()
ax.set_ylim([-75, 75])
plt.title('Residuals Logistic')
plt.xlabel('Days')
plt.ylabel('Residual')
plt.savefig('figures/residuals/logistic_r.png')
plt.close()

ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

popt, pcov, infodict, mesg, ier = curve_fit(lyu_cdf, xdata, ydata, bounds=[(0.001, -np.inf, -np.inf),(np.inf, np.inf, np.inf)], full_output=True)
fit_y = lyu_cdf(fit_x, *popt)
day_of_x_defects = fit_x[fit_y.searchsorted(0.98*popt[0], 'left')]
plt.plot(xdata, ydata, 'b-', label='data')
plt.axhline(y=0.98*popt[0], color='g', linestyle='-')
plt.axvline(x=day_of_x_defects, color='y', linestyle='-')
plt.text(day_of_x_defects + 12, popt[0]-150, f'{day_of_x_defects}', color='y', fontsize=12, ha='center', va='bottom')
plt.plot(fit_x, fit_y, 'r-', label='fit: n=%5.0f, a=%5.3f, b=%5.3f' % (round(popt[0]), popt[1], popt[2]))
plt.title('Lyu model')
plt.xlabel('Days')
plt.ylabel('Cumulative sum of defects')
plt.legend()
plt.savefig('figures/fits/lyu.png')
plt.close()

residuals = infodict.get("fvec")
plt.plot(xdata,residuals, 'o')
plt.axhline(y=0, color='k', linestyle='--')
ax = plt.gca()
ax.set_ylim([-75, 75])
plt.title('Residuals Lyu')
plt.xlabel('Days')
plt.ylabel('Residual')
plt.savefig('figures/residuals/lyu_r.png')
plt.close()

ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)