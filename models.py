import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.stats import weibull_min
import numpy as np

def rayleigh(x, a, b):
    return a * x * np.exp(-b * pow(x, 2))

def normal(t, a, b, c):
    return np.exp(a * pow(t, 2) + b * t + c)

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
print(str(total_defects))

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
