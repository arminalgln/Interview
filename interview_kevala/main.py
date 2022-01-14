################
# Q1 block 1
################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Reading data and change the index to datetime
path_to_file = 'interview_kevala/data/cleaned_hourly_2012_2014_150_sps.csv'
data = pd.read_csv(path_to_file)
print(data.shape)
print(data.head(5))
data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S')
data.set_index('Unnamed: 0', inplace=True)

# Plot some of the AMIs data
k = data.keys()
plt.ioff()
# ami = 'MT_001'
AMIs = [1, 2, 10, 15, 20]
for i in AMIs:
    plt.plot(data[k[i]].values)
    plt.title('Load for AMI {}'.format(k[i]))
    plt.xlabel('Hours')
    plt.ylabel('Power consumption (kW)')
    plt.show()

# min max mean values for some AMIs
window = ['D', 'W', 'M']

for j in AMIs:
    ami = k[j]
    fig, (fig1, fig2, fig3) = plt.subplots(3, 1)
    list_of_figs = [fig1, fig2, fig3]
    for i, w in enumerate(window):
        fg = list_of_figs[i]
        fg.plot(data[ami].resample(w).min())
        fg.plot(data[ami].resample(w).mean())
        fg.plot(data[ami].resample(w).max())
    plt.show()
#%%
################
# Q1 block 2
################
# plot mean value of all AMIs
plt.scatter(data.mean(), data.max())
plt.title('mean vs. max values for all AMIs')
plt.xlabel('Mean values for whole time span')
plt.ylabel('Max values for whole time span')
plt.show()

#plot mean value for each AMI
plt.plot(data.mean().values)
plt.show()
#%%
################
# Q1 block 3
################
#anomalies
anomalies =[]
for ami in k:
    if 0 in data[ami].resample('W').mean().values:
        anomalies.append(ami)
print(len(anomalies))
print(anomalies)

#%%
################
# Q2 block
################
ami = 'MT_128'
ts = data[ami].loc['2013-06-01':'2013-06-03']
def daily_max(ts):
    return ts.resample('D').max()

print(daily_max(ts))
print(daily_max(data.loc['2013-06':'2013-09']))

#%%
################
# Q3 block 1
################
ami = 'MT_028'
ts = data[ami].loc['2013-06':'2013-09']
for i in range(ts.shape[0]//24-1):
    plt.plot(ts[i*24:(i+1)*24].values)
plt.title('hourly data for some consecutive days for AMI {}'.format(ami))
plt.xlabel('hours')
plt.ylabel('power consumption kW')
plt.show()

def daily_mean(ts):
    return ts.resample('D').mean()

dm = daily_max(ts)
x = np.arange(0,7)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 8)
for i in range(dm.shape[0]//7-1):
    plt.plot(dm[i*7:(i+1)*7].values, '*')
plt.ylim([ts.min(), ts.max()])
labels = list(dm[i*7:(i+1)*7].index.day_name())
plt.xticks(x, labels, rotation='vertical')
plt.title('daily mean power consumption for AMI {}'.format(ami))
plt.xlabel('days')
plt.ylabel('daily mean values for power consumption')
plt.show()

#%%
################
# Q3 block 2
################
def highest_coeffs(ts, number_of_indexes, number_of_days_shift, max_mean):
    dmmax = daily_max(ts)
    if max_mean == 'max':
        dm = daily_max(ts)
    elif max_mean == 'mean':
        dm = daily_mean(ts)

    shifts = []
    for sh in range(1, number_of_days_shift):
        shifted_dm = dm.shift(sh)
        cr = np.corrcoef(dmmax[sh:].values, shifted_dm[sh:].values)[0, 1]
        shifts.append(cr)
    sorted_shifts = sorted(shifts, reverse=True)[0: number_of_indexes]
    best_indexes = [shifts.index(i) + 1 for i in sorted_shifts]

    return dm, best_indexes

#highest correlation in max values
start = '2013-01'
end = '2013-12'
# best_shifted_days = [1, 2, 3, 7, 14, 21]
ami = 'MT_028'
ts = data[ami].loc[start:end]
dm, best_indexes = highest_coeffs(ts, 3, 30, 'max')

#same day previous week max
shifts = best_indexes
print(best_indexes)

for sh in shifts:
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 8)
    shifted_dm = dm.shift(sh)
    plt.scatter(dm[sh:], shifted_dm[sh:])
    plt.title('MAX values with previous {} days, correlation between the features is : {}'
              .format(sh, np.corrcoef(dm[sh:].values, shifted_dm[sh:].values)[0, 1]))
    plt.xlabel('future max value')
    plt.ylabel('previous {} days max value'.format(sh))
    plt.show()

#highest correlation in mean values
dm, best_indexes = highest_coeffs(ts, 3, 30, 'mean')
shifts = best_indexes
print(best_indexes)
for sh in shifts:
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 8)
    shifted_dm = dm.shift(sh)
    plt.scatter(dm[sh:], shifted_dm[sh:])
    plt.title('MEAN values with previous {} days, correlation between the features is : {}'
              .format(sh, np.corrcoef(dm[sh:].values, shifted_dm[sh:].values)[0, 1]))
    plt.xlabel('future max value')
    plt.ylabel('previous {} days max value'.format(sh))
    plt.show()


#%%
################
# Q3 block 3
################
start = '2012'
end = '2014'
coeffs = {}
all_best_indexes = {'max': [], 'mean': []}
for ami in k:
    coeffs[ami] = {}
    ts = data[ami].loc[start:end]
    for feat in ['mean', 'max']:
        coeffs[ami][feat] = {}
        dm, best_indexes = highest_coeffs(ts, 3, 30, feat)
        coeffs[ami][feat]['indexes'] = best_indexes
        all_best_indexes[feat].append(best_indexes)
        cfs = []
        for sh in best_indexes:
            shifted_dm = dm.shift(sh)
            cfs.append(np.corrcoef(dm[sh:].values, shifted_dm[sh:].values)[0, 1])
        coeffs[ami][feat]['cfs'] = cfs

for feat in ['mean', 'max']:
    all_best_indexes[feat] = np.array(all_best_indexes[feat]).ravel()
    values, counts = np.unique(all_best_indexes[feat], return_counts=True)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 8)
    plt.plot(values, counts, 'o')
    plt.title('Number of specific shifted days window observed in highest coeff of daily {} AMIs data'.format(feat))
    plt.ylabel('number of times repeated')
    plt.xlabel('number of days shifted')
    plt.show()


#%%
################
# Q3 block 4
################
start = '2012-01'
end = '2014'
# best_shifted_days = [1, 2, 3, 7, 14, 21]
ami = 'MT_028'
ts = data[ami].loc[start:end]
dm = daily_max(ts)
dm_season = dm.groupby(dm.index.month%12 // 3 + 1)
dm_month = dm.groupby(dm.index.month)
N = 5
for i in range(N):
    plt.plot([1, 2, 3, 4],[dm_season.get_group(j+1).quantile(i/(N-1)) for j in range(len(dm_season))])
plt.legend(['Quantile {}'.format(i/(N-1)) for i in range(N)])
plt.title('seasonal relationship, different quantiles and daily max values')
plt.xlabel('seasons')
plt.ylabel('max values')
plt.show()

for i in range(N):
    plt.plot(np.arange(1,13) ,[dm_month.get_group(j+1).quantile(i/(N-1)) for j in range(len(dm_month))])
plt.legend(['Quantile {}'.format(i/(N-1)) for i in range(N)])
plt.title('monthly relationship, different quantiles and daily max values')
plt.xlabel('months')
plt.ylabel('max values')
plt.show()

