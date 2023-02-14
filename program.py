# Data analysis project TESTS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, ccf
from itertools import combinations
from scipy.special import binom
import networkx as nx
import matplotlib as mpl
import os
from functions import count_lies, interevent

columns = ['id', 'location', 'category',
           'date', 'repeated_ids', 'repeated_count']
df = (pd.read_csv("data_clean.csv", usecols=columns)).iloc[::-1]  # needed data sorted chronologically

#%% ACF
everything = count_lies(category='Election')
data = np.array(list(everything.values()))

fig, ax=plt.subplots(figsize=(8,6))
plot_acf(data, ax=ax, adjusted=True, lags=80, title='Autocorrelation for Economy')
ax.set(ylim=[-0.25,1.1], xlabel='$\\tau$', ylabel='$ACF(\\tau)$')
# fig.savefig('autocorrelation_economy.pdf', dpi=200, bbox_inches='tight')


#%% Interevent distribution for some repeated lies
cool_ids=[29704, 30973, 30995, 31608]

for lie in cool_ids:
    slope, stderr, npoints=interevent(lie, save=True)
    print(f'Parameters for ID {lie}: {slope}+-{stderr}; {npoints} points')


#%% Cross-correlations
n_data = 1489  # I've done this too many times
categories = list(np.unique(list(df['category'])))
# some categories have value 'nan', and are at the end of the list, we remove them
categories.pop()
ts_all = np.zeros((len(categories), n_data))
cross_corr = np.zeros((int(binom(len(categories), 2)), n_data))
dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[i] = category
    ts_all[i] = list(count_lies(category).values())

comb=combinations(range(len(categories)), 2) 
#combinations returns an iterator that "runs out" after looping through it, we will need to create it every time we need it
for i, item in enumerate(comb):
    cross_corr[i] = ccf(ts_all[item[0]], ts_all[item[1]])


comb=combinations(range(len(categories)), 2)
os.mkdir("ccf_figures") #if folder already exists this will give FileExistsError, please delete the folder and try again
for i, item in enumerate(comb):
	fig, ax=plt.subplots(figsize=(8,6))
	ax.set(title=f'CCF for {categories[item[0]]}, {categories[item[1]]}', xlabel='$\\tau$', ylabel='$CCF(\\tau)$')
	ax.plot(cross_corr[i])
	fig.savefig(f"ccf_figures/ccf_{item[0]}_{item[1]}.pdf", dpi=200, bbox_inches='tight')
	plt.close()
	
#%% Cross-correlation threshold analysis
thresh = np.arange(0.1, 1.02, 0.02)

# para tau=0
groups_tau0 = np.zeros(len(thresh))
for j, value in enumerate(thresh):
    ccf_bin = np.where(np.abs(cross_corr[:, 0]) > value, 1, 0)
    adj = np.zeros((len(categories), len(categories)))
    comb = combinations(range(len(categories)), 2)
    for i, item in enumerate(comb):
        adj[item[0]][item[1]] = ccf_bin[i]
        adj[item[1]][item[0]] = adj[item[0]][item[1]]

    G = nx.convert_matrix.from_numpy_array(adj)
    nx.set_node_attributes(G, dict_categories, name='Category')
    lbls = {j: G.nodes[j]['Category'][:3] for j in range(len(G.nodes))}
    circ = nx.circular_layout(G)
    spr = nx.spring_layout(G)
    shell = nx.shell_layout(G)
    spec = nx.spectral_layout(G)
    kk = nx.kamada_kawai_layout(G)
    spir = nx.spiral_layout(G)
    rand = nx.random_layout(G)
    groups_tau0[j] = len([len(c) for c in nx.connected_components(G)])
    # plt.figure()
    # plt.title(f"Correlation network at $\\tau=0$ with threshold $C={np.round(value,2):.2f}$")
    # nx.draw_networkx_nodes(G, circ, node_color=list(
    #     dict_categories.keys()), cmap=mpl.colormaps['tab20'], node_size=350)
    # nx.draw_networkx_edges(G, circ)
    # nx.draw_networkx_labels(G, circ, lbls)
    # plt.savefig(f"gif_tau0/tau0_{np.round(value,2):.2f}.png", dpi=200, bbox_inches='tight')
    # plt.close()

plt.figure()
plt.title("Number of connected components at $\\tau=0$")
plt.xlabel("Cross-correlation threshold $C^t$")
plt.ylabel('Number of connected components')
plt.yticks(ticks=np.unique(groups_tau0))
plt.xticks(list(plt.xticks()[0]+max(thresh[groups_tau0 == 1])))
plt.scatter(thresh, groups_tau0)
plt.savefig("componentes.pdf", dpi=200, bbox_inches='tight')

cstar = max(thresh[groups_tau0 == 1])

# %% Significance of threshold
tsz1 = ts_all[2]
tsz2 = ts_all[11]  # estas dos series dan un C~0.31 a tau=0
rng = np.random.default_rng()

n_shuff = 100000
C = np.zeros(n_shuff)

for i in range(n_shuff):
    rng.shuffle(tsz1)
    rng.shuffle(tsz2)
    C[i] = ccf(tsz1, tsz2)[0]

zscore = (cstar-C.mean())/C.std()
print(f"The Z-score is: {zscore}")
# %% Critical threshold, vary tau
categories = list(np.unique(list(df['category'])))
categories.pop()
dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[i] = category
    ts_all[i] = list(count_lies(category).values())
    #%%
groups = np.zeros(cross_corr.shape[1])
os.mkdir("gif_tau_bien")
for time in range(cross_corr.shape[1]):
    ccf_bin = np.where(np.abs(cross_corr) > cstar, 1, 0)
    adj = np.zeros((len(categories), len(categories)))
    comb = combinations(range(len(categories)), 2)

    for i, item in enumerate(comb):
        adj[item[0]][item[1]] = ccf_bin[i, time]
        adj[item[1]][item[0]] = adj[item[0]][item[1]]

    G = nx.convert_matrix.from_numpy_array(adj)
  
    groups[time] = len([len(c) for c in nx.connected_components(G)])
    
    nx.set_node_attributes(G, dict_categories, name='Category')
    lbls = {j: G.nodes[j]['Category'][:3] for j in range(len(G.nodes))}
    circ = nx.circular_layout(G)
    kk = nx.kamada_kawai_layout(G)
    plt.figure()
    plt.title(f'Network at lag $\\tau=${time}')
    nx.draw_networkx_nodes(G,kk, node_color=list(
       dict_categories.keys()), cmap=mpl.colormaps['tab20'], node_size=350)
    nx.draw_networkx_edges(G, kk)
    nx.draw_networkx_labels(G, kk,lbls)
    plt.savefig(f'gif_tau_bien/network_{time:04d}.png', dpi=200, bbox_inches='tight')
    plt.close()

# plt.figure()
# plt.title("Number of connected components at $C^*$")
# plt.ylabel("Number of components")
# plt.xlabel("$\\tau$")
# plt.yticks(np.arange(0,len(categories)+2,2))
# plt.plot(groups[:365*3])
# for j in range(1, 4):
#     plt.axvline(365*j, ls='--', c='red')

# plt.savefig("cstar.pdf", dpi=200, bbox_inches='tight')
# %% Stationarity
categories = list(np.unique(list(df['category'])))
categories.pop()

#for all the data

everything = count_lies()
data = np.array(list(everything.values()))
res = adfuller(data)

# Printing the p-value of the ADF test 
print('p-value for all the data: %f' % res[1])

#repeat for every category
for item in categories:
    everything = count_lies(category=item)
    data = np.array(list(everything.values()))
    res = adfuller(data)
    
    # Printing the p-value of the ADF test 
    print(f'p-value for category {item}: %f' % res[1])



# fig, ax=plt.subplots(figsize=(8,6))
# ax.plot(data)
# ax.set(xlabel='Time (days)', ylabel='Number of lies', title='Time series, whole data set')
# fig.savefig('whole_data.pdf', dpi=200, bbox_inches='tight')
