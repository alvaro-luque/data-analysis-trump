import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os 
from math import isnan
from scipy.stats import linregress

columns = ['id', 'location', 'category',
           'date', 'repeated_ids', 'repeated_count']
df = (pd.read_csv("data_clean.csv", usecols=columns)).iloc[::-1]  # needed data sorted chronologically


def count_lies(category=None, location=None):
    '''
    The aim of this function is to count all the elements of our dataframe by days,
    given optionally the category and location

    All variables should be strings since they are stored this way in the dataframe
    '''
    years = [f'20{j}' for j in range(17, 22)]
    months = [f'0{i}' for i in range(1, 10)]+['10', '11', '12']
    days = [f'0{k}' for k in range(1, 10)]+[f'{k}' for k in range(10, 32)]
    lies = {}
    thresh = (2021-2017)*365+1*30+20  # last day of data
    thresh_down = 50  # first day of data

    if category == None and location == None:
        for year in years:
            for month in months:
                for day in days:
                    index = (int(year)-2017)*365+int(month)*30+int(day)
                    if index > thresh:
                        break
                    elif index < thresh_down:
                        continue
                    value = len(df[df['date'] == f'{month}/{day}/{year}'])
                    lies[f'{month}/{day}/{year}'] = value

                else:
                    continue
                break
            else:
                continue
            break

        return lies

    elif category == None and location != None:
        for year in years:
            for month in months:
                for day in days:
                    index = (int(year)-2017)*365+int(month)*30+int(day)
                    if index > thresh:
                        break
                    elif index < thresh_down:
                        continue
                    value = len(
                        df[(df['date'] == f'{month}/{day}/{year}') & (df['location'] == location)])
                    lies[f'{month}/{day}/{year}'] = value

                else:
                    continue
                break
            else:
                continue
            break

        return lies

    elif category != None and location == None:
        for year in years:
            for month in months:
                for day in days:
                    index = (int(year)-2017)*365+int(month)*30+int(day)
                    if index > thresh:
                        break
                    elif index < thresh_down:
                        continue
                    value = len(
                        df[(df['date'] == f'{month}/{day}/{year}') & (df['category'] == category)])
                    lies[f'{month}/{day}/{year}'] = value

                else:
                    continue
                break
            else:
                continue
            break

        return lies

    else:
        for year in years:
            for month in months:
                for day in days:
                    index = (int(year)-2017)*365+int(month)*30+int(day)
                    if index > thresh:
                        break
                    elif index < thresh_down:
                        continue

                    value = len(df[(df['date'] == f'{month}/{day}/{year}') & (
                        df['category'] == category) & (df['location'] == location)])
                    lies[f'{month}/{day}/{year}'] = value

                else:
                    continue
                break
            else:
                continue
            break

        return lies


def interevent(lie_id, save=False):
    str_format = '%m/%d/%Y'
    initial_date = df[df['id'] == lie_id]['date'].values[0]
    date = datetime.datetime.strptime(initial_date, str_format)
    repeated_ids = [int(x) for x in df[df['id'] == lie_id]['repeated_ids'].values[0].split(', ')]
    repeated_timedelta = [0]  # we append the timedelta of the first date
    for x in repeated_ids:
        repeat_date = datetime.datetime.strptime(df[df['id'] == x]['date'].values[0], str_format)
        d1, d2 = min(repeat_date, date), max(repeat_date, date)
        delta = (d2-d1).days
        if delta == 0:
            continue
        else:
            repeated_timedelta.append(delta)
    repeated_timedelta = sorted(repeated_timedelta)
    dist = np.diff(repeated_timedelta)
    days, counts=np.unique(dist, return_counts=True)
    # print(len(days), len(counts))
    slope, intercept, rvalue, pvalue, stderr=linregress(np.log(days[1::]), np.log(counts[1::]/sum(counts[1::])))
    fig, ax=plt.subplots(figsize=(8,6))
    ax.scatter(days,counts/sum(counts), label='Interevent data')
    ax.plot(days[1::],np.exp(intercept)*days[1::]**slope, ls='--', color='black',label=f'Fit, $r^2=${rvalue**2:.4f}')
    ax.set(yscale='log', xscale='log', xlabel='$\\tau$', ylabel='$P(\\tau)$', title=f'Interevent distribution for ID {lie_id}')
    ax.legend()
    if save:
        try:
            os.mkdir('interevent_figures')
            fig.savefig(f'interevent_figures/interevent_{lie_id}.pdf', dpi=200, bbox_inches='tight')
        
        except FileExistsError:
            fig.savefig(f'interevent_figures/interevent_{lie_id}.pdf', dpi=200, bbox_inches='tight')

    return slope, stderr

    
def unique_ids():
    # This function returns all unique lies in the database
    # two ids have nan as value, we need to eliminate them
    all_ids = sorted(df['id'])
    ids_clean = [int(x) for x in all_ids if not isnan(x)]
    unique_ids = []
    for item in ids_clean:
    
        try:
            repeated_ids = [int(x) for x in df[df['id'] == item]['repeated_ids'].values[0].split(', ')]
            # if this fails, it's because there is only one element in repeated_ids, an AttributeError will be raised (exception)
            
            #if no AttributeError was raised, we continue with the inspection
            check=np.array([i in unique_ids for i in repeated_ids])
            #if any element in check is True, this means that this unique id is already registered
            if check.any():
                continue
            else:
                unique_ids.append(item) #if none of the repeated is in unique, add item to the list
                
        except AttributeError:
            one_id = df[df['id'] == item]['repeated_ids'].values[0]

            if isnan(one_id):
                # there are no repeated_ids (one_id==nan), append to the unique list
                unique_ids.append(item)
            else:
                #there is only one repeated id but it is a number, it may be or may be not in the unique list
                if one_id in unique_ids:
                    continue
                else:
                    unique_ids.append(item) 
    
    # THIS FUNCTION DOESN'T WORK DUE TO AN ERROR IN THE DATABASE WHERE MANY LIES REPEATED IDS ARE WRITTEN ERRATICALLY
    return unique_ids

