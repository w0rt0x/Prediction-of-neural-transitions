import os
from os.path import isfile, join
from typing import Tuple
import pandas as pd
from collections import Counter


def get_all_pop(path: str=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten'):
    """
    returns all population-names
    """
    populations = set()
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in files:
        if "_class.mat" in i:
            populations.add(i[:-10])

        if "_lact.mat" in i:
            populations.add(i[:-9])
    return list(populations)


def sort_out_populations(populations: list, path: str=r'D:\Dataframes\PCA\2', percent:float=0.0, num_class:int=4) ->Tuple[list, list]:
    """
    returns 2 lists (ok and not_ok), where all populations that are in the ok-list have all 4 classes
    """
    ok = []
    not_ok = []

    for pop in populations:
        df = pd.read_csv(path + '\\{}.csv'.format(pop))
        liste = df['response'].tolist()

        response = list(set(liste))
        response.remove('0')

        if len(response) != num_class:
            not_ok.append(pop)
        else:
            if percent == 0.0:
                ok.append(pop)
            else:
                occurences = Counter(liste)
                # remove day 4
                del occurences["0"]
                keys = occurences.keys()
                s = 0
                for key in keys:
                    s = s + occurences[key]
                for key in keys:
                    occurences[key] = occurences[key] / s
                    
                add = True
                for key in keys:
                    if occurences[key] < percent:
                        add = False
                if add:
                    ok.append(pop)
                else:
                    not_ok.append(pop)
    
    return ok, not_ok