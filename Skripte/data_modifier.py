import pandas as pd 
import csv 
from scipy.io import loadmat 
import os
from os.path import isfile, join

def convert_all_files(path, dest_direc):
    """
    converts all given .mat files into csv-files
    """
    populations = set()
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in files:
        if "_class.mat" in i:
            populations.add(i[:-10])

        if "_lact.mat" in i:
            populations.add(i[:-9])
    
    # removing dubs
    populations = list(populations)

    # Converting all to csv
    for i in populations:
        path_lact = path + "\\" + i + "_lact.mat"
        path_class = path + "\\" + i + "_class.mat"
        convert_data(path_lact, path_class, i, dest_direc)


def convert_data(path_lact, path_class, name, dest):
    # Extracting Header Info: (day, trial)
    mat = loadmat(path_class)
    days = mat['class_all_days'][0]
    trials = mat['class_stim_evo'][0]
    # merging to (day, trial) - Tupel
    header = list(map(lambda d, t:(d,t), days, trials))
    header.insert(0, "Neuron_ID")

    # Extracting neuronal response
    mat = loadmat(path_lact)
    data = []
    for i in range(len(mat['vecs_all_days'])):
        data.append([i])
        for j in range(len(mat['vecs_all_days'][i])):
            data[i].append(mat['vecs_all_days'][i][j])

    # Saving it to csv-file
    data.insert(0, header)
    with open(dest + "\\" + name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    

def del_files(path):
    """
    deletes unneeded matlab files
    """
    onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in range(len(onlyfiles) -1, -1, -1):
        if ".mat" in onlyfiles[i]:
            if "_lact.mat" in onlyfiles[i] or "_class.mat" in onlyfiles[i]:
                pass
            else:
                os.remove(onlyfiles[i])

if __name__ == "__main__":
    convert_all_files(r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten", r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten_CSV")
    
